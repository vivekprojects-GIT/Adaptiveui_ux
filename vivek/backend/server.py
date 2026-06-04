"""FastAPI server for the Adaptive Presentation Engine backend.

Endpoints
---------
- GET  /                : Vue SPA entry
- GET  /api/healthz     : process liveness (no external calls)
- GET  /api/health      : LLM connectivity check (use sparingly)
- GET  /api/state       : bandit posteriors
- POST /api/chat_plain  : baseline chat (no strategy selection)
- POST /api/chat        : adaptive chat (non-streaming)
- POST /api/chat_stream : adaptive chat (SSE streaming — Claude-style)
- POST /api/rate        : explicit feedback (reward update)
- POST /api/preference  : user preference warm-start
- POST /api/reset       : clear user session

SSE stream events (`evt.type`)
------------------------------
- strategy        : bandit selected strategy + posteriors (emitted first)
- response_delta  : token chunk inside <RESPONSE>
- widget_start    : model opened <WIDGET>
- widget_delta    : raw token chunk inside <WIDGET> (live, Claude-style)
- done            : final assembled payload (canonical response + polished widget)
"""

from __future__ import annotations

import json
import os
import time
import threading
import uuid
from typing import Iterable

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import config, llm
from .combined_prompt import (
    build_combined_system_prompt,
    build_combined_user_prompt,
    extract_embeddable_html_document,
    finalize_widget_schema_json,
    is_social_or_greeting_turn,
    parse_combined_output,
    widget_schema_json_is_valid,
)
from .engine import engine, USERB_ID
from .auth import (
    authenticate_login,
    create_access_token,
    decode_access_token,
    get_user_by_id,
    get_user_by_email,
    register_user,
    seed_users_from_db,
    update_password,
)
from . import db as persistence
from .utils import (
    detect_explore_trigger,
    enforce_response,
    fast_valence,
    negative_strength,
)
from .widget_prompt import estimate_widget_height, inject_design_system
from .widget_stream import parse_combined_stream


# ---------------------------------------------------------------------------
# Enforcement helpers
# ---------------------------------------------------------------------------

def _maybe_enforce_primitive(user_message: str, strategy: str, response: str) -> str:
    """Apply strict format except for short greeting/thanks-only turns."""
    if not config.STRICT_PRIMITIVES or not response:
        return response
    if is_social_or_greeting_turn(user_message):
        return response
    return enforce_response(strategy, response)


# ---------------------------------------------------------------------------
# Auth plumbing
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)

_password_reset_lock = threading.Lock()
_password_reset_tokens: dict[str, dict] = {}
_PASSWORD_RESET_TTL_SECONDS = int(os.getenv("PASSWORD_RESET_TTL_SECONDS", "900"))


def require_user_id(credentials=Depends(bearer_scheme)) -> str:
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = credentials.credentials
    try:
        return decode_access_token(token)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")


def _is_admin_user(user_id: str) -> bool:
    """
    Admin check. Previous behavior silently promoted everyone to admin when
    ADMIN_* config was missing; that's a production footgun. We now require
    explicit admin configuration — no config means no admin.
    """
    if not getattr(config, "ADMIN_CONFIGURED", False):
        return False

    if user_id in set(map(str, getattr(config, "ADMIN_USER_IDS", []) or [])):
        return True

    rec = get_user_by_id(user_id)
    if not rec:
        return False

    username_n = (rec.username or "").strip().lower()
    email_n = (rec.email or "").strip().lower()
    return username_n in {u.lower() for u in (config.ADMIN_USERNAMES or [])} or email_n in {
        e.lower() for e in (config.ADMIN_EMAILS or [])
    }


def require_admin_user_id(user_id: str = Depends(require_user_id)) -> str:
    if not _is_admin_user(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user_id


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse_pack(evt: dict) -> str:
    """Pack a JSON event into an SSE data frame."""
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


def _looks_truncated_widget_html(html: str) -> bool:
    if not html:
        return True
    lower = html.lower()
    if lower.count("<style") > lower.count("</style>"):
        return True
    if lower.count("<script") > lower.count("</script>"):
        return True
    if lower.count("<body") > lower.count("</body>"):
        return True
    if lower.count("<html") > lower.count("</html>"):
        return True
    import re as _re
    if _re.search(r"[<{(]$", html.strip()):
        return True
    return False


def _json_layout_is_only_numeric_index_arrays(schema_str: str) -> bool:
    """
    Detect bogus JSON widgets where every block is text like '[0,1,2]' (e.g. tic-tac-toe
    win lines) instead of real UI. Those pass structural validation but are not widgets.
    """
    import json as _json
    import re as _re

    try:
        o = _json.loads(schema_str)
    except _json.JSONDecodeError:
        return False
    layout = o.get("layout")
    if not isinstance(layout, list) or len(layout) < 2:
        return False
    pat = _re.compile(r"^\s*\[\s*\d+(\s*,\s*\d+)*\s*\]\s*$")
    for item in layout:
        if not isinstance(item, dict):
            return False
        if str(item.get("type", "")).lower() != "text":
            return False
        if not pat.match(str(item.get("content", ""))):
            return False
    return True


def _dispatch_json_mode_widget(widget_payload_raw: str) -> tuple[str, str, int, str]:
    """
    STRICT components-only: always resolve to a JSON schema rendered by the registry
    components. The HTML escape-hatch is DISABLED — model-generated HTML is never
    rendered. Returns (widget_schema, widget_html, widget_height, widget_debug_tag);
    widget_html is always "".
    """
    raw = (widget_payload_raw or "").strip()
    if not raw:
        return "", "", 0, ""

    finalized = finalize_widget_schema_json(raw)
    if widget_schema_json_is_valid(finalized) and not _json_layout_is_only_numeric_index_arrays(finalized):
        return finalized, "", 0, "json_schema_ok"
    if _json_layout_is_only_numeric_index_arrays(finalized):
        return "", "", 0, "json_degenerate_layout"
    return finalized, "", 0, "json_schema_invalid"


def _should_generate_widget(message: str) -> bool:
    """Lightweight intent gate so small-talk / explainers do not force widgets."""
    import re as _re

    text = (message or "").strip().lower()
    if not text:
        return False
    low_signal = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "got it", "cool"}
    if text in low_signal:
        return False

    widget_triggers = (
        "chart", "graph", "plot", "dashboard", "table", "compare", "comparison",
        "trend", "timeseries", "time series", "distribution", "heatmap", "scatter",
        "pie", "bar", "line", "kpi", "analytics", "analyze", "analysis", "forecast",
        "breakdown", "report", "visualize", "visualise", "show me", "insight", "metrics",
        "tic tac", "tic-tac", "tictac", "game", "playable", "simulator", "write code",
        "html page", "mini app", "calculator app",
    )
    if any(t in text for t in widget_triggers):
        return True

    text_only_intents = (
        "explain", "what is", "why", "how does", "difference between", "define",
        "summarize", "summarise", "plan", "roadmap", "steps", "implementation plan",
    )
    if any(t in text for t in text_only_intents):
        return False

    has_numeric_cue = bool(_re.search(r"\b\d+(\.\d+)?%?\b", text))
    has_time_cue = any(t in text for t in ("daily", "weekly", "monthly", "quarterly", "yearly", "over time", "timeline"))
    has_compare_cue = any(t in text for t in ("vs", "versus", "compare", "top", "rank", "distribution"))
    if has_numeric_cue and (has_time_cue or has_compare_cue):
        return True

    return False


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ChatPlainReq(BaseModel):
    uid: str | None = None
    message: str


class ChatReq(BaseModel):
    uid: str | None = None
    message: str


class RateReq(BaseModel):
    uid: str | None = None
    strategy: str
    x_vec: list[float]
    reward: float


class PreferenceReq(BaseModel):
    uid: str | None = None
    strategies: list[str] = []
    lock: bool = False


class ResetReq(BaseModel):
    uid: str | None = None


class PrimitiveCreateReq(BaseModel):
    name: str
    instruction: str


class PrimitiveUpdateReq(BaseModel):
    name: str
    instruction: str


class StrategyCreateReq(BaseModel):
    id: str
    label: str
    instruction: str
    enabled: bool = True
    event_name: str = ""


class StrategyUpdateReq(BaseModel):
    label: str
    instruction: str
    enabled: bool = True
    event_name: str = ""


class AuthRegisterReq(BaseModel):
    username: str
    email: str
    password: str


class AuthLoginReq(BaseModel):
    username_or_email: str
    password: str


class ForgotPasswordReq(BaseModel):
    email: str


class ResetPasswordReq(BaseModel):
    token: str
    new_password: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Adaptive Presentation Engine")

app.add_middleware(
    CORSMiddleware,
    # allow_credentials=True is incompatible with "*" per CORS spec;
    # keep it False since we use bearer tokens, not cookies.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.on_event("startup")
def _on_startup():
    persistence.init_db()

    if getattr(config, "ADMIN_CONFIGURED", False) and config.ADMIN_USERNAME and config.ADMIN_PASSWORD and config.ADMIN_EMAIL:
        try:
            register_user(username=config.ADMIN_USERNAME, email=config.ADMIN_EMAIL, password=config.ADMIN_PASSWORD)
        except Exception:
            pass

    try:
        users = persistence.load_users_from_db()
        seed_users_from_db(users)
    except Exception:
        pass

    try:
        persistence.load_global_state(engine)
        persistence.load_user_states(engine)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Static / SPA
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = config.INDEX_HTML.read_bytes()
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


@app.get("/api/healthz")
def healthz():
    """Process liveness — cheap, no external dependencies."""
    return {"ok": True}


@app.get("/api/health")
def health():
    if config.LLM_MODE == "openai_compat":
        h = llm.openai_health()
        return {
            "server": "ok",
            "mode": config.LLM_MODE,
            "openai_base_url": config.OPENAI_BASE_URL,
            "model": config.OPENAI_MODEL,
            **h,
        }
    if config.LLM_MODE == "anthropic":
        h = llm.anthropic_health()
        return {"server": "ok", "mode": config.LLM_MODE, **h}
    return {
        "server": "ok",
        "mode": config.LLM_MODE,
        "ok": False,
        "reachable": False,
        "error": "Unsupported LLM_MODE (expected openai_compat or anthropic)",
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
def auth_register(req: AuthRegisterReq):
    try:
        rec = register_user(username=req.username, email=req.email, password=req.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    token = create_access_token(user_id=rec.user_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"user_id": rec.user_id, "username": rec.username, "email": rec.email},
    }


@app.post("/api/auth/login")
def auth_login(req: AuthLoginReq):
    rec = authenticate_login(username_or_email=req.username_or_email, password=req.password)
    if not rec:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(user_id=rec.user_id)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/auth/forgot-password")
def auth_forgot_password(req: ForgotPasswordReq):
    """Dev/demo only. Do NOT return the reset token to the caller in production."""
    email = (req.email or "").strip()
    user = get_user_by_email(email)

    if not user:
        return {"ok": True}

    token = uuid.uuid4().hex
    now = int(time.time())
    expires = now + _PASSWORD_RESET_TTL_SECONDS
    with _password_reset_lock:
        _password_reset_tokens[token] = {"user_id": user.user_id, "expires": expires}

    # Expose reset token only in development to ease local testing.
    expose_token = (os.getenv("ENV", "development") or "development").lower() != "production"
    if expose_token:
        return {"ok": True, "reset_token": token}
    return {"ok": True}


@app.post("/api/auth/reset-password")
def auth_reset_password(req: ResetPasswordReq):
    token = (req.token or "").strip()
    new_password = (req.new_password or "").strip()

    if not token or not new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="token and new_password required")

    now = int(time.time())
    with _password_reset_lock:
        rec = _password_reset_tokens.get(token)
        if not rec:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
        if int(rec.get("expires") or 0) < now:
            _password_reset_tokens.pop(token, None)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
        user_id = str(rec.get("user_id") or "")
        _password_reset_tokens.pop(token, None)

    try:
        ok = update_password(user_id=user_id, new_password=new_password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return {"ok": True}


@app.get("/api/me")
def me(user_id: str = Depends(require_user_id)):
    rec = get_user_by_id(user_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "user_id": rec.user_id,
        "username": rec.username,
        "email": rec.email,
        "is_admin": _is_admin_user(rec.user_id),
    }


# ---------------------------------------------------------------------------
# Bandit state + strategies
# ---------------------------------------------------------------------------

@app.get("/api/state")
def state(user_id: str = Depends(require_user_id)):
    uid = user_id
    x = np.ones(config.D) * 0.5
    ub = engine.get_user(USERB_ID)
    return {
        "posterior": engine.user_posterior(uid, x),
        "global": engine.global_posterior(x),
        "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
        "global_n": engine.global_n,
        "n_users": len(engine.users),
        "msg_count": engine.get_user(uid)["msg_count"],
    }


@app.get("/api/strategies")
def list_strategies(user_id: str = Depends(require_user_id)):
    admin = _is_admin_user(user_id)
    items = []
    for sid, it in getattr(config, "STRATEGY_ITEMS", {}).items():
        enabled = bool(it.get("enabled", True))
        if admin or enabled:
            items.append(
                {
                    "id": sid,
                    "label": it.get("label") or sid,
                    "instruction": it.get("instruction") or "",
                    "enabled": enabled,
                    "event_name": it.get("event_name") or "",
                }
            )
    items.sort(key=lambda x: (not bool(x.get("enabled", True)), str(x.get("id") or "")))
    return {"items": items}


@app.options("/api/strategies")
def strategies_preflight():
    return JSONResponse({"ok": True})


@app.options("/api/strategies/")
def strategies_preflight_slash():
    return JSONResponse({"ok": True})


@app.get("/api/strategies/usage")
def strategies_usage(user_id: str = Depends(require_admin_user_id)):
    return {"by_id": persistence.aggregate_strategy_usage()}


@app.get("/api/strategies/usage/")
def strategies_usage_slash(user_id: str = Depends(require_admin_user_id)):
    return strategies_usage(user_id=user_id)


@app.get("/api/strategies/{sid}/analytics")
def strategy_analytics(
    sid: str,
    days: int = 30,
    user_id: str = Depends(require_admin_user_id),
):
    sid = str(sid or "").strip()
    if not sid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sid is required")
    if sid not in getattr(config, "STRATEGY_ITEMS", {}):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return persistence.get_strategy_analytics(sid, days=days)


@app.post("/api/strategies")
def create_strategy(req: StrategyCreateReq, user_id: str = Depends(require_admin_user_id)):
    sid = str(req.id or "").strip()
    if not sid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="id is required")

    if sid in getattr(config, "STRATEGY_ITEMS", {}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="strategy id already exists")

    event_name = str(req.event_name or "").strip()
    items = list(getattr(config, "STRATEGY_ITEMS", {}).values())
    items.append({
        "id": sid,
        "label": req.label,
        "instruction": req.instruction,
        "enabled": bool(req.enabled),
        "event_name": event_name,
    })

    config.persist_strategies(items)
    engine.reconcile_strategies()
    return {
        "item": {
            "id": sid,
            "label": req.label,
            "instruction": req.instruction,
            "enabled": bool(req.enabled),
            "event_name": event_name,
        }
    }


@app.put("/api/strategies/{sid}")
def update_strategy(sid: str, req: StrategyUpdateReq, user_id: str = Depends(require_admin_user_id)):
    sid = str(sid or "").strip()
    if not sid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sid is required")

    cur = getattr(config, "STRATEGY_ITEMS", {}).get(sid)
    if not cur:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    event_name = str(req.event_name or "").strip() or str(cur.get("event_name") or "").strip()
    items = []
    for it in getattr(config, "STRATEGY_ITEMS", {}).values():
        if it.get("id") == sid:
            items.append({
                "id": sid,
                "label": req.label,
                "instruction": req.instruction,
                "enabled": bool(req.enabled),
                "event_name": event_name,
            })
        else:
            items.append(it)

    config.persist_strategies(items)
    engine.reconcile_strategies()
    return {
        "item": {
            "id": sid,
            "label": req.label,
            "instruction": req.instruction,
            "enabled": bool(req.enabled),
            "event_name": event_name,
        }
    }


@app.post("/api/strategies/{sid}/enable")
def enable_strategy(sid: str, user_id: str = Depends(require_admin_user_id)):
    sid = str(sid or "").strip()
    cur = getattr(config, "STRATEGY_ITEMS", {}).get(sid)
    if not cur:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    items = []
    for it in getattr(config, "STRATEGY_ITEMS", {}).values():
        if it.get("id") == sid:
            items.append({"id": sid, "label": it.get("label") or sid, "instruction": it.get("instruction") or "", "enabled": True})
        else:
            items.append(it)
    config.persist_strategies(items)
    engine.reconcile_strategies()
    return {"ok": True}


@app.post("/api/strategies/{sid}/disable")
def disable_strategy(sid: str, user_id: str = Depends(require_admin_user_id)):
    sid = str(sid or "").strip()
    cur = getattr(config, "STRATEGY_ITEMS", {}).get(sid)
    if not cur:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    items = []
    for it in getattr(config, "STRATEGY_ITEMS", {}).values():
        if it.get("id") == sid:
            items.append({"id": sid, "label": it.get("label") or sid, "instruction": it.get("instruction") or "", "enabled": False})
        else:
            items.append(it)
    config.persist_strategies(items)
    engine.reconcile_strategies()
    return {"ok": True}


@app.delete("/api/strategies/{sid}")
def delete_strategy(sid: str, user_id: str = Depends(require_admin_user_id)):
    sid = str(sid or "").strip()
    cur = getattr(config, "STRATEGY_ITEMS", {}).get(sid)
    if not cur:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    items = [it for it in getattr(config, "STRATEGY_ITEMS", {}).values() if it.get("id") != sid]
    if not items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete all strategies")

    config.persist_strategies(items)
    engine.reconcile_strategies()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@app.get("/api/primitives")
def list_primitives(user_id: str = Depends(require_admin_user_id)):
    return {"items": persistence.list_user_primitives(user_id)}


@app.post("/api/primitives")
def create_primitive(req: PrimitiveCreateReq, user_id: str = Depends(require_admin_user_id)):
    name = (req.name or "").strip()
    inst = (req.instruction or "").strip()
    if not name or not inst:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="name and instruction are required")
    row = persistence.create_user_primitive(user_id=user_id, name=name, instruction=inst)
    return {"item": row}


@app.put("/api/primitives/{prim_id}")
def update_primitive(prim_id: int, req: PrimitiveUpdateReq, user_id: str = Depends(require_admin_user_id)):
    name = (req.name or "").strip()
    inst = (req.instruction or "").strip()
    if not name or not inst:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="name and instruction are required")
    row = persistence.update_user_primitive(user_id=user_id, prim_id=int(prim_id), name=name, instruction=inst)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Primitive not found")
    return {"item": row}


@app.delete("/api/primitives/{prim_id}")
def delete_primitive(prim_id: int, user_id: str = Depends(require_admin_user_id)):
    ok = persistence.delete_user_primitive(user_id=user_id, prim_id=int(prim_id))
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Primitive not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

@app.get("/api/conversation")
def conversation(limit: int = 20, user_id: str = Depends(require_user_id)):
    uid = user_id
    lim = int(limit)
    lim = max(1, min(lim, 50))

    def _pairs_from_msgs(msgs: list[dict]) -> list[dict]:
        out: list[dict] = []
        last_user: str | None = None
        for m in msgs:
            if m.get("role") == "user":
                last_user = str(m.get("content") or "")
            elif m.get("role") == "assistant":
                if last_user is None:
                    continue
                out.append(
                    {
                        "user": last_user,
                        "assistant": str(m.get("content") or ""),
                        "widget_html": str(m.get("widget_html") or ""),
                        "widget_schema": str(m.get("widget_schema") or ""),
                        "widget_height": int(m.get("widget_height") or 0),
                    }
                )
                last_user = None
        return out[-lim:]

    adaptive_msgs = persistence.get_recent_conversation_messages(user_id=uid, pane="adaptive", limit=lim * 4)
    baseline_msgs = persistence.get_recent_conversation_messages(user_id=uid, pane="baseline", limit=lim * 4)

    adaptive_pairs = _pairs_from_msgs(adaptive_msgs)
    baseline_pairs = _pairs_from_msgs(baseline_msgs)

    if not adaptive_pairs:
        adaptive_user = engine.get_user(uid)
        adaptive_pairs = (adaptive_user.get("history") or [])[-lim:]
    if not baseline_pairs:
        baseline_user = engine.get_user(uid + "_plain")
        baseline_pairs = (baseline_user.get("history") or [])[-lim:]

    return {"adaptive": {"history": adaptive_pairs}, "baseline": {"history": baseline_pairs}}


# ---------------------------------------------------------------------------
# Background persistence (off the hot path)
# ---------------------------------------------------------------------------

def _persist_after_response(
    *,
    user_id: str,
    uid: str,
    msg: str,
    response: str,
    strategy: str | None,
    elapsed: float | None,
    widget_present: bool,
    pane: str,
    widget_html: str = "",
    widget_schema: str = "",
    widget_height: int = 0,
) -> None:
    """Persist bandit state + conversation log after the response has been sent.

    Runs in FastAPI BackgroundTasks so HTTP latency is not gated by SQLite commits.
    """
    try:
        persistence.persist_user_state(engine, uid)
        if pane == "adaptive":
            persistence.persist_global_state(engine)
        persistence.log_conversation_message(
            user_id=user_id, pane=pane, role="user", content=msg,
            strategy=strategy if pane == "adaptive" else None,
        )
        persistence.log_conversation_message(
            user_id=user_id, pane=pane, role="assistant", content=response,
            strategy=strategy if pane == "adaptive" else None,
            elapsed=elapsed, widget=widget_present,
            widget_html=widget_html or "",
            widget_schema=widget_schema or "",
            widget_height=int(widget_height or 0),
        )
    except Exception:
        # Persistence failures must never surface to the user.
        pass


# ---------------------------------------------------------------------------
# Baseline chat
# ---------------------------------------------------------------------------

@app.post("/api/chat_plain")
def chat_plain(req: ChatPlainReq, bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    uid = user_id + "_plain"
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    user = engine.get_user(uid)
    ctx: list[str] = []
    for t in user["history"][-6:]:
        ctx += [f"User: {t['user']}", f"Assistant: {t['assistant']}"]
    ctx.append(f"User: {msg}")
    prompt = "\n".join(ctx)
    system = "You are a helpful AI assistant."

    try:
        base_mode = (config.BASELINE_LLM_MODE or config.LLM_MODE).lower()
        if base_mode == "openai_compat":
            response, elapsed, mode = llm.call_openai_compat(prompt, system, timeout=60)
        elif base_mode == "anthropic":
            response, elapsed, mode = llm.call_anthropic(prompt, system, timeout=60)
        else:
            raise RuntimeError("Unsupported BASELINE_LLM_MODE (expected openai_compat or anthropic)")
    except Exception as e:
        return JSONResponse({"error": f"LLM error: {str(e)}"}, status_code=500)

    if not response:
        return JSONResponse({"error": "LLM returned empty response. Check model/service."}, status_code=500)

    user["history"].append({"user": msg, "assistant": response})
    user["history"] = user["history"][-20:]

    bg.add_task(
        _persist_after_response,
        user_id=user_id, uid=uid, msg=msg, response=response,
        strategy=None, elapsed=elapsed, widget_present=False, pane="baseline",
    )

    return {"response": response, "elapsed": elapsed, "llm_mode": mode}


# ---------------------------------------------------------------------------
# Adaptive chat — non-streaming
# ---------------------------------------------------------------------------

def _build_adaptive_prompt(uid: str, msg: str):
    """Shared prompt-assembly + bandit-select step for adaptive endpoints."""
    user = engine.get_user(uid)

    ev = fast_valence(msg, user["last_response"])
    auto_detected = False
    auto_r = None

    if user["last_response"] and user["last_x"] is not None and user["last_strategy"]:
        reward = float(np.clip(0.5 + 0.45 * ev["pos"] - 0.45 * ev["neg"], 0.05, 0.95))
        engine.update(uid, user["last_strategy"], np.array(user["last_x"]), reward)
        auto_detected = True
        auto_r = reward

    explicit = False
    force_explore = bool(detect_explore_trigger(msg) or (ev.get("neg", 0.0) >= config.NEG_EXPLORE_THRESHOLD))
    neg_s = negative_strength(ev)

    strat, scores, x, prev = engine.select(
        uid, msg, force_explore=force_explore, neg_strength=neg_s, explicit_strategy=None
    )

    format_rule = config.STRATEGIES.get(strat, "Be helpful and clear.")
    widget_required = _should_generate_widget(msg)

    prim_block = ""
    if _is_admin_user(uid):
        user_prims = persistence.list_user_primitives(uid)
        if user_prims:
            prim_lines = []
            for p in user_prims:
                nm = str(p.get("name") or "").strip()
                inst = str(p.get("instruction") or "").strip()
                if nm and inst:
                    prim_lines.append(f"- {nm}: {inst}")
            if prim_lines:
                prim_block = "\n\n## User primitives (follow these as constraints)\n" + "\n".join(prim_lines) + "\n"

    combined_system = build_combined_system_prompt(
        strategy_id=strat,
        format_rule=format_rule,
        primitive_extra_context=(getattr(config, "SKILLS_CONTENT", "") or "") + prim_block,
        user_message=msg,
        widget_required=widget_required,
        forbidden_components=None,
        required_components=None,
    )
    combined_prompt = build_combined_user_prompt(user_message=msg, history=user["history"])

    return {
        "user": user, "ev": ev, "auto_detected": auto_detected, "auto_r": auto_r,
        "explicit": explicit, "force_explore": force_explore,
        "strat": strat, "scores": scores, "x": x, "prev": prev,
        "format_rule": format_rule, "widget_required": widget_required,
        "combined_system": combined_system, "combined_prompt": combined_prompt,
    }


def _post_done_payload(
    *,
    uid: str,
    ctx: dict,
    elapsed: float | None,
    mode: str,
    response: str,
    widget_html: str,
    widget_schema: str,
    widget_height: int,
    widget_debug: str,
    raw_preview: str,
):
    ub = engine.get_user(USERB_ID)
    ev = ctx["ev"]
    x = ctx["x"]
    strat = ctx["strat"]
    prev = ctx["prev"]
    scores = ctx["scores"]
    return {
        "response": response,
        "strategy": strat,
        "prev_strategy": prev,
        "explicit": ctx["explicit"],
        "force_explore": ctx["force_explore"],
        "instruction": config.STRATEGIES.get(strat, ""),
        "format_rule": ctx["format_rule"],
        "elapsed": elapsed,
        "llm_mode": mode,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "x_vec": x.tolist(),
        "posterior": engine.user_posterior(uid, x),
        "global": engine.global_posterior(x),
        "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
        "global_n": engine.global_n,
        "auto_detected": ctx["auto_detected"],
        "auto_r": ctx["auto_r"],
        "auto_reason": ev["reason"],
        "widget_html": widget_html or "",
        "widget_schema": widget_schema or "",
        "widget_height": widget_height,
        "widget_debug": widget_debug,
        "widget_raw_preview": raw_preview if not (widget_html or widget_schema) else "",
    }


@app.post("/api/chat")
def chat(req: ChatReq, bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    uid = user_id
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    ctx = _build_adaptive_prompt(uid, msg)

    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 4000)
    combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 45)

    try:
        adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()
        if adapt_mode == "openai_compat":
            raw_combined, elapsed, mode = llm.call_openai_compat(
                ctx["combined_prompt"], ctx["combined_system"],
                timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2,
            )
        elif adapt_mode == "anthropic":
            raw_combined, elapsed, mode = llm.call_anthropic(
                ctx["combined_prompt"], ctx["combined_system"],
                timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2,
            )
        else:
            raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")
    except Exception as e:
        return JSONResponse({"error": f"LLM error: {str(e)}"}, status_code=500)

    if not raw_combined:
        return JSONResponse({"error": "LLM returned empty response. Check model/service."}, status_code=500)

    response, widget_payload_raw = parse_combined_output(raw_combined)
    if not response:
        response = raw_combined.strip()
    response = _maybe_enforce_primitive(msg, ctx["strat"], response)

    widget_html = ""
    widget_schema = ""
    widget_height = 0
    widget_debug = ""
    widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
    raw_preview = ""

    if widget_payload_raw:
        if widget_mode == "json":
            widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(widget_payload_raw)
            widget_debug = tag or ""
        else:
            if _looks_truncated_widget_html(widget_payload_raw):
                widget_debug = "combined_widget_truncated"
            else:
                widget_html = widget_payload_raw
                widget_height = estimate_widget_height(widget_payload_raw)
                widget_debug = "combined_widget_ok"
    else:
        widget_debug = "combined_no_schema" if widget_mode == "json" else "combined_no_widget_tag"
        raw_preview = (raw_combined or "")[:800]
        if widget_mode != "json" and ctx["widget_required"]:
            placeholder = (
                "<html><head></head><body>"
                "<div class='widget-root card'><div class='card-title'>Interactive widget</div>"
                "<div class='empty'>No widget returned.</div></div>"
                "</body></html>"
            )
            widget_html = inject_design_system(placeholder)
            widget_height = estimate_widget_height(widget_html)
            widget_debug = "fallback_widget_generated"
        elif not ctx["widget_required"]:
            widget_debug = "widget_skipped_by_intent"

    user = ctx["user"]
    user["history"].append({"user": msg, "assistant": response})
    user["history"] = user["history"][-20:]
    user["last_message"] = msg
    user["last_response"] = response
    user["last_strategy"] = ctx["strat"]
    user["last_x"] = ctx["x"].tolist()
    user["msg_count"] += 1

    payload = _post_done_payload(
        uid=uid, ctx=ctx, elapsed=elapsed, mode=mode, response=response,
        widget_html=widget_html, widget_schema=widget_schema,
        widget_height=widget_height, widget_debug=widget_debug, raw_preview=raw_preview,
    )

    bg.add_task(
        _persist_after_response,
        user_id=user_id, uid=uid, msg=msg, response=response,
        strategy=ctx["strat"], elapsed=elapsed,
        widget_present=bool(widget_html or widget_schema), pane="adaptive",
        widget_html=widget_html or "",
        widget_schema=widget_schema or "",
        widget_height=int(widget_height or 0),
    )

    return payload


# ---------------------------------------------------------------------------
# Adaptive chat — SSE streaming (Claude-style live widget streaming)
# ---------------------------------------------------------------------------

@app.post("/api/chat_stream")
def chat_stream(req: ChatReq, bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    uid = user_id
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    ctx = _build_adaptive_prompt(uid, msg)
    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 4000)
    combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 45)
    adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()

    ub = engine.get_user(USERB_ID)
    strat = ctx["strat"]
    x = ctx["x"]
    scores = ctx["scores"]
    prev = ctx["prev"]
    ev = ctx["ev"]

    def gen() -> Iterable[str]:
        t0 = time.time()

        # Initial strategy event — UI paints immediately.
        yield sse_pack({
            "type": "strategy",
            "strategy": strat,
            "instruction": config.STRATEGIES.get(strat, ""),
            "format_rule": ctx["format_rule"],
            "elapsed": None,
            "force_explore": ctx["force_explore"],
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "x_vec": x.tolist(),
            "posterior": engine.user_posterior(uid, x),
            "global": engine.global_posterior(x),
            "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
            "global_n": engine.global_n,
            "prev_strategy": prev,
            "explicit": ctx["explicit"],
            "auto_detected": ctx["auto_detected"],
            "auto_r": ctx["auto_r"],
            "auto_reason": ev["reason"],
        })

        widget_html = ""
        widget_schema = ""
        widget_height = 0
        widget_debug = ""
        widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
        raw_preview = ""
        response = ""
        mode = adapt_mode
        elapsed_out: float | None = None
        widget_payload_raw = ""

        try:
            if adapt_mode == "anthropic":
                stream = llm.stream_anthropic(
                    ctx["combined_prompt"], ctx["combined_system"],
                    timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2,
                )

                # Strict primitives: buffer response until </RESPONSE>, then emit
                # the enforced version. Otherwise stream raw tokens directly.
                emit_raw_response = not (
                    config.STRICT_PRIMITIVES and not is_social_or_greeting_turn(msg)
                )

                response_closed = False
                for event in parse_combined_stream(
                    stream,
                    emit_response_deltas=emit_raw_response,
                    emit_widget_deltas=True,
                ):
                    etype = event[0]
                    if etype == "response_delta":
                        yield sse_pack({"type": "response_delta", "delta": event[1]})
                    elif etype == "response_closed":
                        response_closed = True
                        response = _maybe_enforce_primitive(msg, strat, event[1])
                        if not emit_raw_response and response:
                            for i in range(0, len(response), 180):
                                yield sse_pack({"type": "response_delta", "delta": response[i:i + 180]})
                    elif etype == "widget_start":
                        yield sse_pack({"type": "widget_start"})
                    elif etype == "widget_delta":
                        # LIVE widget streaming — Claude-style.
                        yield sse_pack({"type": "widget_delta", "delta": event[1]})
                    elif etype == "complete":
                        if not response_closed:
                            response = _maybe_enforce_primitive(msg, strat, event[1])
                            if not emit_raw_response and response:
                                for i in range(0, len(response), 180):
                                    yield sse_pack({"type": "response_delta", "delta": response[i:i + 180]})
                        widget_payload_raw = event[2]
                        break

                elapsed_out = round(time.time() - t0, 1)
                mode = "anthropic"

            elif adapt_mode == "openai_compat":
                # No token streaming on this path; simulate progressive deltas.
                raw_combined, elapsed, mode = llm.call_openai_compat(
                    ctx["combined_prompt"], ctx["combined_system"],
                    timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2,
                )
                response, widget_payload_raw = parse_combined_output(raw_combined)
                if not response:
                    response = raw_combined.strip()
                response = _maybe_enforce_primitive(msg, strat, response)
                for i in range(0, len(response), 180):
                    yield sse_pack({"type": "response_delta", "delta": response[i:i + 180]})
                if widget_payload_raw:
                    yield sse_pack({"type": "widget_start"})
                    for i in range(0, len(widget_payload_raw), 400):
                        yield sse_pack({"type": "widget_delta", "delta": widget_payload_raw[i:i + 400]})
                elapsed_out = elapsed
            else:
                raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")

            # Finalize widget payload for the canonical 'done' event.
            if widget_payload_raw:
                if widget_mode == "json":
                    widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(widget_payload_raw)
                    widget_debug = tag or ""
                else:
                    import re as _re
                    raw_widget = widget_payload_raw
                    if "```" in raw_widget:
                        fence = _re.search(r"```(?:json|html)?\s*(.*?)```", raw_widget, _re.DOTALL | _re.IGNORECASE)
                        raw_widget = fence.group(1).strip() if fence else _re.sub(r"```\w*", "", raw_widget).strip()
                    if "<" in raw_widget and ">" in raw_widget:
                        if "<html" not in raw_widget.lower():
                            raw_widget = f"<html><head></head><body>{raw_widget}</body></html>"
                        raw_widget = _re.sub(r"<!DOCTYPE[^>]*>", "", raw_widget, flags=_re.IGNORECASE).strip()
                        widget_payload_raw = inject_design_system(raw_widget)
                        if _looks_truncated_widget_html(widget_payload_raw):
                            widget_debug = "stream_widget_truncated"
                        else:
                            widget_html = widget_payload_raw
                            widget_height = estimate_widget_height(widget_html)
            if widget_mode != "json" and not widget_html:
                widget_debug = widget_debug or "no_widget"
            if not response and not widget_payload_raw:
                response = "(No content)"

        except Exception as e:
            msg_err = str(e).strip() or repr(e)
            yield sse_pack({
                "type": "done",
                "strategy": strat,
                "format_rule": ctx["format_rule"],
                "elapsed": None,
                "llm_mode": mode,
                "response": "",
                "widget_html": "",
                "widget_schema": "",
                "widget_height": 0,
                "widget_debug": f"stream_error:{msg_err}",
                "error": msg_err,
                "force_explore": ctx["force_explore"],
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": ctx["explicit"],
                "auto_detected": ctx["auto_detected"],
                "auto_r": ctx["auto_r"],
                "auto_reason": ev["reason"],
            })
            return

        # Update in-memory history
        user = ctx["user"]
        user["history"].append({"user": msg, "assistant": response})
        user["history"] = user["history"][-20:]
        user["last_message"] = msg
        user["last_response"] = response
        user["last_strategy"] = strat
        user["last_x"] = x.tolist()
        user["msg_count"] += 1

        yield sse_pack({
            "type": "done",
            "strategy": strat,
            "format_rule": ctx["format_rule"],
            "elapsed": elapsed_out,
            "llm_mode": mode,
            "response": response,
            "widget_html": widget_html or "",
            "widget_schema": widget_schema or "",
            "widget_height": widget_height,
            "widget_debug": widget_debug,
            "force_explore": ctx["force_explore"] and (ctx["explicit"] is None),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "x_vec": x.tolist(),
            "posterior": engine.user_posterior(uid, x),
            "global": engine.global_posterior(x),
            "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
            "global_n": engine.global_n,
            "prev_strategy": prev,
            "explicit": ctx["explicit"],
            "auto_detected": ctx["auto_detected"],
            "auto_r": ctx["auto_r"],
            "auto_reason": ev["reason"],
        })

        # Persistence off the hot path
        bg.add_task(
            _persist_after_response,
            user_id=user_id, uid=uid, msg=msg, response=response,
            strategy=strat, elapsed=elapsed_out,
            widget_present=bool(widget_html or widget_schema), pane="adaptive",
            widget_html=widget_html or "",
            widget_schema=widget_schema or "",
            widget_height=int(widget_height or 0),
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Feedback / preferences / reset
# ---------------------------------------------------------------------------

@app.post("/api/rate")
def rate(req: RateReq, bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    uid = user_id
    strategy = req.strategy
    x = np.array(req.x_vec, dtype=float)
    reward = float(req.reward)
    if strategy not in config.STRATEGY_NAMES:
        return JSONResponse({"error": "bad request"}, status_code=400)
    engine.update(uid, strategy, x, reward)
    bg.add_task(persistence.persist_user_state, engine, uid)
    bg.add_task(persistence.persist_global_state, engine)
    ub = engine.get_user(USERB_ID)
    return {
        "posterior": engine.user_posterior(uid, x),
        "global": engine.global_posterior(x),
        "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
        "global_n": engine.global_n,
    }


@app.post("/api/preference")
def preference(req: PreferenceReq, bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    uid = user_id
    engine.apply_preferences(uid, req.strategies, lock=bool(req.lock))
    bg.add_task(persistence.persist_user_state, engine, uid)
    return {"posterior": engine.user_posterior(uid)}


@app.post("/api/conversation/clear")
def conversation_clear(bg: BackgroundTasks, user_id: str = Depends(require_user_id)):
    """Remove stored chat + widgets for this account; keep bandit posteriors and reward history."""
    plain_uid = user_id + "_plain"
    persistence.delete_conversation_logs(user_id=user_id, pane="adaptive")
    persistence.delete_conversation_logs(user_id=user_id, pane="baseline")
    engine.clear_conversation_thread(user_id)
    engine.clear_conversation_thread(plain_uid)
    bg.add_task(persistence.persist_user_state, engine, user_id)
    bg.add_task(persistence.persist_user_state, engine, plain_uid)
    return {"ok": True}


@app.post("/api/reset")
def reset(req: ResetReq, user_id: str = Depends(require_user_id)):
    plain_uid = user_id + "_plain"
    engine.reset_user(user_id)
    engine.reset_user(plain_uid)
    persistence.delete_user_state(user_id)
    persistence.delete_user_state(plain_uid)
    persistence.delete_conversation_logs(user_id=user_id, pane="adaptive")
    persistence.delete_conversation_logs(user_id=user_id, pane="baseline")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static assets + SPA fallback
# ---------------------------------------------------------------------------

_frontend_dist_assets_dir = config.INDEX_HTML.parent / "assets"
if _frontend_dist_assets_dir.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist_assets_dir), html=False),
        name="frontend_assets",
    )


@app.get("/{full_path:path}", response_class=HTMLResponse)
def spa_fallback(full_path: str):
    # Never serve HTML for API URLs — return JSON 404 so client parsers don't break.
    if full_path.startswith("api"):
        return JSONResponse({"detail": "Not found"}, status_code=status.HTTP_404_NOT_FOUND)
    html = config.INDEX_HTML.read_bytes()
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


def run_server():
    """Run with uvicorn (production-ready)."""
    import uvicorn

    port = int(os.getenv("PORT", "5051"))
    print("=" * 60)
    print(f"  http://localhost:{port}   mode: {config.LLM_MODE}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
