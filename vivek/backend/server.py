"""FastAPI server for the Adaptive Presentation Engine backend.

This file replaces the older `http.server` implementation and serves:
  - GET  /                : frontend (index.html)
  - GET  /api/health      : LLM connectivity check
  - GET  /api/state       : Thompson posterior and UI state
  - POST /api/chat_plain  : baseline chat (no strategy selection)
  - POST /api/chat        : adaptive chat (non-streaming)
  - POST /api/chat_stream : adaptive chat (SSE streaming)
  - POST /api/rate        : feedback update
  - POST /api/preference : preference update
  - POST /api/reset       : reset user session

The frontend expects a stream of JSON events with `evt.type` fields:
  - strategy
  - response_delta
  - widget_delta (mostly for openai_compat path)
  - done
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import uuid
from typing import Generator, Tuple

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
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


def _maybe_enforce_primitive(user_message: str, strategy: str, response: str) -> str:
    """Apply strict format except for short greeting/thanks-only turns (prompt already relaxes those)."""
    if not config.STRICT_PRIMITIVES or not response:
        return response
    if is_social_or_greeting_turn(user_message):
        return response
    return enforce_response(strategy, response)


bearer_scheme = HTTPBearer(auto_error=False)

_password_reset_lock = threading.Lock()
# In-memory reset tokens for demo/dev purposes.
# Production should store tokens securely (DB/Redis) + send via email.
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
    Admin check based on config lists.

    If no admin config is provided, keep previous behavior (everyone can access primitives).
    """
    if not getattr(config, "ADMIN_CONFIGURED", False):
        return True

    if user_id in set(map(str, getattr(config, "ADMIN_USER_IDS", []) or [])):
        return True

    rec = get_user_by_id(user_id)
    if not rec:
        return False

    username_n = (rec.username or "").strip().lower()
    email_n = (rec.email or "").strip().lower()
    return username_n in {u.lower() for u in (config.ADMIN_USERNAMES or [])} or email_n in {e.lower() for e in (config.ADMIN_EMAILS or [])}


def require_admin_user_id(user_id: str = Depends(require_user_id)) -> str:
    if not _is_admin_user(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user_id


def sse_pack(evt: dict) -> str:
    """Pack a JSON event into SSE data frame."""
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


def _parse_streamed_response(
    chunks, *, emit_raw_response_deltas: bool = True
) -> Generator[Tuple[str, ...], None, None]:
    """Parse streaming LLM output.

    Events:
    - ('response_delta', str) — raw token deltas inside <RESPONSE> (if emit_raw_response_deltas).
    - ('response_closed', str) — full text inside <RESPONSE>...</RESPONSE> (yielded before <WIDGET> finishes).
    - ('complete', str, str) — final response_text and widget_raw (after </WIDGET> or EOF).
    If emit_raw_response_deltas is False, buffer <RESPONSE> until </RESPONSE>; caller streams enforced text.
    """
    buffer = ""
    state = "preamble"  # preamble | response | widget
    response_sent_len = 0
    response_text = ""
    response_end_tag = "</RESPONSE>"
    response_start_tag = "<RESPONSE>"
    widget_start_tag = "<WIDGET>"
    widget_end_tag = "</WIDGET>"
    tag_max_len = max(len(response_end_tag), len(widget_end_tag))

    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        buf_upper = buffer.upper()

        if state == "preamble":
            if response_start_tag.upper() in buf_upper:
                idx = buf_upper.find(response_start_tag.upper()) + len(response_start_tag)
                buffer = buffer[idx:]
                buf_upper = buffer.upper()
                state = "response"
                response_sent_len = 0

        if state == "response":
            if response_end_tag.upper() in buf_upper:
                end_idx = buf_upper.find(response_end_tag.upper())
                to_send = buffer[:end_idx][response_sent_len:]
                if to_send and emit_raw_response_deltas:
                    yield ("response_delta", to_send)
                response_text = buffer[:end_idx].strip()
                buffer = buffer[end_idx + len(response_end_tag) :]
                buf_upper = buffer.upper()
                state = "widget_looking"
                yield ("response_closed", response_text)
            else:
                safe_len = max(0, len(buffer) - tag_max_len)
                if safe_len > response_sent_len:
                    to_send = buffer[response_sent_len:safe_len]
                    if emit_raw_response_deltas:
                        yield ("response_delta", to_send)
                    response_sent_len = safe_len

        if state == "widget_looking":
            if widget_start_tag.upper() in buf_upper:
                idx = buf_upper.find(widget_start_tag.upper()) + len(widget_start_tag)
                buffer = buffer[idx:]
                buf_upper = buffer.upper()
                state = "widget"

        if state == "widget":
            if widget_end_tag.upper() in buf_upper:
                end_idx = buf_upper.find(widget_end_tag.upper())
                widget_raw = buffer[:end_idx].strip()
                yield ("complete", response_text, widget_raw)
                return

    # Stream ended without full parse - yield what we have
    if state == "response" and buffer:
        remaining = buffer[response_sent_len:]
        if remaining and emit_raw_response_deltas:
            yield ("response_delta", remaining)
        response_text = buffer
    elif state == "preamble" and buffer.strip():
        # Model didn't use XML tags (e.g. returns "6" for "2*3") - treat raw as response.
        response_text = buffer.strip()
        if response_text and emit_raw_response_deltas:
            yield ("response_delta", response_text)
    widget_raw = buffer if state == "widget" else ""
    yield ("complete", response_text, widget_raw)


def _looks_truncated_widget_html(html: str) -> bool:
    """Heuristic check for obviously cut-off widget HTML."""
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
    if re.search(r"[<{(]$", html.strip()):
        return True
    return False


def _dispatch_json_mode_widget(widget_payload_raw: str) -> tuple[str, str, int, str]:
    """
    Prefer a valid JSON schema for the Vue renderer. If the model returned HTML/JS instead
    (common for sliders/calculators), fall back to iframe HTML.

    Returns:
        (widget_schema, widget_html, widget_height, widget_debug_tag)
    """
    raw = (widget_payload_raw or "").strip()
    if not raw:
        return "", "", 0, ""
    finalized = finalize_widget_schema_json(raw)
    if widget_schema_json_is_valid(finalized):
        return finalized, "", 0, "json_schema_ok"
    doc = extract_embeddable_html_document(raw)
    if doc:
        full = inject_design_system(doc)
        if _looks_truncated_widget_html(full):
            return finalized, "", 0, "json_html_fallback_truncated"
        return "", full, estimate_widget_height(full), "json_html_fallback"
    return finalized, "", 0, "json_schema_invalid"


def _should_generate_widget(message: str) -> bool:
    """
    Lightweight intent gate so simple chat/explanations do not force widgets.
    """
    text = (message or "").strip().lower()
    if not text:
        return False
    low_signal = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "got it",
        "cool",
    }
    if text in low_signal:
        return False

    # High-confidence visualization / analytics intents.
    widget_triggers = (
        "chart",
        "graph",
        "plot",
        "dashboard",
        "table",
        "compare",
        "comparison",
        "trend",
        "timeseries",
        "time series",
        "distribution",
        "heatmap",
        "scatter",
        "pie",
        "bar",
        "line",
        "kpi",
        "analytics",
        "analyze",
        "analysis",
        "forecast",
        "breakdown",
        "report",
        "visualize",
        "visualise",
        "show me",
        "insight",
        "metrics",
        "kpi",
    )
    if any(t in text for t in widget_triggers):
        return True

    # Questions that are typically better as text-only.
    text_only_intents = (
        "explain",
        "what is",
        "why",
        "how does",
        "difference between",
        "define",
        "summarize",
        "summarise",
        "plan",
        "roadmap",
        "steps",
        "implementation plan",
    )
    if any(t in text for t in text_only_intents):
        return False

    # Data-like cues: numbers/percentages/time windows usually benefit from widgets.
    has_numeric_cue = bool(re.search(r"\b\d+(\.\d+)?%?\b", text))
    has_time_cue = any(t in text for t in ("daily", "weekly", "monthly", "quarterly", "yearly", "over time", "timeline"))
    has_compare_cue = any(t in text for t in ("vs", "versus", "compare", "top", "rank", "distribution"))
    if has_numeric_cue and (has_time_cue or has_compare_cue):
        return True

    # Conservative default: no widget unless clearly useful.
    return False


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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup():
    persistence.init_db()

    # Optionally auto-create the configured admin (so you can log in and manage primitives).
    if getattr(config, "ADMIN_CONFIGURED", False) and config.ADMIN_USERNAME and config.ADMIN_PASSWORD and config.ADMIN_EMAIL:
        try:
            register_user(username=config.ADMIN_USERNAME, email=config.ADMIN_EMAIL, password=config.ADMIN_PASSWORD)
        except Exception:
            # Username/email might already exist; that's fine.
            pass

    # Seed the in-memory auth store from persisted users.
    try:
        users = persistence.load_users_from_db()
        seed_users_from_db(users)
    except Exception:
        # Auth can still work for dev if DB isn't ready.
        pass

    # Restore bandit (global + per-user) state.
    try:
        persistence.load_global_state(engine)
        persistence.load_user_states(engine)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = config.INDEX_HTML.read_bytes()
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


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
    """
    Dev/demo forgot-password endpoint.
    Returns a reset token so you can test the flow from the UI.
    """
    email = (req.email or "").strip()
    user = get_user_by_email(email)

    # Always return success to avoid user enumeration.
    if not user:
        return {"ok": True, "reset_token": None}

    token = uuid.uuid4().hex
    now = int(time.time())
    expires = now + _PASSWORD_RESET_TTL_SECONDS
    with _password_reset_lock:
        _password_reset_tokens[token] = {"user_id": user.user_id, "expires": expires}

    return {"ok": True, "reset_token": token}


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
    """
    Returns strategy bars for the bandit.
    - Non-admin: returns enabled strategies only.
    - Admin: returns all strategies (enabled + disabled).
    """
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
    # Stable ordering: enabled first, then alpha by id.
    items.sort(key=lambda x: (not bool(x.get("enabled", True)), str(x.get("id") or "")))
    return {"items": items}


# Explicit preflight handling (avoid 405 for OPTIONS with Authorization headers).
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


@app.options("/api/strategies/usage")
def strategies_usage_preflight():
    return JSONResponse({"ok": True})


@app.options("/api/strategies/usage/")
def strategies_usage_preflight_slash():
    return JSONResponse({"ok": True})


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


@app.get("/api/strategies/{sid}/analytics/")
def strategy_analytics_slash(
    sid: str,
    days: int = 30,
    user_id: str = Depends(require_admin_user_id),
):
    return strategy_analytics(sid=sid, days=days, user_id=user_id)


@app.options("/api/strategies/{sid}/analytics")
def strategy_analytics_preflight(sid: str):
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}/analytics/")
def strategy_analytics_preflight_slash(sid: str):
    return JSONResponse({"ok": True})


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


@app.post("/api/strategies/")
def create_strategy_slash(req: StrategyCreateReq, user_id: str = Depends(require_admin_user_id)):
    return create_strategy(req=req, user_id=user_id)


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


@app.put("/api/strategies/{sid}/")
def update_strategy_slash(sid: str, req: StrategyUpdateReq, user_id: str = Depends(require_admin_user_id)):
    return update_strategy(sid=sid, req=req, user_id=user_id)


@app.options("/api/strategies/{sid}")
def strategy_item_preflight(sid: str):
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}/")
def strategy_item_preflight_slash(sid: str):
    return JSONResponse({"ok": True})


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


@app.post("/api/strategies/{sid}/enable/")
def enable_strategy_slash(sid: str, user_id: str = Depends(require_admin_user_id)):
    return enable_strategy(sid=sid, user_id=user_id)


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


@app.post("/api/strategies/{sid}/disable/")
def disable_strategy_slash(sid: str, user_id: str = Depends(require_admin_user_id)):
    return disable_strategy(sid=sid, user_id=user_id)


@app.options("/api/strategies/{sid}/enable")
def enable_strategy_preflight(sid: str):
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}/enable/")
def enable_strategy_preflight_slash(sid: str):
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}/disable")
def disable_strategy_preflight(sid: str):
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}/disable/")
def disable_strategy_preflight_slash(sid: str):
    return JSONResponse({"ok": True})


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


@app.delete("/api/strategies/{sid}/")
def delete_strategy_slash(sid: str, user_id: str = Depends(require_admin_user_id)):
    return delete_strategy(sid=sid, user_id=user_id)


@app.options("/api/strategies/{sid}/delete")
def delete_strategy_preflight_delete(sid: str):
    # Some clients may send OPTIONS to /delete - keep harmless.
    return JSONResponse({"ok": True})


@app.options("/api/strategies/{sid}")
def delete_strategy_preflight(sid: str):
    return JSONResponse({"ok": True})


@app.get("/api/primitives")
def list_primitives(user_id: str = Depends(require_admin_user_id)):
    return {"items": persistence.list_user_primitives(user_id)}


# Explicit preflight handling (some browsers/dev setups still hit 405 for OPTIONS).
@app.options("/api/primitives")
def primitives_preflight():
    return JSONResponse({"ok": True})


@app.options("/api/primitives/")
def primitives_preflight_slash():
    return JSONResponse({"ok": True})


@app.post("/api/primitives")
def create_primitive(req: PrimitiveCreateReq, user_id: str = Depends(require_admin_user_id)):
    name = (req.name or "").strip()
    inst = (req.instruction or "").strip()
    if not name or not inst:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="name and instruction are required")
    row = persistence.create_user_primitive(user_id=user_id, name=name, instruction=inst)
    return {"item": row}


@app.post("/api/primitives/")
def create_primitive_slash(req: PrimitiveCreateReq, user_id: str = Depends(require_user_id)):
    return create_primitive(req, user_id=user_id)


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


@app.put("/api/primitives/{prim_id}/")
def update_primitive_slash(prim_id: int, req: PrimitiveUpdateReq, user_id: str = Depends(require_user_id)):
    return update_primitive(prim_id=prim_id, req=req, user_id=user_id)


@app.options("/api/primitives/{prim_id}")
def primitive_item_preflight(prim_id: int):
    return JSONResponse({"ok": True})


@app.options("/api/primitives/{prim_id}/")
def primitive_item_preflight_slash(prim_id: int):
    return JSONResponse({"ok": True})


@app.delete("/api/primitives/{prim_id}")
def delete_primitive(prim_id: int, user_id: str = Depends(require_admin_user_id)):
    ok = persistence.delete_user_primitive(user_id=user_id, prim_id=int(prim_id))
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Primitive not found")
    return {"ok": True}


@app.delete("/api/primitives/{prim_id}/")
def delete_primitive_slash(prim_id: int, user_id: str = Depends(require_user_id)):
    return delete_primitive(prim_id=prim_id, user_id=user_id)


@app.get("/api/conversation")
def conversation(limit: int = 20, user_id: str = Depends(require_user_id)):
    """
    Load per-user conversation history from SQLite.

    Returns separate histories for:
    - adaptive pane: {user_id}
    - baseline pane: {user_id}_plain
    """
    uid = user_id
    lim = int(limit)
    lim = max(1, min(lim, 50))

    # Prefer SQL logs (true conversation history). Falls back to engine history if empty.
    def _pairs_from_msgs(msgs: list[dict]) -> list[dict]:
        out: list[dict] = []
        last_user: str | None = None
        for m in msgs:
            if m.get("role") == "user":
                last_user = str(m.get("content") or "")
            elif m.get("role") == "assistant":
                if last_user is None:
                    continue
                out.append({"user": last_user, "assistant": str(m.get("content") or "")})
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


@app.post("/api/chat_plain")
def chat_plain(req: ChatPlainReq, user_id: str = Depends(require_user_id)):
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
            response, elapsed, mode = llm.call_openai_compat(prompt, system, timeout=120)
        elif base_mode == "anthropic":
            response, elapsed, mode = llm.call_anthropic(prompt, system, timeout=120)
        else:
            raise RuntimeError("Unsupported BASELINE_LLM_MODE (expected openai_compat or anthropic)")
    except Exception as e:
        return JSONResponse({"error": f"LLM error: {str(e)}"}, status_code=500)

    if not response:
        return JSONResponse({"error": "LLM returned empty response. Check model/service."}, status_code=500)

    user["history"].append({"user": msg, "assistant": response})
    user["history"] = user["history"][-20:]

    # Persist session history for this user.
    persistence.persist_user_state(engine, uid)
    persistence.log_conversation_message(user_id=user_id, pane="baseline", role="user", content=msg)
    persistence.log_conversation_message(user_id=user_id, pane="baseline", role="assistant", content=response, elapsed=elapsed)

    return {"response": response, "elapsed": elapsed, "llm_mode": mode}


def _post_done_payload(
    *,
    uid: str,
    strat: str,
    prev: str | None,
    explicit: bool,
    force_explore: bool,
    instruction: str,
    format_rule: str,
    elapsed: float | None,
    mode: str,
    scores: dict,
    x: np.ndarray,
    auto_detected: bool,
    auto_r: float | None,
    ev_reason: str,
    response: str,
    widget_html: str,
    widget_schema: str,
    widget_height: int,
    widget_debug: str,
    raw_preview: str,
):
    ub = engine.get_user(USERB_ID)
    return {
        "response": response,
        "strategy": strat,
        "prev_strategy": prev,
        "explicit": explicit,
        "force_explore": force_explore,
        "instruction": instruction,
        "format_rule": format_rule,
        "elapsed": elapsed,
        "llm_mode": mode,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "x_vec": x.tolist(),
        "posterior": engine.user_posterior(uid, x),
        "global": engine.global_posterior(x),
        "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
        "global_n": engine.global_n,
        "auto_detected": auto_detected,
        "auto_r": auto_r,
        "auto_reason": ev_reason,
        "widget_html": widget_html or "",
        "widget_schema": widget_schema or "",
        "widget_height": widget_height,
        "widget_debug": widget_debug,
        "widget_raw_preview": raw_preview if not (widget_html or widget_schema) else "",
    }


@app.post("/api/chat")
def chat(req: ChatReq, user_id: str = Depends(require_user_id)):
    uid = user_id
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

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
    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
    combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)
    widget_required = _should_generate_widget(msg)
    widget_required = _should_generate_widget(msg)

    # Inject user primitives (per-user) into the system prompt.
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

    t0 = time.time()
    try:
        adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()
        if adapt_mode == "openai_compat":
            raw_combined, elapsed, mode = llm.call_openai_compat(
                combined_prompt, combined_system, timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2
            )
        elif adapt_mode == "anthropic":
            raw_combined, elapsed, mode = llm.call_anthropic(
                combined_prompt, combined_system, timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2
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
    response = _maybe_enforce_primitive(msg, strat, response)

    widget_html = ""
    widget_schema = ""
    widget_height = 0
    widget_debug = ""
    widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
    raw_preview = ""

    if widget_payload_raw:
        if widget_mode == "json":
            widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(widget_payload_raw)
            if tag:
                widget_debug = widget_debug or tag
        else:
            if _looks_truncated_widget_html(widget_payload_raw):
                widget_debug = "combined_widget_truncated"
            else:
                widget_html = widget_payload_raw
                widget_height = estimate_widget_height(widget_payload_raw)
                widget_debug = widget_debug or "combined_widget_ok"
    else:
        widget_debug = widget_debug or ("combined_no_schema" if widget_mode == "json" else "combined_no_widget_tag")
        raw_preview = (raw_combined or "")[:800]
        if widget_mode != "json" and widget_required:
            placeholder = "<html><head></head><body><div class='widget-root card'><div class='card-title'>Interactive widget</div><div class='empty'>No widget returned.</div></div></body></html>"
            widget_html = inject_design_system(placeholder)
            widget_height = estimate_widget_height(widget_html)
            widget_debug = "fallback_widget_generated"
        elif not widget_required:
            widget_debug = "widget_skipped_by_intent"

    # Update history
    user["history"].append({"user": msg, "assistant": response})
    user["history"] = user["history"][-20:]
    user["last_message"] = msg
    user["last_response"] = response
    user["last_strategy"] = strat
    user["last_x"] = x.tolist()
    user["msg_count"] += 1

    payload = _post_done_payload(
        uid=uid,
        strat=strat,
        prev=prev,
        explicit=explicit,
        force_explore=force_explore and (explicit is None),
        instruction=config.STRATEGIES[strat],
        format_rule=format_rule,
        elapsed=elapsed,
        mode=mode,
        scores=scores,
        x=x,
        auto_detected=auto_detected,
        auto_r=auto_r,
        ev_reason=ev["reason"],
        response=response,
        widget_html=widget_html,
        widget_schema=widget_schema,
        widget_height=widget_height,
        widget_debug=widget_debug,
        raw_preview=raw_preview,
    )

    # Persist user + global bandit updates.
    persistence.persist_user_state(engine, uid)
    persistence.persist_global_state(engine)
    persistence.log_conversation_message(user_id=user_id, pane="adaptive", role="user", content=msg, strategy=strat)
    persistence.log_conversation_message(
        user_id=user_id,
        pane="adaptive",
        role="assistant",
        content=response,
        strategy=strat,
        elapsed=elapsed,
        widget=bool(widget_html or widget_schema),
    )

    return payload


@app.post("/api/chat_stream")
def chat_stream(req: ChatReq, user_id: str = Depends(require_user_id)):
    uid = user_id
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

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
    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
    combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)
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

    adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()

    ub = engine.get_user(USERB_ID)

    def gen():
        t0 = time.time()
        # Initial strategy event
        yield sse_pack(
            {
                "type": "strategy",
                "strategy": strat,
                "instruction": config.STRATEGIES[strat],
                "format_rule": format_rule,
                "elapsed": None,
                "force_explore": force_explore,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            }
        )

        widget_html = ""
        widget_schema = ""
        widget_height = 0
        widget_debug = ""
        widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
        raw_preview = ""
        response = ""
        mode = adapt_mode

        try:
            if adapt_mode == "openai_compat":
                raw_combined, elapsed, mode = llm.call_openai_compat(
                    combined_prompt, combined_system, timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2
                )
                response, widget_payload_raw = parse_combined_output(raw_combined)
                if not response:
                    response = raw_combined.strip()
                response = _maybe_enforce_primitive(msg, strat, response)

                if widget_payload_raw:
                    if widget_mode == "json":
                        widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(
                            widget_payload_raw
                        )
                        widget_debug = tag
                    else:
                        if _looks_truncated_widget_html(widget_payload_raw):
                            widget_debug = "combined_widget_truncated"
                        else:
                            widget_html = widget_payload_raw
                            widget_height = estimate_widget_height(widget_html)
                            widget_debug = "combined_widget_ok"
                else:
                    widget_debug = "combined_no_widget_tag" if widget_mode != "json" else "combined_no_schema"
                    raw_preview = (raw_combined or "")[:800]
                    if widget_mode != "json" and widget_required:
                        placeholder = "<html><head></head><body><div class='widget-root card'><div class='card-title'>Interactive widget</div><div class='empty'>No widget returned.</div></div></body></html>"
                        widget_html = inject_design_system(placeholder)
                        widget_height = estimate_widget_height(widget_html)
                        widget_debug = "fallback_widget_generated"
                    elif not widget_required:
                        widget_debug = "widget_skipped_by_intent"

                # Stream response deltas + widget deltas (to keep UI progress behavior)
                for i in range(0, len(response), 180):
                    yield sse_pack({"type": "response_delta", "delta": response[i : i + 180]})
                payload = widget_schema if (widget_mode == "json" and widget_schema) else widget_html
                if payload:
                    # Signal UI that widget generation/processing is starting.
                    yield sse_pack({"type": "widget_start"})
                    for i in range(0, len(payload), 900):
                        yield sse_pack({"type": "widget_delta", "delta": payload[i : i + 900]})

                elapsed_out = elapsed

            elif adapt_mode == "anthropic":
                stream = llm.stream_anthropic(
                    combined_prompt, combined_system, timeout=combined_timeout, max_tokens=combined_max_tokens, temperature=0.2
                )
                response = ""
                widget_payload_raw = ""
                response_done = False
                emit_raw = not (config.STRICT_PRIMITIVES and not is_social_or_greeting_turn(msg))
                for event_type, *args in _parse_streamed_response(stream, emit_raw_response_deltas=emit_raw):
                    if event_type == "response_delta":
                        yield sse_pack({"type": "response_delta", "delta": args[0]})
                    elif event_type == "response_closed":
                        response_done = True
                        response = _maybe_enforce_primitive(msg, strat, args[0])
                        if not emit_raw:
                            for i in range(0, len(response), 180):
                                yield sse_pack({"type": "response_delta", "delta": response[i : i + 180]})
                        yield sse_pack({"type": "widget_start"})
                    elif event_type == "complete":
                        if not response_done:
                            response = _maybe_enforce_primitive(msg, strat, args[0])
                            if not emit_raw:
                                for i in range(0, len(response), 180):
                                    yield sse_pack({"type": "response_delta", "delta": response[i : i + 180]})
                            yield sse_pack({"type": "widget_start"})
                        widget_payload_raw = args[1]
                        break
                elapsed_out = round(time.time() - t0, 1)
                mode = "anthropic"

                if not response and not widget_payload_raw:
                    response = "(No content)"

                if widget_payload_raw:
                    if widget_mode == "json":
                        widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(
                            widget_payload_raw
                        )
                        widget_debug = tag
                    else:
                        raw_widget = widget_payload_raw
                        if "```" in raw_widget:
                            fence = re.search(r"```(?:json|html)?\s*(.*?)```", raw_widget, re.DOTALL | re.IGNORECASE)
                            raw_widget = fence.group(1).strip() if fence else re.sub(r"```\w*", "", raw_widget).strip()
                        if "<" in raw_widget and ">" in raw_widget:
                            if "<html" not in raw_widget.lower():
                                raw_widget = f"<html><head></head><body>{raw_widget}</body></html>"
                            raw_widget = re.sub(r"<!DOCTYPE[^>]*>", "", raw_widget, flags=re.IGNORECASE).strip()
                            widget_payload_raw = inject_design_system(raw_widget)
                            if _looks_truncated_widget_html(widget_payload_raw):
                                widget_debug = "stream_widget_truncated"
                            else:
                                widget_html = widget_payload_raw
                                widget_height = estimate_widget_height(widget_html)
                if widget_mode != "json" and not widget_html:
                    widget_debug = "no_widget"

            else:
                raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")

        except Exception as e:
            msg_err = str(e).strip() or repr(e)
            yield sse_pack(
                {
                    "type": "done",
                    "strategy": strat,
                    "format_rule": format_rule,
                    "elapsed": None,
                    "llm_mode": mode,
                    "response": "",
                    "widget_html": "",
                    "widget_schema": "",
                    "widget_height": 0,
                    "widget_debug": f"stream_error:{msg_err}",
                    "error": msg_err,
                    "force_explore": force_explore,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "x_vec": x.tolist(),
                    "posterior": engine.user_posterior(uid, x),
                    "global": engine.global_posterior(x),
                    "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                    "global_n": engine.global_n,
                    "prev_strategy": prev,
                    "explicit": explicit,
                    "auto_detected": auto_detected,
                    "auto_r": auto_r,
                    "auto_reason": ev["reason"],
                }
            )
            return

        # Update history
        user["history"].append({"user": msg, "assistant": response})
        user["history"] = user["history"][-20:]
        user["last_message"] = msg
        user["last_response"] = response
        user["last_strategy"] = strat
        user["last_x"] = x.tolist()
        user["msg_count"] += 1

        # Persist bandit + history after the final response for this stream.
        persistence.persist_user_state(engine, uid)
        persistence.persist_global_state(engine)
        persistence.log_conversation_message(user_id=user_id, pane="adaptive", role="user", content=msg, strategy=strat)
        persistence.log_conversation_message(
            user_id=user_id,
            pane="adaptive",
            role="assistant",
            content=response,
            strategy=strat,
            elapsed=elapsed_out,
            widget=bool(widget_html or widget_schema),
        )

        yield sse_pack(
            {
                "type": "done",
                "strategy": strat,
                "format_rule": format_rule,
                "elapsed": elapsed_out,
                "llm_mode": mode,
                "response": response,
                "widget_html": widget_html or "",
                "widget_schema": widget_schema or "",
                "widget_height": widget_height,
                "widget_debug": widget_debug,
                "force_explore": force_explore and (explicit is None),
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            }
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/rate")
def rate(req: RateReq, user_id: str = Depends(require_user_id)):
    uid = user_id
    strategy = req.strategy
    x = np.array(req.x_vec, dtype=float)
    reward = float(req.reward)
    if strategy not in config.STRATEGY_NAMES:
        return JSONResponse({"error": "bad request"}, status_code=400)
    engine.update(uid, strategy, x, reward)
    persistence.persist_user_state(engine, uid)
    persistence.persist_global_state(engine)
    ub = engine.get_user(USERB_ID)
    return {
        "posterior": engine.user_posterior(uid, x),
        "global": engine.global_posterior(x),
        "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
        "global_n": engine.global_n,
    }


@app.post("/api/preference")
def preference(req: PreferenceReq, user_id: str = Depends(require_user_id)):
    uid = user_id
    engine.apply_preferences(uid, req.strategies, lock=bool(req.lock))
    persistence.persist_user_state(engine, uid)
    return {"posterior": engine.user_posterior(uid)}


@app.post("/api/reset")
def reset(req: ResetReq, user_id: str = Depends(require_user_id)):
    # Reset adaptive state + baseline state (chat_plain) for this user.
    plain_uid = user_id + "_plain"
    engine.reset_user(user_id)
    engine.reset_user(plain_uid)
    persistence.delete_user_state(user_id)
    persistence.delete_user_state(plain_uid)
    persistence.delete_conversation_logs(user_id=user_id, pane="adaptive")
    persistence.delete_conversation_logs(user_id=user_id, pane="baseline")
    return {"ok": True}


# -----------------------------------------------------------------------------
# Serve Vue production build (SPA)
# -----------------------------------------------------------------------------
_frontend_dist_assets_dir = config.INDEX_HTML.parent / "assets"
if _frontend_dist_assets_dir.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist_assets_dir), html=False),
        name="frontend_assets",
    )


@app.get("/{full_path:path}", response_class=HTMLResponse)
def spa_fallback(full_path: str):
    # Let the existing `/api/*` routes win; this handler only runs for
    # non-API paths.
    if full_path.startswith("api"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
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

"""HTTP server and request handlers for the Adaptive Presentation Engine backend.

Serves:
  GET /              - Frontend (index.html from frontend/)
  GET /api/health    - LLM connectivity check
  GET /api/state     - Thompson Sampling posterior and user state
  POST /api/chat     - Adaptive chat (strategy + response + widget)
  POST /api/chat_plain - Baseline chat (no strategy selection)
  POST /api/rate     - Feedback (thumbs up/down) for posterior update
  POST /api/reset    - Reset user session

Uses ThreadingMixIn for concurrent requests. CORS enabled for all origins.
"""

import json
import os
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np

from . import config, llm
from .engine import engine, USERB_ID
from .widget_prompt import (
    estimate_widget_height,
    inject_design_system,
)
from .combined_prompt import (
    build_combined_system_prompt,
    build_combined_user_prompt,
    extract_embeddable_html_document,
    finalize_widget_schema_json,
    is_social_or_greeting_turn,
    parse_combined_output,
    widget_schema_json_is_valid,
)
from .utils import (
    fast_valence,
    enforce_response,
    detect_explore_trigger,
    negative_strength,
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Handler(BaseHTTPRequestHandler):
    """Handles all HTTP requests. Suppresses default request logging."""

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        """Serve the frontend index.html (from frontend/ folder)."""
        try:
            html = config.INDEX_HTML.read_bytes()
        except FileNotFoundError:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        raw = self.rfile.read(n)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Don't crash the handler thread; return a sentinel the caller can handle.
            try:
                preview = raw[:500].decode("utf-8", errors="replace")
            except Exception:
                preview = repr(raw[:200])
            return {"__invalid_json__": True, "__raw_preview__": preview}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._html(); return

        if p == "/api/health":
            if config.LLM_MODE == "openai_compat":
                h = llm.openai_health()
                self._json({"server": "ok", "mode": config.LLM_MODE, "openai_base_url": config.OPENAI_BASE_URL, "model": config.OPENAI_MODEL, **h})
            elif config.LLM_MODE == "anthropic":
                h = llm.anthropic_health()
                self._json({"server": "ok", "mode": config.LLM_MODE, **h})
            else:
                self._json({"server": "ok", "mode": config.LLM_MODE, "ok": False, "reachable": False, "error": "Unsupported LLM_MODE (expected openai_compat or anthropic)"})
            return

        if p == "/api/state":
            uid  = self.path.split("uid=")[-1] if "uid=" in self.path else "demo"
            user = engine.get_user(uid)
            x    = np.ones(config.D) * 0.5
            ub   = engine.get_user(USERB_ID)
            self._json({
                "posterior": engine.user_posterior(uid, x),
                "global":    engine.global_posterior(x),
                "userb":     engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":  engine.global_n,
                "n_users":   len(engine.users),
                "msg_count": user["msg_count"],
            }); return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        p    = self.path.split("?")[0]
        body = self._body()
        if isinstance(body, dict) and body.get("__invalid_json__"):
            self._json({"error": "invalid_json", "preview": body.get("__raw_preview__", "")}, 400)
            return

        if p == "/api/chat_plain":
            uid = body.get("uid", "demo") + "_plain"
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Build a simple conversation prompt (no bandit, no enforced format).
            ctx = []
            for t in user["history"][-6:]:
                ctx += [f"User: {t['user']}", f"Assistant: {t['assistant']}"]
            ctx.append(f"User: {msg}")
            prompt = "\n".join(ctx)

            system = "You are a helpful AI assistant."

            try:
                base_mode = (config.BASELINE_LLM_MODE or config.LLM_MODE).lower()
                if base_mode == "openai_compat":
                    response, elapsed, mode = llm.call_openai_compat(prompt, system, timeout=120)
                elif base_mode == "anthropic":
                    response, elapsed, mode = llm.call_anthropic(prompt, system, timeout=120)
                else:
                    raise RuntimeError("Unsupported BASELINE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                self._json({"error": f"LLM error: {str(e)}"}, 500)
                return

            if not response:
                self._json({"error": "LLM returned empty response. Check model/service."}, 500)
                return

            user["history"].append({"user": msg, "assistant": response})
            user["history"] = user["history"][-20:]

            self._json({
                "response": response,
                "elapsed":  elapsed,
                "llm_mode": mode,
            }); return

        if p == "/api/chat":
            uid = body.get("uid", "demo")
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Auto-reward previous turn using valence heuristic.
            ev            = fast_valence(msg, user["last_response"])
            auto_detected = False
            auto_r        = None

            if user["last_response"] and user["last_x"] is not None and user["last_strategy"]:
                reward = float(np.clip(0.5 + 0.45*ev["pos"] - 0.45*ev["neg"], 0.05, 0.95))
                engine.update(uid, user["last_strategy"], np.array(user["last_x"]), reward)
                auto_detected = True
                auto_r        = reward

            # --- Corrective exploration (no keyword-based strategy override) ---
            explicit = False  # No keyword override; model understands the question
            force_explore = bool(detect_explore_trigger(msg) or (ev.get("neg", 0.0) >= config.NEG_EXPLORE_THRESHOLD))
            neg_s = negative_strength(ev)

            strat, scores, x, prev = engine.select(
                uid, msg,
                force_explore=force_explore,
                neg_strength=neg_s,
                explicit_strategy=None,
            )

            format_rule = config.STRATEGIES.get(strat, "Be helpful and clear.")
            combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
            combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)

            # ── Combined single-call (response + widget in one LLM output) ──
            combined_system = build_combined_system_prompt(
                strategy_id=strat,
                format_rule=format_rule,
                primitive_extra_context=getattr(config, "SKILLS_CONTENT", "") or "",
                user_message=msg,
                forbidden_components=None,
                required_components=None,
            )
            combined_prompt = build_combined_user_prompt(msg, user["history"])
            try:
                adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()
                if adapt_mode == "openai_compat":
                    raw_combined, elapsed, mode = llm.call_openai_compat(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                elif adapt_mode == "anthropic":
                    raw_combined, elapsed, mode = llm.call_anthropic(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                else:
                    raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                self._json({"error": f"LLM error: {str(e)}"}, 500)
                return
            if not raw_combined:
                self._json({"error": "LLM returned empty response. Check model/service."}, 500)
                return

            response, widget_payload_raw = parse_combined_output(raw_combined)
            if not response:
                response = raw_combined.strip()
            response = _maybe_enforce_primitive(msg, strat, response)
            widget_html = ""
            widget_schema = ""
            widget_height = 0
            widget_debug = ""
            widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
            raw_preview = ""

            if widget_payload_raw:
                if widget_mode == "json":
                    widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(widget_payload_raw)
                    if tag:
                        widget_debug = widget_debug or tag
                else:
                    if _looks_truncated_widget_html(widget_payload_raw):
                        widget_debug = "combined_widget_truncated"
                    else:
                        widget_html = widget_payload_raw
                        widget_height = estimate_widget_height(widget_payload_raw)
                        widget_debug = widget_debug or "combined_widget_ok"
            else:
                widget_debug = widget_debug or ("combined_no_schema" if widget_mode == "json" else "combined_no_widget_tag")
                raw_preview = (raw_combined or "")[:800]
                if widget_mode != "json":
                    # Never return a blank iframe: serve a safe interactive placeholder mini-app.
                    placeholder = """<html><head></head><body>
  <div class="widget-root card">
    <div class="card-title">Interactive widget</div>
    <div style="color:var(--text2);font-size:13px;line-height:1.6">
      Could not generate a widget for this turn. Try rephrasing or providing more details.
    </div>
  </div>
</body></html>""".strip()
                    widget_html = inject_design_system(placeholder)
                    widget_height = estimate_widget_height(widget_html)
                    widget_debug = "fallback_widget_generated"

            # No primitive fallback. If widget is missing/invalid, return text-only.

            user["history"].append({"user": msg, "assistant": response})
            user["history"]       = user["history"][-20:]
            user["last_message"]  = msg
            user["last_response"] = response
            user["last_strategy"] = strat
            user["last_x"]        = x.tolist()
            user["msg_count"]    += 1

            ub = engine.get_user(USERB_ID)
            self._json({
                "response":      response,
                "strategy":      strat,
                "prev_strategy": prev,
                "explicit":      explicit,
                "force_explore": force_explore and (explicit is None),
                "instruction":   config.STRATEGIES[strat],
                "elapsed":       elapsed,
                "llm_mode":      mode,
                "scores":        {k: round(v, 4) for k, v in scores.items()},
                "x_vec":         x.tolist(),
                "posterior":     engine.user_posterior(uid, x),
                "global":        engine.global_posterior(x),
                "userb":         engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":      engine.global_n,
                "auto_detected": auto_detected,
                "auto_r":        auto_r,
                "auto_reason":   ev["reason"],
                "widget_html":   widget_html,
                "widget_schema": widget_schema,
                "widget_height": widget_height,
                "widget_debug":  widget_debug,
                "widget_raw_preview": raw_preview if not (widget_html or widget_schema) else "",
            }); return

        if p == "/api/reward":
            uid      = body.get("uid", "demo")
            strategy = body.get("strategy")
            x_vec    = body.get("x_vec")
            reward   = float(body.get("reward", 0.5))
            if strategy not in config.STRATEGY_NAMES or x_vec is None:
                self._json({"error": "bad request"}, 400); return
            x = np.array(x_vec, dtype=float)
            engine.update(uid, strategy, x, reward)
            ub = engine.get_user(USERB_ID)
            self._json({
                "posterior": engine.user_posterior(uid, x),
                "global":    engine.global_posterior(x),
                "userb":     engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":  engine.global_n,
            }); return

        if p == "/api/preference":
            uid = body.get("uid", "demo")
            strategies = body.get("strategies", [])
            lock = bool(body.get("lock", False))
            engine.apply_preferences(uid, strategies, lock=lock)
            self._json({"posterior": engine.user_posterior(uid)}); return

        if p == "/api/reset":
            engine.reset_user(body.get("uid", "demo"))
            self._json({"ok": True}); return

        # ── NEW: adaptive streaming endpoint (Claude-like) ────────────────
        if p == "/api/chat_stream":
            uid = body.get("uid", "demo")
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Auto-reward previous turn using valence heuristic.
            ev            = fast_valence(msg, user["last_response"])
            auto_detected = False
            auto_r        = None

            if user["last_response"] and user["last_x"] is not None and user["last_strategy"]:
                reward = float(np.clip(0.5 + 0.45*ev["pos"] - 0.45*ev["neg"], 0.05, 0.95))
                engine.update(uid, user["last_strategy"], np.array(user["last_x"]), reward)
                auto_detected = True
                auto_r        = reward

            explicit = False  # No keyword override; model understands the question
            force_explore = bool(detect_explore_trigger(msg) or (ev.get("neg", 0.0) >= config.NEG_EXPLORE_THRESHOLD))
            neg_s = negative_strength(ev)

            strat, scores, x, prev = engine.select(
                uid, msg,
                force_explore=force_explore,
                neg_strength=neg_s,
                explicit_strategy=None,
            )

            format_rule = config.STRATEGIES.get(strat, "Be helpful and clear.")
            combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
            combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)
            combined_system = build_combined_system_prompt(
                strategy_id=strat,
                format_rule=format_rule,
                primitive_extra_context=getattr(config, "SKILLS_CONTENT", "") or "",
                user_message=msg,
                forbidden_components=None,
                required_components=None,
            )
            combined_prompt = build_combined_user_prompt(user_message=msg, history=user["history"])

            adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()

            # Start NDJSON stream.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self._cors()
            self.end_headers()

            ub = engine.get_user(USERB_ID)

            def send_nd(evt: dict):
                try:
                    line = json.dumps(evt, ensure_ascii=False)
                    self.wfile.write(line.encode("utf-8") + b"\n")
                    self.wfile.flush()
                except Exception:
                    pass

            def send_done_error(err: str):
                send_nd({
                    "type": "done",
                    "strategy": strat,
                    "elapsed": None,
                    "llm_mode": adapt_mode,
                    "response": "",
                    "widget_html": "",
                    "widget_schema": "",
                    "widget_height": 0,
                    "widget_debug": f"stream_error:{err}",
                    "error": err,
                    "force_explore": force_explore,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "x_vec": x.tolist(),
                    "posterior": engine.user_posterior(uid, x),
                    "global": engine.global_posterior(x),
                    "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                    "global_n": engine.global_n,
                    "prev_strategy": prev,
                    "explicit": explicit,
                    "auto_detected": auto_detected,
                    "auto_r": auto_r,
                    "auto_reason": ev["reason"],
                })

            # Initial strategy event so UI updates immediately.
            send_nd({
                "type": "strategy",
                "strategy": strat,
                "instruction": config.STRATEGIES[strat],
                "elapsed": None,
                "force_explore": force_explore,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            })

            t0 = time.time()
            try:
                if adapt_mode == "openai_compat":
                    raw_combined, elapsed, mode = llm.call_openai_compat(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                    response, widget_payload_raw = parse_combined_output(raw_combined)
                    if not response:
                        response = raw_combined.strip()
                    response = _maybe_enforce_primitive(msg, strat, response)
                    widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
                    widget_html = ""
                    widget_schema = ""
                    widget_height = 0
                    widget_debug = "nonstream"
                    if widget_payload_raw:
                        if widget_mode == "json":
                            widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(
                                widget_payload_raw
                            )
                            widget_debug = tag
                        else:
                            if _looks_truncated_widget_html(widget_payload_raw):
                                widget_debug = "stream_widget_truncated"
                            else:
                                widget_html = widget_payload_raw
                                widget_height = estimate_widget_height(widget_payload_raw)
                    else:
                        if widget_mode != "json":
                            placeholder = "<html><head></head><body><div class='widget-root card'><div class='card-title'>Interactive widget</div><div class='empty'>No widget returned.</div></div></body></html>"
                            widget_html = inject_design_system(placeholder)
                            widget_height = estimate_widget_height(widget_html)
                            widget_debug = "fallback_widget_generated"
                    for i in range(0, len(response), 180):
                        send_nd({"type": "response_delta", "delta": response[i : i + 180]})
                    payload = (
                        widget_schema
                        if (widget_mode == "json" and widget_schema)
                        else widget_html
                    )
                    if payload:
                        for i in range(0, len(payload), 900):
                            send_nd({"type": "widget_delta", "delta": payload[i : i + 900]})
                elif adapt_mode == "anthropic":
                    stream = llm.stream_anthropic(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                    response = ""
                    widget_payload_raw = ""
                    response_done = False
                    emit_raw = not (config.STRICT_PRIMITIVES and not is_social_or_greeting_turn(msg))
                    for event_type, *args in _parse_streamed_response(stream, emit_raw_response_deltas=emit_raw):
                        if event_type == "response_delta":
                            send_nd({"type": "response_delta", "delta": args[0]})
                        elif event_type == "response_closed":
                            response_done = True
                            response = _maybe_enforce_primitive(msg, strat, args[0])
                            if not emit_raw:
                                for i in range(0, len(response), 180):
                                    send_nd({"type": "response_delta", "delta": response[i : i + 180]})
                            send_nd({"type": "widget_start"})
                        elif event_type == "complete":
                            if not response_done:
                                response = _maybe_enforce_primitive(msg, strat, args[0])
                                if not emit_raw:
                                    for i in range(0, len(response), 180):
                                        send_nd({"type": "response_delta", "delta": response[i : i + 180]})
                                send_nd({"type": "widget_start"})
                            widget_payload_raw = args[1]
                            break
                    elapsed = round(time.time() - t0, 1)
                    mode = "anthropic"
                    if not response and not widget_payload_raw:
                        response = "(No content)"
                    widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
                    widget_html = ""
                    widget_schema = ""
                    widget_height = 0
                    widget_debug = "streamed"
                    if widget_payload_raw:
                        if widget_mode == "json":
                            widget_schema, widget_html, widget_height, tag = _dispatch_json_mode_widget(
                                widget_payload_raw
                            )
                            widget_debug = tag
                        else:
                            raw_widget = widget_payload_raw
                            if "```" in raw_widget:
                                fence = re.search(r"```(?:json|html)?\s*(.*?)```", raw_widget, re.DOTALL | re.IGNORECASE)
                                raw_widget = fence.group(1).strip() if fence else re.sub(r"```\w*", "", raw_widget).strip()
                            if "<" in raw_widget and ">" in raw_widget:
                                if "<html" not in raw_widget.lower():
                                    raw_widget = f"<html><head></head><body>{raw_widget}</body></html>"
                                raw_widget = re.sub(r"<!DOCTYPE[^>]*>", "", raw_widget, flags=re.IGNORECASE).strip()
                                widget_payload_raw = inject_design_system(raw_widget)
                                if _looks_truncated_widget_html(widget_payload_raw):
                                    widget_debug = "stream_widget_truncated"
                                else:
                                    widget_html = widget_payload_raw
                                    widget_height = estimate_widget_height(widget_html)
                    if widget_mode != "json" and not widget_html:
                        widget_debug = "no_widget"
                else:
                    raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                msg = str(e).strip()
                if not msg:
                    msg = repr(e)
                send_done_error(f"LLM error ({type(e).__name__}): {msg}")
                return

            # Update history as in /api/chat.
            user["history"].append({"user": msg, "assistant": response})
            user["history"] = user["history"][-20:]
            user["last_message"]  = msg
            user["last_response"] = response
            user["last_strategy"] = strat
            user["last_x"]        = x.tolist()
            user["msg_count"]    += 1

            send_nd({
                "type": "done",
                "strategy": strat,
                "elapsed": elapsed,
                "llm_mode": mode,
                "response": response,
                "widget_html": widget_html or "",
                "widget_schema": widget_schema or "",
                "widget_height": widget_height,
                "widget_debug": widget_debug,
                "force_explore": force_explore,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            })
            return

        self.send_response(404); self.end_headers()


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_legacy_server():
    print("=" * 60)
    print(f"  http://localhost:5051   mode: {config.LLM_MODE}")
    print("=" * 60)

    PORT = int(os.getenv("PORT", "5051"))
    ThreadedServer(("0.0.0.0", PORT), Handler).serve_forever()
