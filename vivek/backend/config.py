"""Configuration and constants for the Adaptive Presentation Engine backend.

Loads environment variables from .env (via python-dotenv) and defines runtime
configuration: LLM provider selection, API keys, Thompson Sampling params, paths.
Restart the server after changing .env.
"""

from pathlib import Path
from dotenv import load_dotenv
import json
import os
import threading

# Load .env into os.environ; override=True lets existing env vars win (Docker/CI).
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# LLM provider and endpoint routing
# -----------------------------------------------------------------------------
# Primary mode: "anthropic" (Claude) or "openai_compat" (OpenAI/Groq).
LLM_MODE = os.getenv("LLM_MODE", "anthropic").lower()

# When true (default), strategy primitives are mandatory in prompts and
# enforce_response() reshapes <RESPONSE> to match. Set STRICT_PRIMITIVES=0 for soft hints only.
STRICT_PRIMITIVES = os.getenv("STRICT_PRIMITIVES", "1").strip().lower() in {"1", "true", "yes", "on"}

# Per-endpoint overrides: /api/chat_plain -> BASELINE, /api/chat -> ADAPTIVE.
BASELINE_LLM_MODE = os.getenv("BASELINE_LLM_MODE", LLM_MODE).lower()
ADAPTIVE_LLM_MODE = os.getenv("ADAPTIVE_LLM_MODE", LLM_MODE).lower()

# Widget output: "html" (full HTML in iframe) or "json" (UI schema for frontend renderer).
WIDGET_MODE = os.getenv("WIDGET_MODE", "json").strip().lower()

# Max time (sec) and tokens for combined response+widget generation.
COMBINED_TIMEOUT_SECONDS = int(os.getenv("COMBINED_TIMEOUT_SECONDS", "30"))
COMBINED_MAX_TOKENS = int(os.getenv("COMBINED_MAX_TOKENS", "2800"))

# OpenAI-compatible API (Groq, OpenAI, or any chat/completions provider)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")

# Anthropic Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

# Anthropic Fast Mode: beta header for faster inference; we retry on failure.
ANTHROPIC_FAST_MODE_ENABLED = os.getenv("ANTHROPIC_FAST_MODE_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
ANTHROPIC_FAST_MODE_SPEED = os.getenv("ANTHROPIC_FAST_MODE_SPEED", "fast")  # "fast" or "standard"
ANTHROPIC_FAST_MODE_BETA = os.getenv("ANTHROPIC_FAST_MODE_BETA", "fast-mode-2026-02-01")

# Thompson Sampling: D=feature dim, LAMBDA=reg, GAMMA=decay, ALPHA_G=global prior strength.
D = 10
LAMBDA = 0.01
GAMMA = 0.99
ALPHA_G = 0.05

# Base temperature for Thompson Sampling (exploration).
TS_TEMPERATURE = float(os.getenv("TS_TEMPERATURE", "2.0"))

# On strong negative feedback: boost temp, penalize last strategy, damp its posterior.
NEG_EXPLORE_THRESHOLD = float(os.getenv("NEG_EXPLORE_THRESHOLD", "0.40"))
EXPLORE_TEMP_BOOST = float(os.getenv("EXPLORE_TEMP_BOOST", "2.0"))
EXPLORE_SCORE_PENALTY = float(os.getenv("EXPLORE_SCORE_PENALTY", "1.75"))

NEG_MU_SHRINK = float(os.getenv("NEG_MU_SHRINK", "0.25"))
NEG_SINV_SHRINK = float(os.getenv("NEG_SINV_SHRINK", "0.35"))

# Strategy primitives: style constraints injected into LLM prompts.
# These are bootstrapped defaults. At runtime we may override them from
# `strategies.json` to make the bandit "strategy bars" admin-manageable.
DEFAULT_STRATEGIES = {
    "structured_bullets": "Use 3-5 bullet points only (start each line with '- '). No intro sentence. Do NOT ask questions. Do NOT use numbered lists.",
    "narrative_prose":    "Write 2-3 short paragraphs. No bullet points.",
    "concise_direct":     "Reply in at most 3 sentences. Be direct.",
    "socratic_questions": "Brief acknowledgement, then ask 1-2 clarifying questions.",
    "step_by_step":       "Numbered list of 3-6 steps only.",
    # NEW
    "comparison_table":   "Return a single MARKDOWN TABLE only. Use columns that help compare options (e.g., Option | Pros | Cons | Best for). No bullets outside the table.",
    "visualization":      "Return a simple TEXT visualization only (ASCII bar chart or small table-of-values). Put it in a fenced code block. No extra prose outside the code block.",
}

# Module-level strategy state (may be overwritten by `reload_strategies()`).
# - `STRATEGY_ITEMS`: all strategies (enabled + disabled), keyed by id.
# - `STRATEGIES`: instruction map for all strategies, keyed by id.
# - `STRATEGY_NAMES`: enabled strategy ids (used by bandit selection/summary loops).
STRATEGY_ITEMS: dict[str, dict] = {}
STRATEGIES: dict[str, str] = dict(DEFAULT_STRATEGIES)
STRATEGY_NAMES: list[str] = list(DEFAULT_STRATEGIES.keys())
STRATEGY_LABELS: dict[str, str] = {
    "structured_bullets": "Structured Bullets",
    "narrative_prose": "Narrative Prose",
    "concise_direct": "Concise & Direct",
    "socratic_questions": "Socratic Questions",
    "step_by_step": "Step-by-Step",
    "comparison_table": "Comparison Table",
    "visualization": "Visualization",
}
K = len(STRATEGY_NAMES)

# Paths: HERE = backend/, parent = vivek/ (project root).
HERE = Path(__file__).resolve().parent
# Frontend entry point served at GET /. Prefer the Vite production build.
_FRONTEND_VUE_DIST_INDEX = HERE.parent / "frontend-vue" / "dist" / "index.html"
INDEX_HTML = (
    _FRONTEND_VUE_DIST_INDEX
    if _FRONTEND_VUE_DIST_INDEX.exists()
    else HERE.parent / "frontend" / "index.html"
)

# Skills document — high-level chart/widget guidance injected into LLM prompts.
# Override with SKILLS_PATH env var if needed.
SKILLS_PATH = Path(os.getenv("SKILLS_PATH", str(HERE.parent / "skills.md")))
SKILLS_CONTENT = SKILLS_PATH.read_text(encoding="utf-8") if SKILLS_PATH.exists() else ""

# -----------------------------------------------------------------------------
# Dynamic strategy store (admin-manageable bandit bars)
# -----------------------------------------------------------------------------
STRATEGIES_JSON_PATH = os.getenv("STRATEGIES_JSON_PATH", str(HERE.parent / "strategies.json"))
# RLock because `persist_strategies()` calls `reload_strategies()` which also acquires the lock.
_STRATEGIES_LOCK = threading.RLock()

def _read_strategies_json() -> dict:
    """
    strategies.json format:
      {
        "version": 1,
        "items": [
          { "id": "...", "label": "...", "instruction": "...", "enabled": true, "event_name": "Decision" }
        ]
      }
    """
    try:
        with open(STRATEGIES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def reload_strategies() -> None:
    """
    Reload `strategies.json` into module-level globals.
    Safe to call multiple times.
    """
    global STRATEGY_ITEMS, STRATEGIES, STRATEGY_NAMES, STRATEGY_LABELS, K

    with _STRATEGIES_LOCK:
        data = _read_strategies_json()
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            # Bootstrapped defaults (no dynamic strategies yet).
            STRATEGY_ITEMS = {k: {"id": k, "instruction": v, "label": STRATEGY_LABELS.get(k, k), "enabled": True, "event_name": ""} for k, v in DEFAULT_STRATEGIES.items()}
            STRATEGIES = dict(DEFAULT_STRATEGIES)
            STRATEGY_NAMES = list(DEFAULT_STRATEGIES.keys())
            K = len(STRATEGY_NAMES)
            return

        next_items: dict[str, dict] = {}
        next_strategies: dict[str, str] = {}
        next_labels: dict[str, str] = {}
        enabled_ids: list[str] = []

        for it in items:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("id") or "").strip()
            instruction = str(it.get("instruction") or "").strip()
            if not sid or not instruction:
                continue
            label = str(it.get("label") or sid).strip()
            enabled = bool(it.get("enabled", True))
            event_name = str(it.get("event_name") or "").strip()

            next_items[sid] = {"id": sid, "label": label, "instruction": instruction, "enabled": enabled, "event_name": event_name}
            next_strategies[sid] = instruction
            next_labels[sid] = label
            if enabled:
                enabled_ids.append(sid)

        # If admin disabled everything, keep at least the bootstrap defaults enabled to avoid breaking TS loops.
        if not enabled_ids:
            enabled_ids = list(DEFAULT_STRATEGIES.keys())
            next_items = {k: {"id": k, "label": STRATEGY_LABELS.get(k, k), "instruction": v, "enabled": True, "event_name": ""} for k, v in DEFAULT_STRATEGIES.items()}
            next_strategies = dict(DEFAULT_STRATEGIES)
            next_labels = dict(STRATEGY_LABELS)

        STRATEGY_ITEMS = next_items
        STRATEGIES = next_strategies
        STRATEGY_LABELS = next_labels
        STRATEGY_NAMES = enabled_ids
        K = len(STRATEGY_NAMES)


def _atomic_write_strategies_json(data: dict) -> None:
    tmp = f"{STRATEGIES_JSON_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STRATEGIES_JSON_PATH)


def persist_strategies(items: list[dict]) -> None:
    """
    Persist `strategies.json` then reload module globals.
    `items` should be a list of:
      {id, label, instruction, enabled, event_name}
    """
    payload = {"version": 1, "items": items}
    with _STRATEGIES_LOCK:
        _atomic_write_strategies_json(payload)
        reload_strategies()


# Initial load at import time.
reload_strategies()

# -----------------------------------------------------------------------------
# Auth (JWT)
# -----------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-me-please")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", str(60 * 60 * 24 * 7)))

# -----------------------------------------------------------------------------
# Admin authorization (primitives are admin-only)
# -----------------------------------------------------------------------------
# Admins can manage primitives and have their primitives injected into the LLM prompt.
#
# Setup options:
# 1) Provide a single admin to auto-create on startup:
#    - ADMIN_USERNAME
#    - ADMIN_EMAIL
#    - ADMIN_PASSWORD
# 2) Or mark existing users as admin via lists:
#    - ADMIN_USERNAMES (comma-separated)
#    - ADMIN_EMAILS (comma-separated)
# 3) Or mark specific users by id:
#    - ADMIN_USER_IDS (comma-separated)
#
# If no admin config is provided, the backend keeps the previous behavior
# (any authenticated user can access primitives).
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

ADMIN_USERNAMES = _split_csv(os.getenv("ADMIN_USERNAMES", "")) or ([ADMIN_USERNAME] if ADMIN_USERNAME else [])
ADMIN_EMAILS = _split_csv(os.getenv("ADMIN_EMAILS", "")) or ([ADMIN_EMAIL] if ADMIN_EMAIL else [])
ADMIN_USER_IDS = _split_csv(os.getenv("ADMIN_USER_IDS", ""))

# Used by server-side gating logic.
ADMIN_CONFIGURED = bool(ADMIN_USERNAMES or ADMIN_EMAILS or ADMIN_USER_IDS)

# SQLite persistence
DB_PATH = os.getenv("DB_PATH", str(HERE.parent / "adaptiveui.sqlite3"))

# JSON persistence for user primitives (when DB/SQL is not desired).
PRIMITIVES_JSON_PATH = os.getenv("PRIMITIVES_JSON_PATH", str(HERE.parent / "user_primitives.json"))
