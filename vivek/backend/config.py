"""Configuration and constants for the Adaptive Presentation Engine backend.

Loads environment variables from .env (via python-dotenv) and defines runtime
configuration: LLM provider selection, API keys, Thompson Sampling params, paths.
Restart the server after changing .env.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env into os.environ; override=True lets existing env vars win (Docker/CI).
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# LLM provider and endpoint routing
# -----------------------------------------------------------------------------
# Primary mode: "anthropic" (Claude) or "openai_compat" (OpenAI/Groq).
LLM_MODE = os.getenv("LLM_MODE", "anthropic").lower()

# If false (default), strategy “primitives” are treated as soft style hints
# and the backend will not aggressively post-process text to enforce them.
STRICT_PRIMITIVES = os.getenv("STRICT_PRIMITIVES", "0").strip().lower() in {"1", "true", "yes", "on"}

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
STRATEGIES = {
    "structured_bullets": "Use 3-5 bullet points only (start each line with '- '). No intro sentence. Do NOT ask questions. Do NOT use numbered lists.",
    "narrative_prose":    "Write 2-3 short paragraphs. No bullet points.",
    "concise_direct":     "Reply in at most 3 sentences. Be direct.",
    "socratic_questions": "Brief acknowledgement, then ask 1-2 clarifying questions.",
    "step_by_step":       "Numbered list of 3-6 steps only.",
    # NEW
    "comparison_table":   "Return a single MARKDOWN TABLE only. Use columns that help compare options (e.g., Option | Pros | Cons | Best for). No bullets outside the table.",
    "visualization":      "Return a simple TEXT visualization only (ASCII bar chart or small table-of-values). Put it in a fenced code block. No extra prose outside the code block.",
}
STRATEGY_NAMES = list(STRATEGIES.keys())
K = len(STRATEGY_NAMES)

# Paths: HERE = backend/, parent = vivek/ (project root).
HERE = Path(__file__).resolve().parent
# Frontend entry point served at GET /. Modular layout: frontend/ holds all UI assets.
INDEX_HTML = HERE.parent / "frontend" / "index.html"

# Skills document — high-level chart/widget guidance injected into LLM prompts.
# Override with SKILLS_PATH env var if needed.
SKILLS_PATH = Path(os.getenv("SKILLS_PATH", str(HERE.parent / "skills.md")))
SKILLS_CONTENT = SKILLS_PATH.read_text(encoding="utf-8") if SKILLS_PATH.exists() else ""
