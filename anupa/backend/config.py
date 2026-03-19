"""Configuration and constants for the Adaptive Presentation Engine backend."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(override=True)
# Runtime mode: 'openai_compat' uses an OpenAI-compatible remote API,
# otherwise the code will prefer Anthropic (Claude) endpoints.
LLM_MODE = os.getenv("LLM_MODE", "openai_compat").lower()

# If false (default), strategy “primitives” are treated as soft style hints
# and the backend will not aggressively post-process text to enforce them.
STRICT_PRIMITIVES = os.getenv("STRICT_PRIMITIVES", "0").strip().lower() in {"1", "true", "yes", "on"}

# Provider routing:
# - /api/chat_plain uses BASELINE_LLM_MODE
# - /api/chat uses ADAPTIVE_LLM_MODE
BASELINE_LLM_MODE = os.getenv("BASELINE_LLM_MODE", LLM_MODE).lower()
ADAPTIVE_LLM_MODE = os.getenv("ADAPTIVE_LLM_MODE", LLM_MODE).lower()

# Widget rendering mode:
# - "html": model outputs full HTML in <WIDGET> (rendered in iframe)
# - "json": model outputs JSON UI schema in <WIDGET> (rendered by frontend renderer)
WIDGET_MODE = os.getenv("WIDGET_MODE", "json").strip().lower()

# Combined (Claude-style) generation limits
COMBINED_TIMEOUT_SECONDS = int(os.getenv("COMBINED_TIMEOUT_SECONDS", "30"))
COMBINED_MAX_TOKENS = int(os.getenv("COMBINED_MAX_TOKENS", "2800"))

# OpenAI-compatible endpoint (Groq / other providers that offer OpenAI API)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")

# Anthropic (Claude) API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

# Anthropic Fast Mode (optional; safe to enable because we retry on failure)
# Implementation uses the Anthropic beta header + request body speed field via SDK-supported
# `extra_headers` / `extra_body`.
ANTHROPIC_FAST_MODE_ENABLED = os.getenv("ANTHROPIC_FAST_MODE_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
ANTHROPIC_FAST_MODE_SPEED = os.getenv("ANTHROPIC_FAST_MODE_SPEED", "fast")  # "fast" or "standard"
ANTHROPIC_FAST_MODE_BETA = os.getenv("ANTHROPIC_FAST_MODE_BETA", "fast-mode-2026-02-01")

# Model / engine hyperparameters
D = 10
LAMBDA = 0.01
GAMMA = 0.99
ALPHA_G = 0.05

# Base Thompson temperature (used when not forcing exploration)
TS_TEMPERATURE = float(os.getenv("TS_TEMPERATURE", "2.0"))

# ---- Exploration / correction knobs (Option B) ----
# When we detect strong negative feedback, we:
#  - boost temperature (more exploration)
#  - penalize repeating the last strategy (strongly)
#  - damp the posterior for the last strategy (reduce confidence)
NEG_EXPLORE_THRESHOLD = float(os.getenv("NEG_EXPLORE_THRESHOLD", "0.40"))  # ev["neg"] >= this => explore
EXPLORE_TEMP_BOOST = float(os.getenv("EXPLORE_TEMP_BOOST", "2.0"))
EXPLORE_SCORE_PENALTY = float(os.getenv("EXPLORE_SCORE_PENALTY", "1.75"))

# Posterior damping strength (scaled by neg_strength in [0,1])
NEG_MU_SHRINK = float(os.getenv("NEG_MU_SHRINK", "0.25"))          # shrink mean magnitude
NEG_SINV_SHRINK = float(os.getenv("NEG_SINV_SHRINK", "0.35"))      # shrink precision -> bigger covariance

# Strategy primitives and descriptions used to constrain generation style
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

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE.parent / "index.html"
