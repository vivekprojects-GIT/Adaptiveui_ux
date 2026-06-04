"""Anthropic client for the RAG app (reuses the project's .env, with the auth fix)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project .env (one level up).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Some environments export an EMPTY ANTHROPIC_AUTH_TOKEN, which makes the SDK send an
# invalid "Bearer " header and fail every call with APIConnectionError. Drop blanks so
# it falls back to ANTHROPIC_API_KEY (x-api-key).
for _v in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS"):
    if not (os.environ.get(_v) or "").strip():
        os.environ.pop(_v, None)

import anthropic  # noqa: E402

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def complete(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> str:
    """Single-shot completion; returns concatenated text."""
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text").strip()
