"""LLM helpers for Anthropic and OpenAI-compatible endpoints.

This module centralizes HTTP calls and provides small helpers used by the
server to call either Anthropic (Claude) or an OpenAI-compatible API.
"""

import json
import urllib.request
import urllib.error
import socket
import time
import concurrent.futures
from typing import Tuple

from . import config

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None
    _anthropic_import_error = "import_failed"
else:
    _anthropic_import_error = ""


def _post_json_url(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    """POST JSON and return decoded response.

    Raises the underlying urllib errors to the caller for handling.
    """
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json_url(url: str, headers: dict | None = None, timeout: int = 15) -> dict:
    """GET from a URL and return the decoded JSON response.

    Args:
        url: Target URL.
        headers: Optional dict of additional HTTP headers.
        timeout: Request timeout in seconds.

    Returns:
        Decoded JSON response as a dict.
    """
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "Mozilla/5.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call_openai_compat(
    prompt: str,
    system: str,
    timeout: int = 120,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> Tuple[str, float, str]:
    """Call an OpenAI-compatible chat/completions endpoint.

    Returns (text, elapsed_seconds, mode).
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY env var")
    t0 = time.time()
    payload = {
        "model": config.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    data = _post_json_url(f"{config.OPENAI_BASE_URL}/chat/completions", payload, headers=headers, timeout=timeout)
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return (text or str(data)), round(time.time() - t0, 1), "openai_compat"


def call_anthropic(
    prompt: str,
    system: str,
    timeout: int = 120,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> Tuple[str, float, str]:
    """Call Anthropic Messages API (Claude) and return (text, elapsed_seconds, mode)."""
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var")
    if anthropic is None:
        raise RuntimeError(f"Missing `anthropic` package. Import error: {_anthropic_import_error}")

    fast_kwargs = {}
    if getattr(config, "ANTHROPIC_FAST_MODE_ENABLED", False):
        fast_kwargs = {
            # Fast Mode is controlled by the `anthropic-beta` header.
            "extra_headers": {"anthropic-beta": getattr(config, "ANTHROPIC_FAST_MODE_BETA", "fast-mode-2026-02-01")},
            # SDK may not expose `speed` directly in our version, but it can be set via extra_body.
            "extra_body": {"speed": getattr(config, "ANTHROPIC_FAST_MODE_SPEED", "fast")},
        }

    t0 = time.time()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Anthropic SDK handles timeouts internally; we keep our signature for parity.
    # Some SDK/provider combinations may ignore the SDK-level timeout and hang
    # for a long time. Enforce a wall-clock timeout so the server can respond
    # with an error instead of freezing the UI.
    def _create(with_fast: dict) -> object:
        return client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **with_fast,
        )

    msg = None
    try:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(_create, fast_kwargs)
            msg = fut.result(timeout=timeout)
        finally:
            # Do not block waiting for a stuck SDK call to finish.
            ex.shutdown(wait=False, cancel_futures=True)
    except concurrent.futures.TimeoutError as e:
        raise RuntimeError(f"Anthropic request timed out after {timeout}s") from e
    except Exception:
        # If Fast Mode beta isn't accepted for this request/model/SDK version, retry normally.
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(_create, {})
            msg = fut.result(timeout=timeout)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    # `content` is typically a list of blocks; concatenate everything we can.
    # This avoids missing parts of the model output if some blocks don't expose
    # `.text` (e.g., non-text block representations).
    texts: list[str] = []
    for block in getattr(msg, "content", []) or []:
        t = getattr(block, "text", None)
        if t is not None:
            texts.append(str(t))
        else:
            texts.append(str(block))
    text = "\n".join([t for t in texts if t]).strip()
    if not text:
        # Fallback: stringify response object.
        text = str(msg).strip()

    return text, round(time.time() - t0, 1), "anthropic"


def stream_anthropic(
    prompt: str,
    system: str,
    timeout: int = 120,
    max_tokens: int = 400,
    temperature: float = 0.2,
):
    """Yield Anthropic content deltas (token-by-token).

    Yields:
        str chunks of text from `content_block_delta`.
    """
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var")
    if anthropic is None:
        raise RuntimeError(f"Missing `anthropic` package. Import error: {_anthropic_import_error}")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    fast_kwargs = {}
    if getattr(config, "ANTHROPIC_FAST_MODE_ENABLED", False):
        fast_kwargs = {
            "extra_headers": {"anthropic-beta": getattr(config, "ANTHROPIC_FAST_MODE_BETA", "fast-mode-2026-02-01")},
            "extra_body": {"speed": getattr(config, "ANTHROPIC_FAST_MODE_SPEED", "fast")},
        }

    try:
        with client.messages.stream(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **fast_kwargs,
        ) as stream:
            for event in stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    chunk = getattr(delta, "text", None) if delta is not None else None
                    if chunk:
                        yield str(chunk)
    except Exception:
        # Retry without fast-mode params.
        with client.messages.stream(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        ) as stream:
            for event in stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    chunk = getattr(delta, "text", None) if delta is not None else None
                    if chunk:
                        yield str(chunk)


def openai_health(timeout: int = 10) -> dict:
    """Return basic health info for an OpenAI-compatible endpoint."""
    try:
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"} if config.OPENAI_API_KEY else {}
        data = _get_json_url(f"{config.OPENAI_BASE_URL}/models", headers=headers, timeout=timeout)
        ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
        return {"ok": True, "reachable": True, "models": ids[:25]}
    except Exception as e:
        return {"ok": False, "reachable": False, "models": [], "error": str(e)}


def anthropic_health() -> dict:
    """Return basic health info for Anthropic configuration.

    Anthropic does not expose a simple unauthenticated models endpoint like OpenAI.
    We keep this as a lightweight config check.
    """
    ok = bool(getattr(config, "ANTHROPIC_API_KEY", "") and getattr(config, "ANTHROPIC_MODEL", ""))
    return {"ok": ok, "reachable": ok, "model": getattr(config, "ANTHROPIC_MODEL", "")}
