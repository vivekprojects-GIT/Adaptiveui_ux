"""Incremental parser for combined <RESPONSE>/<WIDGET> output.

Unlike the previous parser which buffered the entire widget body before
emitting anything, this yields widget deltas as they arrive — giving the
frontend Claude-style, token-by-token widget streaming.

Event types yielded
-------------------
- ("response_delta", str)  : raw response tokens (inside <RESPONSE>)
- ("response_closed", str) : full response text when </RESPONSE> is seen
- ("widget_start", "")     : first token of <WIDGET> opens
- ("widget_delta", str)    : raw widget tokens (inside <WIDGET>)
- ("complete", response_text, widget_raw) : after </WIDGET> or EOF

Callers decide whether to forward response_delta (strict-primitive mode may
prefer to buffer until the full response is known before re-streaming it).
"""

from __future__ import annotations

from typing import Generator, Iterable, Tuple


_RESPONSE_OPEN = "<RESPONSE>"
_RESPONSE_CLOSE = "</RESPONSE>"
_WIDGET_OPEN = "<WIDGET>"
_WIDGET_CLOSE = "</WIDGET>"

# Keep enough buffer around tag-ends to avoid splitting a tag across emits.
_TAG_SAFETY = max(len(_RESPONSE_CLOSE), len(_WIDGET_CLOSE))


def parse_combined_stream(
    chunks: Iterable[str],
    *,
    emit_response_deltas: bool = True,
    emit_widget_deltas: bool = True,
) -> Generator[Tuple[str, ...], None, None]:
    """Yield structured events from an iterable of raw LLM chunks.

    The function is a pure state machine: preamble -> response -> widget_pending ->
    widget -> complete. Each chunk is appended to a rolling buffer and emitted
    up to a safe boundary so we never emit half of a closing tag.
    """
    buf = ""
    state = "preamble"
    widget_started_emitted = False
    response_text = ""
    response_emitted_len = 0
    widget_text = ""
    widget_emitted_len = 0

    def _find(needle: str, haystack_upper: str) -> int:
        return haystack_upper.find(needle)

    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        buf_upper = buf.upper()

        # --- Enter <RESPONSE> ---
        if state == "preamble":
            idx = _find(_RESPONSE_OPEN, buf_upper)
            if idx != -1:
                buf = buf[idx + len(_RESPONSE_OPEN):]
                buf_upper = buf.upper()
                state = "response"
                response_emitted_len = 0

        # --- Stream <RESPONSE> content until </RESPONSE> ---
        if state == "response":
            close_idx = _find(_RESPONSE_CLOSE, buf_upper)
            if close_idx != -1:
                to_send = buf[:close_idx][response_emitted_len:]
                if to_send and emit_response_deltas:
                    yield ("response_delta", to_send)
                response_text = buf[:close_idx].strip()
                buf = buf[close_idx + len(_RESPONSE_CLOSE):]
                buf_upper = buf.upper()
                state = "widget_pending"
                yield ("response_closed", response_text)
            else:
                safe_len = max(0, len(buf) - _TAG_SAFETY)
                if safe_len > response_emitted_len:
                    delta = buf[response_emitted_len:safe_len]
                    if emit_response_deltas:
                        yield ("response_delta", delta)
                    response_emitted_len = safe_len

        # --- Wait for <WIDGET> tag open ---
        if state == "widget_pending":
            idx = _find(_WIDGET_OPEN, buf_upper)
            if idx != -1:
                buf = buf[idx + len(_WIDGET_OPEN):]
                buf_upper = buf.upper()
                state = "widget"
                widget_emitted_len = 0
                if not widget_started_emitted:
                    widget_started_emitted = True
                    yield ("widget_start", "")

        # --- Stream <WIDGET> content until </WIDGET> ---
        if state == "widget":
            close_idx = _find(_WIDGET_CLOSE, buf_upper)
            if close_idx != -1:
                to_send = buf[:close_idx][widget_emitted_len:]
                if to_send and emit_widget_deltas:
                    yield ("widget_delta", to_send)
                widget_text = buf[:close_idx].strip()
                yield ("complete", response_text, widget_text)
                return
            safe_len = max(0, len(buf) - _TAG_SAFETY)
            if safe_len > widget_emitted_len:
                delta = buf[widget_emitted_len:safe_len]
                if emit_widget_deltas:
                    yield ("widget_delta", delta)
                widget_emitted_len = safe_len

    # Stream ended without clean closure — salvage what we can.
    if state == "response":
        remaining = buf[response_emitted_len:]
        if remaining and emit_response_deltas:
            yield ("response_delta", remaining)
        response_text = (response_text or buf).strip()
    elif state == "widget":
        remaining = buf[widget_emitted_len:]
        if remaining and emit_widget_deltas:
            yield ("widget_delta", remaining)
        widget_text = (widget_text or buf).strip()
    elif state == "preamble" and buf.strip():
        # Model didn't use XML tags — treat the whole thing as the response.
        response_text = buf.strip()
        if emit_response_deltas:
            yield ("response_delta", response_text)

    yield ("complete", response_text, widget_text)
