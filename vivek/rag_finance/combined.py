"""Single-call RAG generation: ONE LLM call produces BOTH the grounded answer and
the widget JSON. The allowed block types + chart kinds are read from the SINGLE
source of truth (widget-registry.json) so generate stays in sync with validate/render.

Output format:
  <RESPONSE> grounded answer </RESPONSE>
  <WIDGET>   {"layout": [...]} </WIDGET>
"""

from __future__ import annotations

import re

from .llm import complete
from .registry import allowed_chart_kinds, allowed_types


def _build_system_prompt() -> str:
    types = ", ".join(allowed_types())            # ← from widget-registry.json
    kinds = ", ".join(allowed_chart_kinds())      # ← from widget-registry.json
    return (
        "You are a financial analyst assistant AND a widget builder. Using ONLY the provided context "
        "(retrieved filings), produce TWO sections, in this exact order:\n\n"
        "<RESPONSE>\n"
        "A grounded answer (3-6 sentences). Quote figures exactly as they appear in the context. "
        "Never invent numbers, companies, or periods. If the context lacks the answer, say so.\n"
        "</RESPONSE>\n"
        "<WIDGET>\n"
        "A STRICT JSON widget that visualizes ONLY numbers you stated in <RESPONSE>.\n"
        "</WIDGET>\n\n"
        "WIDGET rules:\n"
        "- Inside <WIDGET> output RAW JSON only (no markdown, no comments).\n"
        f"- Use ONLY these block types: {types}.\n"
        f"- chart \"kind\" must be ONE of: {kinds}.\n"
        "- Every numeric value MUST also appear in your <RESPONSE>. Do not introduce new numbers.\n"
        "- If <RESPONSE> states no chartable numbers, output <WIDGET>{\"layout\": []}</WIDGET>.\n\n"
        "Schema:\n"
        "{ \"layout\": [\n"
        "  { \"type\": \"text\", \"content\": \"...\" },\n"
        "  { \"type\": \"kpi_row\", \"items\": [ { \"label\": \"...\", \"value\": <number>, \"unit\": \"...\" } ] },\n"
        "  { \"type\": \"chart\", \"title\": \"...\", \"kind\": \"<one kind>\", \"points\": [ { \"label\": \"...\", \"value\": <number> } ] }\n"
        "] }\n\n"
        "Chart guidance:\n"
        "- SINGLE series: use \"points\": [{\"label\",\"value\"}].\n"
        "- MULTIPLE series (stacked, combo, grouped, radar): use\n"
        "  \"series\": [ { \"name\": \"...\", \"kind\": \"bar|line\"(combo only), \"points\": [ {\"label\",\"value\"} ] }, ... ]\n"
        "  (all series share the same labels). For \"combo\", mark each series kind bar or line.\n"
        "- Pick kind by content: line/area=trend; bar/hbar=comparison; stacked=parts over periods; "
        "pie/donut/funnel=composition or stages; gauge=single metric; radar=multi-dimension; combo=mix bar+line."
    )


def _extract(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def generate(query: str, chunks: list[str]) -> tuple[str, str]:
    """Returns (response_text, widget_raw_json) from ONE LLM call."""
    context = "\n\n---\n\n".join(chunks)
    user = (
        f"CONTEXT (retrieved filings):\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Produce <RESPONSE> then <WIDGET> exactly as specified. Use only the context for the answer, "
        "and only numbers stated in your <RESPONSE> for the widget."
    )
    raw = complete(_build_system_prompt(), user, max_tokens=2000, temperature=0.1)
    response = _extract(raw, "RESPONSE") or raw.strip()
    widget = _extract(raw, "WIDGET") or '{"layout": []}'
    return response, widget
