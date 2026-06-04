"""Step 2 — build the widget JSON DIRECTLY from the synthesized response.

No evidence packet: the answer IS the source. The builder may only visualize
numbers/facts explicitly stated in the answer, using the exact numbers as written.
The validator then checks that every widget number actually appears in the answer.
"""

from __future__ import annotations

from .llm import complete

_SYS = (
    "You are a widget builder for a finance dashboard. You turn an analyst ANSWER into a "
    "STRICT JSON widget. Hard rules:\n"
    "- Visualize ONLY facts and numbers EXPLICITLY stated in the answer. Do NOT add numbers, "
    "periods, entities, or claims that are not written in the answer.\n"
    "- Use the EXACT numbers as written in the answer (same value; drop currency symbols/commas, "
    "keep the scale, e.g. write 1177.5 for \"$1,177.5M\").\n"
    "- If the answer states no chartable numbers, return {\"layout\": []}.\n"
    "- Output RAW JSON only. No markdown, no comments, no prose.\n\n"
    "Schema:\n"
    "{ \"layout\": [\n"
    "  { \"type\": \"text\", \"content\": \"...\" },\n"
    "  { \"type\": \"kpi_row\", \"items\": [ { \"label\": \"...\", \"value\": <number>, \"unit\": \"...\" } ] },\n"
    "  { \"type\": \"chart\", \"title\": \"...\", \"kind\": \"line|bar|pie|donut\", \"points\": [ { \"label\": \"...\", \"value\": <number> } ] }\n"
    "] }\n"
    "Choose kind: line/bar for trends or comparisons over periods; pie/donut for composition/shares."
)

_FORMAT_GUIDE = {
    "kpi_row": "Present the main block as a kpi_row of the key metrics stated in the answer.",
    "line": "Present the main block as a chart with kind 'line' (trend over periods).",
    "bar": "Present the main block as a chart with kind 'bar' (comparison).",
    "area": "Present the main block as a chart with kind 'area' (cumulative trend).",
    "pie": "Present the main block as a chart with kind 'pie' (composition / share).",
    "donut": "Present the main block as a chart with kind 'donut' (composition / share).",
    "text": "Return only a single text block summarizing the answer (no chart/kpi).",
}


def build_widget(query: str, response: str, chosen_format: str = "") -> str:
    fmt_line = ""
    if chosen_format:
        guide = _FORMAT_GUIDE.get(chosen_format, f"Use format '{chosen_format}'.")
        fmt_line = (
            f"\nPRESENTATION FORMAT (chosen by the adaptive engine): {chosen_format}\n"
            f"{guide} You may add ONE short text block for framing. Do not use other chart kinds.\n"
        )
    user = (
        f"USER QUESTION:\n{query}\n\n"
        f"ANALYST ANSWER (visualize ONLY the numbers/facts written here):\n{response}\n"
        f"{fmt_line}\n"
        "Build the widget JSON now using only numbers present in the answer. "
        "Return {\"layout\": []} if the answer has no chartable numbers."
    )
    return complete(_SYS, user, max_tokens=1500, temperature=0.0)
