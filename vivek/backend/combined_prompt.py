"""Single-call combined response + widget generation — Claude-style architecture.

Instead of two sequential LLM calls (response then widget), this module
lets GPT-4 generate BOTH in one pass:

  Output format:
    <RESPONSE>
    [answer text here — follows primitive format rule]
    </RESPONSE>
    <WIDGET>
    [either complete self-contained HTML OR JSON UI schema (depending on WIDGET_MODE)]
    </WIDGET>

The parser splits these apart. Text goes to chat; widget payload goes to renderer.
This matches Claude's architecture: one model, one generation, no sequential delay.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Tuple

from . import config


def strip_widget_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` wrappers often emitted inside <WIDGET> despite instructions."""
    s = (raw or "").strip()
    if not s or "```" not in s:
        return s
    fence = re.search(r"```(?:json|html|javascript|js)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return re.sub(r"```\w*", "", s).strip()


def parse_widget_schema_object(s: str) -> Any | None:
    """Parse first JSON value; tolerate leading/trailing prose via JSONDecoder.raw_decode."""
    s = strip_widget_markdown_fences(s).strip()
    if not s:
        return None
    dec = json.JSONDecoder()
    for start_ch in ("{", "["):
        idx = s.find(start_ch)
        if idx == -1:
            continue
        try:
            obj, _ = dec.raw_decode(s, idx)
            return obj
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


_BLOCK_TYPES = frozenset({"text", "kpi_row", "chart", "table", "action_row", "image", "stat_card", "progress", "badge_row"})

_TYPE_ALIASES: dict[str, str] = {
    "markdown": "text",
    "md": "text",
    "paragraph": "text",
    "rich_text": "text",
    "kpi": "kpi_row",
    "kpis": "kpi_row",
    "kpirow": "kpi_row",
    "metrics": "kpi_row",
    "metric_row": "kpi_row",
    "data_table": "table",
    "grid": "table",
    "actions": "action_row",
    "action": "action_row",
    "plot": "chart",
    "graph": "chart",
    "line_chart": "chart",
    "bar_chart": "chart",
    "photo": "image",
    "picture": "image",
    "img": "image",
    "illustration": "image",
}


def _normalize_layout_block(entry: Any) -> dict[str, Any]:
    """Turn primitives, malformed entries, and common aliases into valid block dicts."""
    if entry is None:
        return {"type": "text", "content": ""}
    if isinstance(entry, (str, int, float, bool)):
        return {"type": "text", "content": str(entry)}
    if isinstance(entry, list):
        return {"type": "text", "content": json.dumps(entry, ensure_ascii=False)}
    if not isinstance(entry, dict):
        return {"type": "text", "content": str(entry)}

    b: dict[str, Any] = dict(entry)
    raw_t = str(b.get("type") or "").strip().lower().replace("-", "_")
    t = _TYPE_ALIASES.get(raw_t, raw_t)
    if t in _BLOCK_TYPES:
        b["type"] = t
        return b

    items = b.get("items")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict) and isinstance(first.get("label"), str):
            v = first.get("value")
            if isinstance(v, (str, int, float, bool)):
                b["type"] = "kpi_row"
                return b
    if isinstance(b.get("chart"), dict):
        b["type"] = "chart"
        return b
    src_val = b.get("src")
    if isinstance(src_val, str) and src_val.strip():
        b["type"] = "image"
        return b
    if isinstance(b.get("rows"), list):
        b["type"] = "table"
        return b
    if isinstance(b.get("buttons"), list):
        b["type"] = "action_row"
        return b
    for key in ("content", "body", "text", "markdown", "md"):
        v = b.get(key)
        if isinstance(v, str):
            return {"type": "text", "content": v}

    return {"type": "text", "content": json.dumps(b, ensure_ascii=False)}


def _sanitize_layout(layout: list[Any]) -> list[dict[str, Any]]:
    return [_normalize_layout_block(e) for e in layout]


# ── Registry-driven validation (keeps streaming; cleans the FINAL schema) ────
# Reads the SAME widget-registry.json used by the renderer to (1) drop block
# types the renderer can't draw, and (2) drop blocks with no renderable data
# (e.g. a chart with an unsupported kind or no data) — so the finalized widget
# never shows a blank/garbage block.
_REGISTRY_CACHE: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        try:
            _REGISTRY_CACHE = json.loads(Path(__file__).resolve().parent.parent / "frontend-vue" / "src" / "widget-registry.json")  # type: ignore[arg-type]
        except Exception:
            try:
                _REGISTRY_CACHE = json.loads(
                    (Path(__file__).resolve().parent.parent / "frontend-vue" / "src" / "widget-registry.json").read_text(encoding="utf-8")
                )
            except Exception:
                _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE or {}


def _registry_block_types() -> set[str]:
    blocks = _load_registry().get("blocks") or []
    types = {str(b.get("type")).lower() for b in blocks if isinstance(b, dict) and b.get("type")}
    return types or set(_BLOCK_TYPES)


def _registry_chart_kinds() -> set[str]:
    for b in _load_registry().get("blocks") or []:
        if isinstance(b, dict) and str(b.get("type")).lower() == "chart":
            kinds = b.get("kinds")
            if isinstance(kinds, list) and kinds:
                return {str(k).lower() for k in kinds}
    return {"line", "bar", "area", "scatter", "heatmap", "pie", "donut"}


def _nonempty_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0


def _chart_has_data(chart: dict[str, Any]) -> bool:
    kind = str(chart.get("kind") or "line").lower()
    if kind not in _registry_chart_kinds():
        return False
    if kind == "heatmap":
        m = chart.get("matrix")
        return isinstance(m, list) and any(isinstance(r, list) and len(r) for r in m)
    if kind == "candlestick":
        return _nonempty_list(chart.get("candles"))
    if kind == "boxplot":
        return _nonempty_list(chart.get("boxes"))
    if kind in {"sankey", "graph"}:
        return _nonempty_list(chart.get("links"))
    if kind in {"tree", "mindmap", "org", "orgchart"}:
        # Accept the same variants the renderer reads: root | tree | data | top-level node | items.
        root = chart.get("root") or chart.get("tree") or chart.get("data")
        if isinstance(root, list):
            root = root[0] if root else None
        if isinstance(root, dict) and (root.get("name") or root.get("label") or root.get("children")):
            return True
        if chart.get("name") or _nonempty_list(chart.get("children")) or _nonempty_list(chart.get("items")):
            return True
        return False
    if kind in {"pie", "donut", "funnel", "treemap", "sunburst", "waterfall", "gauge", "rose"}:
        if _nonempty_list(chart.get("items")):
            return True
        s = chart.get("series")
        return isinstance(s, list) and any(isinstance(x, dict) and x.get("values") for x in s)
    # line | bar | hbar | area | scatter | bubble | stacked | combo | histogram | radar
    # | timeseries | polar | parallel | themeriver | scatter3d | bar3d | line3d
    s = chart.get("series")
    if isinstance(s, list) and any(
        isinstance(x, dict) and isinstance(x.get("values"), list) and len(x.get("values")) for x in s
    ):
        return True
    # Fallback: the model may have used items (label/value) for a bar/line — still renderable.
    return _nonempty_list(chart.get("items"))


_NUMERIC_ARRAY_RE = re.compile(r"^\s*\[\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)*\s*\]\s*$")


def _block_is_renderable(b: dict[str, Any]) -> bool:
    t = str(b.get("type") or "").lower()
    if t == "text":
        content = str(b.get("content") or "").strip()
        # Drop bare numeric arrays (e.g. tic-tac-toe win lines "[0,1,2]") — that's not prose.
        return bool(content) and not _NUMERIC_ARRAY_RE.match(content)
    if t == "kpi_row":
        items = b.get("items")
        return isinstance(items, list) and any(
            isinstance(it, dict) and str(it.get("value", "")).strip() for it in items
        )
    if t == "chart":
        chart = b.get("chart")
        return isinstance(chart, dict) and _chart_has_data(chart)
    if t == "table":
        rows = b.get("rows")
        return isinstance(rows, list) and len(rows) > 0
    if t == "action_row":
        btns = b.get("buttons")
        return isinstance(btns, list) and len(btns) > 0
    if t == "image":
        return bool(str(b.get("src") or "").strip())
    if t == "stat_card":
        items = b.get("items")
        return isinstance(items, list) and any(
            isinstance(it, dict) and str(it.get("value", "")).strip() for it in items
        )
    if t == "progress":
        items = b.get("items")
        return isinstance(items, list) and any(
            isinstance(it, dict) and it.get("value") is not None for it in items
        )
    if t == "badge_row":
        items = b.get("items")
        return isinstance(items, list) and any(
            isinstance(it, dict) and str(it.get("label", "")).strip() for it in items
        )
    return False


def _validate_layout_against_registry(layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop unknown block types and blocks with no renderable data."""
    valid_types = _registry_block_types()
    out: list[dict[str, Any]] = []
    for b in layout:
        if not isinstance(b, dict):
            continue
        if str(b.get("type") or "").lower() not in valid_types:
            continue
        if not _block_is_renderable(b):
            continue
        out.append(b)
    return out


def coerce_widget_schema_root(obj: Any) -> dict[str, Any] | None:
    """Ensure root has layout: [] (alias blocks/components; wrap bare arrays)."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return {"version": "1.0", "layout": obj}
    if not isinstance(obj, dict):
        return None
    out = dict(obj)
    # Unwrap { "widget": { "layout": [...] } } or { "schema": {...} }
    if isinstance(out.get("widget"), dict):
        inner = dict(out.pop("widget"))
        out = {**out, **inner}
    if isinstance(out.get("schema"), dict):
        inner = dict(out.pop("schema"))
        out = {**out, **inner}
    if "layout" not in out or not isinstance(out.get("layout"), list):
        if isinstance(out.get("blocks"), list):
            out["layout"] = out["blocks"]
        elif isinstance(out.get("components"), list):
            out["layout"] = out["components"]
        elif isinstance(out.get("Layout"), list):
            out["layout"] = out.pop("Layout")
    # layout is a single block object (common model mistake)
    lay = out.get("layout")
    if isinstance(lay, dict) and lay.get("type"):
        out["layout"] = [lay]
    # Root is one block with no layout key
    if not isinstance(out.get("layout"), list) and out.get("type") in _BLOCK_TYPES:
        ver = out.pop("version", None)
        block = dict(out)
        out = {"version": str(ver or "1.0"), "layout": [block]}
    if isinstance(out.get("layout"), list):
        out["layout"] = _validate_layout_against_registry(_sanitize_layout(out["layout"]))
    return out


def widget_schema_json_is_valid(schema_str: str) -> bool:
    if not (schema_str or "").strip():
        return False
    try:
        o = json.loads(schema_str)
    except json.JSONDecodeError:
        return False
    return isinstance(o, dict) and isinstance(o.get("layout"), list)


def finalize_widget_schema_json(raw: str) -> str:
    """
    Normalize model output for the Vue renderer: strip fences, extract JSON, coerce layout.
    Returns a string safe for JSON.parse on the client (or best-effort stripped text).
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    obj = parse_widget_schema_object(raw)
    if obj is None:
        return strip_widget_markdown_fences(raw)
    coerced = coerce_widget_schema_root(obj)
    if coerced is None:
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps(coerced, ensure_ascii=False)


_JSON_WIDGET_RULE = """
WIDGET JSON SCHEMA MODE (WIDGET_MODE=json):
- The content inside <WIDGET> MUST be valid JSON (no markdown fences, no comments).
- Root object: { "version": "1.0", "layout": [ ... ] }
- layout is an ordered array of blocks (top-to-bottom).
- Supported block types ONLY (do not invent new ones):
  - text:
    { "type": "text", "id": "...", "content": "..." }
  - kpi_row:
    { "type": "kpi_row", "id": "...", "items": [ { "label": "...", "value": "...", "tone": "positive|neutral|negative" }, ... ] }
  - chart (pick "kind" to fit the data; ONLY these kinds render):
    - line | bar | area | scatter  -> use "series" with [x, y] number pairs:
      { "type": "chart", "id": "...", "title": "...", "chart": { "kind": "line|bar|area|scatter", "x_label": "...", "y_label": "...", "series": [ { "name": "...", "color": "blue|orange|green|red|purple", "values": [ [x, y], ... ] }, ... ] } }
    - heatmap (correlation matrix, confusion matrix, any grid of values) -> use labels + a 2D "matrix" (rows = y_labels, cols = x_labels):
      { "type": "chart", "id": "...", "title": "...", "chart": { "kind": "heatmap", "x_labels": ["A","B",...], "y_labels": ["A","B",...], "matrix": [ [1, 0.3, ...], [0.3, 1, ...], ... ] } }
    - pie | donut (composition / share of a whole) -> use "items":
      { "type": "chart", "id": "...", "title": "...", "chart": { "kind": "pie|donut", "items": [ { "label": "...", "value": 42 }, ... ] } }
    Do NOT invent other kinds. For a correlation matrix you MUST use kind "heatmap" with "matrix" (never a line/bar series).
  - table:
    { "type": "table", "id": "...", "title": "...", "columns": [ ... ], "rows": [ [ ... ], ... ] }
  - action_row:
    { "type": "action_row", "id": "...", "buttons": [ { "id": "...", "label": "...", "intent": "..." }, ... ] }
  - image (photos, diagrams, icons, illustrations — any raster or SVG via URL/data URI):
    { "type": "image", "id": "...", "title": "...", "src": "https://... OR data:image/png;base64,...", "alt": "accessible description", "caption": "optional caption", "fit": "contain|cover" }

Data grounding (STRICT — the widget must mirror your <RESPONSE>):
- Use ONLY the exact numbers and labels stated in your <RESPONSE>. Do NOT invent values, round differently, or add labels/values not written in <RESPONSE>.
- For category charts, "x_categories" MUST be the exact entity/segment names you named in <RESPONSE> (e.g. "Data Center", "Gaming") — NEVER 0, 1, 2. Each series value MUST equal the number in <RESPONSE>, aligned to those categories.
- KPI/stat/progress values and labels must also be the exact ones from <RESPONSE>.
- If <RESPONSE> does not state a number or its label, do not put it in the widget (state it in <RESPONSE> first, or omit it).

Honoring an explicitly requested chart type:
- If the user asks for a SPECIFIC chart type (e.g. "as a candlestick", "show a sankey", "pie chart") and it IS one of the supported kinds above, you MUST use exactly that kind — do not substitute another.
- If the requested chart type is NOT in the supported kinds (e.g. 3D surface, map/choropleth, gantt, renko, marimekko, etc.), DO NOT silently render a different chart. Instead: in <RESPONSE>, say plainly that you cannot render that specific chart type yet, then name 1-3 supported kinds that would fit their data well and ask if they'd like one of those. In that case return <WIDGET></WIDGET> (empty) — wait for the user to confirm before rendering an alternative.
- Never pretend an unsupported type is supported, and never relabel a different chart as the requested type.

Interactivity:
- action_row buttons are VISUALS-ONLY: each button must only offer to redraw the data ALREADY shown as a different supported chart kind (e.g. "Show as bar chart", "View as treemap", "Show as pie", "View as horizontal bars"). The button label is sent back as the next prompt, so it must be something you can definitely do with the data on screen.
- NEVER add action buttons that need data you may not have: no new tickers/companies, no new time periods, no "compare X vs Y", no growth/peers/forecasts, no "explain ...". If no alternative chart kind fits the data, omit the action_row.
- OUTPUT JSON ONLY. Never output HTML, <script>, <style>, <canvas>, raw markup, or code inside <WIDGET> — only the JSON schema above. There is no HTML mode.
- If something cannot be expressed with the supported block types (e.g. a playable game, live sliders), DO NOT invent HTML — return <WIDGET></WIDGET> (empty) and explain in <RESPONSE> instead. Never dump raw index arrays like `[0,1,2]`.
- For photographs, diagrams, or icons, use the `image` block with a valid https:// URL or a data: URI. Combine `image` with `text`, `chart`, and `table` blocks as needed.

Dynamic layout (JSON) — avoid static, single-block dashboards:
- Shape `layout` like a short story: context first, then metrics, then detail, then actions. Mix block types (text, kpi_row, chart, table, image, action_row) whenever it improves scanning; do not default to one lonely chart if KPIs or a sentence of framing would help.
- Use `action_row` for obvious follow-up intents; keep blocks ordered top-to-bottom by importance so the widget feels purposeful, not generic.
"""


# ── Registry-driven prompt vocabulary ───────────────────────────────────────
# Single source of truth: frontend-vue/src/widget-registry.json. The SAME file
# the Vue renderer uses to resolve block types is read here to generate the
# "supported block types" section of the prompt — so GENERATE and RENDER can
# never drift. Falls back to the static _JSON_WIDGET_RULE if the file is absent.
_REGISTRY_JSON_PATH = Path(__file__).resolve().parent.parent / "frontend-vue" / "src" / "widget-registry.json"

_JSON_RULE_PREAMBLE = """WIDGET JSON SCHEMA MODE (WIDGET_MODE=json):
- WIDGET WARRANT (decide for yourself — there is no keyword trigger):
  Include a NON-EMPTY widget ONLY IF both are true: (1) a visual materially helps (a comparison of
  several numbers, a trend, a breakdown/share, a flow, or a matrix), AND (2) you can populate it with
  real values from your <RESPONSE> using the supported block types below. If you cannot build it from
  the allowed blocks (or there is no concrete data), return <WIDGET></WIDGET> (empty) and say so in
  <RESPONSE>. Do NOT chart a single number, a definition, a yes/no, or an opinion.
- The content inside <WIDGET> MUST be valid JSON (no markdown fences, no comments).
- Root object: { "version": "1.0", "layout": [ ... ] }
- layout is an ordered array of blocks (top-to-bottom).
- Supported block types ONLY (do not invent new ones):"""

_JSON_RULE_TRAILER = """
NO FABRICATED DATA (most important — applies to <RESPONSE> and <WIDGET>):
- If you do NOT actually have the real figures to answer (e.g. live or historical stock prices, a fund's daily returns, exact financials you are not sure of), DO NOT invent, estimate, or use "illustrative"/"mock"/"example"/"est." data, and DO NOT draw a widget.
- In that case: say plainly in <RESPONSE> that you do not have that specific data, suggest what the user could provide or ask instead, and return <WIDGET></WIDGET> (empty).
- NEVER label a chart or value "mock", "illustrative", "estimated", or "example" — if it is not real, do not render it. A widget must only ever show real, known values.
- Educational math demos (e.g. compound-interest with user-given P/r/t) are fine because the user supplied the inputs; market data you don't have is NOT.

Data grounding (STRICT — the widget must mirror your <RESPONSE>):
- Use ONLY the exact numbers and labels stated in your <RESPONSE>. Do NOT invent values, round differently, or add labels/values not written in <RESPONSE>.
- For category charts, "x_categories" MUST be the exact entity/segment names you named in <RESPONSE> (e.g. "Data Center", "Gaming") — NEVER 0, 1, 2. Each series value MUST equal the number in <RESPONSE>, aligned to those categories.
- KPI/stat/progress values and labels must also be the exact ones from <RESPONSE>.
- If <RESPONSE> does not state a number or its label, do not put it in the widget (state it in <RESPONSE> first, or omit it).

Honoring an explicitly requested chart type:
- If the user asks for a SPECIFIC chart type (e.g. "as a candlestick", "show a sankey", "pie chart") and it IS one of the supported kinds above, you MUST use exactly that kind — do not substitute another.
- If the requested chart type is NOT in the supported kinds (e.g. 3D surface, map/choropleth, gantt, renko, marimekko, etc.), DO NOT silently render a different chart. Instead: in <RESPONSE>, say plainly that you cannot render that specific chart type yet, then name 1-3 supported kinds that would fit their data well and ask if they'd like one of those. In that case return <WIDGET></WIDGET> (empty) — wait for the user to confirm before rendering an alternative.
- Never pretend an unsupported type is supported, and never relabel a different chart as the requested type.

Interactivity:
- action_row buttons are VISUALS-ONLY: each button must only offer to redraw the data ALREADY shown as a different supported chart kind (e.g. "Show as bar chart", "View as treemap", "Show as pie", "View as horizontal bars"). The button label is sent back as the next prompt, so it must be something you can definitely do with the data on screen.
- NEVER add action buttons that need data you may not have: no new tickers/companies, no new time periods, no "compare X vs Y", no growth/peers/forecasts, no "explain ...". If no alternative chart kind fits the data, omit the action_row.
- OUTPUT JSON ONLY. Never output HTML, <script>, <style>, <canvas>, raw markup, or code inside <WIDGET> — only the JSON schema above. There is no HTML mode.
- If something cannot be expressed with the supported block types (e.g. a playable game, live sliders), DO NOT invent HTML — return <WIDGET></WIDGET> (empty) and explain in <RESPONSE> instead. Never dump raw index arrays like `[0,1,2]`.
- For photographs, diagrams, or icons, use the `image` block with a valid https:// URL or a data: URI. Combine `image` with `text`, `chart`, and `table` blocks as needed.

Dynamic layout (JSON) — avoid static, single-block dashboards:
- Shape `layout` like a short story: context first, then metrics, then detail, then actions. Mix block types whenever it improves scanning; do not default to one lonely chart if KPIs or a sentence of framing would help.
- Use `action_row` for obvious follow-up intents; keep blocks ordered top-to-bottom by importance so the widget feels purposeful, not generic."""


def build_json_widget_rule() -> str:
    """Generate the WIDGET_MODE=json rule from the shared widget-registry.json."""
    try:
        data = json.loads(_REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
        blocks = data.get("blocks") if isinstance(data, dict) else None
        if not isinstance(blocks, list) or not blocks:
            return _JSON_WIDGET_RULE.strip()
        entries: list[str] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = str(b.get("type") or "").strip()
            when = str(b.get("whenToUse") or "").strip()
            spec = b.get("spec")
            if not t or not spec:
                continue
            spec_lines = spec if isinstance(spec, list) else [str(spec)]
            body = "\n".join("    " + str(ln) for ln in spec_lines)
            header = f"  - {t}" + (f" — {when}" if when else "") + ":"
            entries.append(f"{header}\n{body}")
        if not entries:
            return _JSON_WIDGET_RULE.strip()
        return _JSON_RULE_PREAMBLE + "\n" + "\n".join(entries) + "\n" + _JSON_RULE_TRAILER
    except Exception:
        return _JSON_WIDGET_RULE.strip()



_OUTPUT_CONTRACT_STRICT = """
OUTPUT CONTRACT (STRICT — MUST FOLLOW)
You MUST return these sections in this exact order:

<RESPONSE>
...text...
</RESPONSE>
<WIDGET>
...widget OR empty...
</WIDGET>

Rules:
- Never omit the <WIDGET> tags, but content may be empty when a widget is not warranted.
- If widget is warranted, return a valid interactive widget (not placeholders).
- If widget is not warranted (simple chit-chat / conceptual text-only), return exactly <WIDGET></WIDGET>.
If you fail to follow this contract, the system will break.
"""

# Strict bandit primitive: extra lines for <RESPONSE> when STRICT_PRIMITIVES is on (prompt-only).
_STRICT_PRIMITIVE_EXTRAS: dict[str, str] = {
    "structured_bullets": (
        "Use only the format in the Rule: 3–5 lines, each starting with '- '. "
        "Do not use numbered lists or paragraph prose as the main answer."
    ),
    "narrative_prose": (
        "Use only short paragraphs (no '- ' bullets, no numbered list as the main answer)."
    ),
    "concise_direct": (
        "Obey the sentence limit in the Rule. No bullet or numbered lists unless the Rule allows."
    ),
    "socratic_questions": (
        "Match the Rule: brief acknowledgement, then 1–2 questions only — not a full tutorial."
    ),
    "step_by_step": (
        "Use only a numbered list (3–6 steps). Do not use '-' bullets for the main steps."
    ),
    "comparison_table": (
        "Output ONLY a GitHub-flavored markdown table: one header row, a |---| separator row, then data rows. "
        "No section titles, bullets, or paragraphs outside the table. Do not put JSON in <RESPONSE>."
    ),
    "visualization": (
        "Write a brief 1-2 sentence lead-in only. The <WIDGET> carries the actual visual. "
        "NEVER draw ASCII charts, tree diagrams (├──), or text-art, and never put code blocks or JSON in <RESPONSE>."
    ),
}


def build_strict_response_rule_line(strategy_id: str) -> str:
    """Instructions so <RESPONSE> matches the selected bandit strategy (strict primitive)."""
    base = (
        "MANDATORY for <RESPONSE> — follow the Strategy and Rule below exactly. "
        "Do not substitute a different format; the UI label must match what you write."
    )
    extra = _STRICT_PRIMITIVE_EXTRAS.get(strategy_id, "").strip()
    if extra:
        return f"{base} {extra}"
    return base


def is_social_or_greeting_turn(user_message: str) -> bool:
    """Short greeting/thanks/goodbye only — no bandit strategy format (used for prompt routing only)."""
    t = (user_message or "").strip()
    if not t or len(t) > 160:
        return False
    tl = " ".join(t.lower().split())
    if "?" in t and len(t) > 25:
        return False
    if re.fullmatch(
        r"(hi|hello|hey|yo|sup|hiya|bye|goodbye|ok|okay|k|cheers|thx|ty|thanks|thank you)"
        r"([!?.])*",
        tl,
    ):
        return True
    if re.fullmatch(
        r"(thanks|thank you)( a lot| so much| again)?([!?.])*",
        tl,
    ):
        return True
    if re.fullmatch(
        r"(good )?(morning|afternoon|evening|night)([!?.])*",
        tl,
    ):
        return True
    if re.fullmatch(r"got it([!?.])*", tl):
        return True
    if re.fullmatch(
        r"(hi|hello|hey)\s+(there|everyone|all|team)([!?.])*",
        tl,
    ):
        return True
    return False


def build_combined_system_prompt(
    strategy_id: str,
    format_rule: str,
    primitive_extra_context: str,
    user_message: str,
    forbidden_components: list[str] | None = None,
    required_components: list[str] | None = None,
) -> str:
    """
    Build the combined system prompt for a single LLM call that outputs
    both the response text and the widget HTML together.

    Args:
        strategy_id:              selected strategy name (e.g. 'comparison_table')
        format_rule:              primitive format instruction for response text
        primitive_extra_context:  widget layout instructions from primitives.json
        forbidden_components:     component names the widget MUST NOT use
        required_components:      component names the widget MUST use
    """
    widget_block = ""
    if primitive_extra_context:
        widget_block = f"""
## Widget layout instructions (follow these exactly)
{primitive_extra_context}
"""

    constraint_block = ""
    if forbidden_components:
        names = ", ".join(forbidden_components)
        constraint_block += (
            "\n## FORBIDDEN — do NOT use these components inside <WIDGET>\n"
            f"{names}\n"
            "If your HTML contains any of these, the widget will be rejected and replaced.\n"
        )
    if required_components:
        names = ", ".join(required_components)
        constraint_block += (
            "\n## REQUIRED — your <WIDGET> MUST contain these components\n"
            f"{names}\n"
            "If your HTML is missing any of these, the widget will be rejected and replaced.\n"
        )

    social_only = is_social_or_greeting_turn(user_message)
    if social_only:
        response_rule_line = (
            "GENERAL / SOCIAL TURN — ignore Strategy and Rule below. "
            "Reply in 1–2 short, natural sentences only. No bullets, tables, numbered lists, or fenced code blocks."
        )
    elif getattr(config, "STRICT_PRIMITIVES", False):
        response_rule_line = build_strict_response_rule_line(strategy_id)
    else:
        response_rule_line = "Treat this as a style hint for <RESPONSE> (do not be rigid)."

    social_turn_banner = ""
    if social_only:
        social_turn_banner = """
═══════════════════════════════════════════════════════
GENERAL QUESTION — GREETING / THANKS / GOODBYE (no bandit strategy)
═══════════════════════════════════════════════════════
The user's message is only a greeting, thanks, acknowledgement, or goodbye.
- <RESPONSE>: Brief, friendly, natural text. Do NOT apply the bandit Strategy or Rule shown below.
- <WIDGET>: Return exactly <WIDGET></WIDGET> (empty). No chart, no dashboard, no placeholder HTML.
═══════════════════════════════════════════════════════
"""

    # Components-only / JSON schema mode is the only mode. The widget vocabulary,
    # warrant rubric, strict grounding, and decline rules all come from the registry
    # (build_json_widget_rule → widget-registry.json). No HTML, no external libraries.
    widget_format_line = "JSON UI schema ONLY (no HTML) for the widget"
    widget_rules_header = "WIDGET RULES — for the JSON schema inside <WIDGET>"
    widget_rules_body = build_json_widget_rule()

    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 7500)
    token_limit_block = f"""
TOKEN LIMIT — you have ~{combined_max_tokens} tokens total for <RESPONSE> + <WIDGET>.
- Prioritize completing the widget. Never stop mid-widget or truncate. A complete, functional widget is required.
- When space is tight: shorten <RESPONSE> (2–5 sentences), not the <WIDGET>. The widget must always be full and working.
- For comparison tables in <RESPONSE>: keep focused so the widget has room. Both must fit.
"""

    return f"""You are an expert AI assistant. Each turn you produce a written answer AND an OPTIONAL interactive widget built ONLY from a fixed set of UI components defined below — there is NO HTML and NO external charting libraries.
{social_turn_banner}
Output style: No emojis. Neat, clean, professional — in both <RESPONSE> text and <WIDGET>.
{token_limit_block}
{_OUTPUT_CONTRACT_STRICT}

For every turn you produce TWO sections in one generation: the <RESPONSE> text first, then the <WIDGET>. The widget may be empty when a visual is not warranted (see WIDGET WARRANT below). Never describe a widget you did not produce (e.g. don't write "the dashboard below" and then return an empty widget).

RENDER NOW — never defer or ask permission for a visual you can make:
- If the user asks for a chart/visualization and you can build it from this conversation or your own knowledge, you MUST output the NON-EMPTY <WIDGET> in THIS turn.
- NEVER reply with "I will plot…", "let me show…", "shall I…", or ask "which dataset?" when the conversation already implies the data — just render the chart now with a sensible default.
- For a clear visualization request, do NOT respond with clarifying questions (even if the Strategy suggests asking) — produce the chart. Only ask a question if the request is genuinely ambiguous AND no reasonable default exists.
- The Strategy only shapes the short framing TEXT; it must NEVER stop you from producing the widget. Only return an empty widget when you truly lack the data or the requested chart type is unsupported (and then say so plainly).

═══════════════════════════════════════════════════════
OUTPUT FORMAT — always use exactly this structure
═══════════════════════════════════════════════════════
<RESPONSE>
[Your answer here — follow the FORMAT RULE below]
</RESPONSE>
<WIDGET>
[{widget_format_line}]
</WIDGET>

═══════════════════════════════════════════════════════
RESPONSE FORMAT RULE — {response_rule_line}
═══════════════════════════════════════════════════════
Strategy: {strategy_id}
Rule: {format_rule}
Do not mention this rule. Do not add <WIDGET> inside <RESPONSE>.
<RESPONSE> is prose for the user — NEVER put JSON, a widget schema, block objects, or code fences in it. All structured data goes ONLY inside <WIDGET>.
NEVER draw visuals as text in <RESPONSE>: no ASCII charts/bars, no tree drawings (├──, └──), no aligned-column tables-as-art. The <WIDGET> is the ONLY place a visualization lives. For a hierarchy/mind map, put it in a chart block with "kind":"tree"|"mindmap"|"org" and a "root", not as text.
CRITICAL — Primitives vs Widget (never confuse these):
- The Strategy/Rule above applies ONLY to <RESPONSE> (text format: bullets, table, prose, etc.). It does NOT constrain <WIDGET>.
- <WIDGET> is SEPARATE and INDEPENDENT. Widget choice depends on the content of your <RESPONSE>, not on the text format.

═══════════════════════════════════════════════════════
{widget_rules_header}
═══════════════════════════════════════════════════════
{widget_rules_body}
{widget_block}
{constraint_block}"""


def build_combined_user_prompt(
    user_message: str,
    history: list[dict],
    max_history: int = 4,
) -> str:
    """Build the user-turn prompt including conversation history."""
    ctx: list[str] = []
    for turn in history[-max_history:]:
        u = turn.get("user", "")
        a = turn.get("assistant", "")
        if u:
            ctx.append(f"User: {u}")
        if a:
            # Strip any <WIDGET>...</WIDGET> from stored history to keep it concise.
            a_clean = re.sub(r"<WIDGET>.*?</WIDGET>", "", a, flags=re.DOTALL).strip()
            ctx.append(f"Assistant: {a_clean}")
    ctx.append(f"User: {user_message}")
    return "\n".join(ctx)


def parse_combined_output(raw: str) -> Tuple[str, str]:
    """
    Parse model combined output into (response_text, widget_payload).

    Returns:
        (response_text, widget_payload)
        Either can be empty string if the tag is missing or parsing fails.
    """
    response_text = ""
    widget_payload = ""

    # Extract <RESPONSE>...</RESPONSE>
    resp_match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", raw, re.DOTALL | re.IGNORECASE)
    if resp_match:
        response_text = resp_match.group(1).strip()
    else:
        # Fallback: everything before <WIDGET> is the response
        widget_start = raw.find("<WIDGET>")
        if widget_start == -1:
            widget_start_upper = raw.upper().find("<WIDGET>")
            if widget_start_upper != -1:
                widget_start = widget_start_upper
        if widget_start > 0:
            response_text = raw[:widget_start].strip()
        else:
            response_text = raw.strip()

    # Extract <WIDGET>...</WIDGET> — JSON schema only (no HTML mode).
    widget_match = re.search(r"<WIDGET>(.*?)</WIDGET>", raw, re.DOTALL | re.IGNORECASE)
    if widget_match:
        raw_widget = widget_match.group(1).strip()
        widget_payload = finalize_widget_schema_json(raw_widget)

    return response_text, widget_payload
