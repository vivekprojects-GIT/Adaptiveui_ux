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
from typing import Any, Tuple

from . import config
from .widget_prompt import inject_design_system


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


_BLOCK_TYPES = frozenset({"text", "kpi_row", "chart", "table", "action_row", "image"})

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
        out["layout"] = _sanitize_layout(out["layout"])
    return out


def widget_schema_json_is_valid(schema_str: str) -> bool:
    if not (schema_str or "").strip():
        return False
    try:
        o = json.loads(schema_str)
    except json.JSONDecodeError:
        return False
    return isinstance(o, dict) and isinstance(o.get("layout"), list)


def extract_embeddable_html_document(raw: str) -> str | None:
    """
    If the model put HTML/JS widgets inside <WIDGET> while WIDGET_MODE=json, return a full HTML
    document suitable for inject_design_system + iframe. Otherwise None.
    """
    s = strip_widget_markdown_fences((raw or "").strip())
    if not s or "<" not in s or ">" not in s:
        return None
    st = s.lstrip()
    if st.startswith("{") and "<div" not in s.lower() and "<html" not in s.lower():
        return None
    low = s.lower()
    if "<html" in low or "<!doctype" in low:
        return re.sub(r"<!DOCTYPE[^>]*>", "", s, flags=re.IGNORECASE).strip()
    if any(
        tag in low
        for tag in (
            "<script",
            "<body",
            "<div",
            "<canvas",
            "<iframe",
            "<form",
            "<input",
            "<button",
            "<style",
        )
    ):
        inner = re.sub(r"<!DOCTYPE[^>]*>", "", s, flags=re.IGNORECASE).strip()
        if "<html" not in inner.lower():
            return f"<html><head></head><body>{inner}</body></html>"
        return inner
    return None


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

Data grounding:
- Prefer data from user message or <RESPONSE>. When real data is unavailable, use illustrative/mock data and label it clearly (e.g. "Example data", "Mock data").

Interactivity:
- Use action_row buttons to request follow-ups via intent strings (e.g., "explain_methodology", "show_risks").
- GAMES, TOYS, and CUSTOM APPS (tic-tac-toe, puzzles, interactive demos, any playable UI): you MUST output a **complete HTML document** (`<html>...</html>` with CSS/JS) inside `<WIDGET>`, NOT a JSON schema. JSON blocks cannot represent a real game board — never dump raw index arrays like `[0,1,2]` as layout items.
- If the user needs true controls (sliders, inputs, live calculator), rich HTML/SVG/canvas, or complex layouts that JSON blocks cannot express, you MAY put a complete mini HTML document (with inline JS) inside <WIDGET> instead of JSON — the app will still render it. Prefer JSON when charts/KPIs/tables/images suffice.
- For photographs, diagrams, or icons in JSON mode, use the `image` block with a valid https:// URL or a data: URI. Combine `image` with `text`, `chart`, and `table` blocks as needed.

Dynamic layout (JSON) — avoid static, single-block dashboards:
- Shape `layout` like a short story: context first, then metrics, then detail, then actions. Mix block types (text, kpi_row, chart, table, image, action_row) whenever it improves scanning; do not default to one lonely chart if KPIs or a sentence of framing would help.
- Use `action_row` for obvious follow-up intents; keep blocks ordered top-to-bottom by importance so the widget feels purposeful, not generic.
"""


# ── Design system injected into combined output ────────────────────────────

_DESIGN_SYSTEM_REMINDER = """
A CSS design system is pre-injected into every widget iframe. Use ONLY these variables:
--bg, --bg2, --bg3 (backgrounds)  --text, --text2, --text3 (text)
--border, --border2 (borders)     --accent, --accent-bg, --accent-b (blue)
--success, --success-bg (green)   --warn, --warn-bg (amber)
--danger, --danger-bg (red)       --radius, --radius-sm, --radius-pill

Pre-built CSS classes (use them directly, no need to redefine):
.card .raised .card-title .tabs .tab .panel .search .pills .pill
.ctrl-row .ctrl-lbl .ctrl-val .btn-group .btn .ask-btn
.badge .b-blue .b-green .b-amber .b-red .b-gray
.metric-grid .metric .metric-lbl .metric-val
.progress-wrap .progress-bar .result-box .result-lbl .result-val .result-sub
.step-row .step-num .step-title .step-desc .count-lbl .empty
"""

_SENDPROMPT_RULE = """
Always define and use this exact bridge function inside <WIDGET>:
  function sendPrompt(t){window.parent.postMessage({type:"streamlit:setComponentValue",value:t},"*");}
Every clickable card, row, chip, and button must call sendPrompt with a specific, contextual message.
"""

_REACTIVE_RUNTIME_RULE = """
Universal reactive mini-app contract (follow for every widget):
- Your widget MUST follow this exact execution model:
  1) Define:
     - const data = ...      // embedded data derived ONLY from user/context and your <RESPONSE>
     - const state = {...}   // ALL user inputs (sliders/filters/selections). Initial values must match exact numbers you used in <RESPONSE>.
  2) Implement:
     - function compute(state, data) { return {...} }  // pure transforms: filter/aggregate/calc/sort. No network.
     - function render() { const c = compute(state, data); ... update DOM + chart + table from c ... }
  3) On load: always call render() once so the widget is never empty.
  4) On interaction: update state -> call render() immediately (instant UX; never call the LLM on slider drag).
  5) sendPrompt: ONLY when new knowledge/data is required. Include current state in the prompt.

Charts:
- Use any public chart/library CDN from cdnjs.cloudflare.com, cdn.jsdelivr.net, unpkg.com, or cdn.plot.ly.
- Chart backgrounds must be transparent for iframe embedding.
- ECharts (preferred): https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js
- Plotly, D3, Chart.js, ApexCharts, and other public viz libraries are allowed.

Forbidden (never include):
- fetch / XMLHttpRequest / WebSocket
- eval / new Function
"""

_DYNAMIC_WIDGET_UX_RULE = """
Dynamic, responsive widgets (not static posters):
- Layout: use flex/grid with wrap, minmax(), and clamp() so content reflows when the iframe is narrow or wide. Prefer fluid widths (%, fr, max-width) over fixed pixel widths for main columns.
- Motion & feedback: add CSS transitions on hover/focus for cards, buttons, and controls; subtle transform (translateY) on hover where it aids affordance. Enable chart library animation (e.g. ECharts animation / animationDuration) so series draw in smoothly.
- Interactivity: expose meaningful controls — tabs, toggles, filters, sliders, dataZoom/brush on charts when data density warrants it. When state changes, re-render charts/tables immediately (same reactive pattern as sliders).
- Depth: combine visuals (chart + KPI strip + short table + optional image/diagram) when the answer benefits; vary structure by use case instead of repeating one template every turn.
"""

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

_LIBRARIES_RULE = """
Allowed libraries for <WIDGET> — use any public CDN from cdnjs.cloudflare.com, cdn.jsdelivr.net, unpkg.com, or cdn.plot.ly.
Use whichever library produces the best visual for the use case. You may combine libraries (e.g. ECharts + Tabulator).

Recommended (pick the best fit):
- ECharts: https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js
- Plotly.js: https://cdn.plot.ly/plotly-2.30.0.min.js
- D3.js: https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js
- Chart.js: https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js
- ApexCharts: https://cdn.jsdelivr.net/npm/apexcharts
- Tabulator (tables): https://cdn.jsdelivr.net/npm/tabulator-tables/dist/js/tabulator.min.js + CSS
- Mermaid (flowcharts, sequence diagrams): https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js
- Three.js (simple 3D): https://cdn.jsdelivr.net/npm/three/build/three.min.js

You may use any other public library from these CDNs. Choose the library that creates the best, most accurate visualization.

Color + theming baseline (applies to every engine):
- Always define a JS palette (array of hex colors) and apply it explicitly to series/marks.
- Detect dark mode with: const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
- Explicitly set: axis label color, grid line color, legend text color, and tooltip styling.

Library choice guidance (use the best fit; do not force the same layout every time):
- Time-series trends (date/time x-axis, >=5 points): ECharts line/area + tooltip + subtle dataZoom.
- Categorical rankings (categories with numeric values): ECharts horizontal bar + click-to-filter + cross-filter table.
- Composition/share: stacked bars (or 100% stacked) with tooltip value + %; pie/donut only when 3–5 short categories.
- Distributions:
  - if you have raw samples: histogram-like bins
  - if you only have summary stats: do not invent bins; use KPI tiles + short explanation.
- Correlation/relationship (x-y pairs): ECharts scatter; highlight outliers.
- Hierarchies: treemap only when parent/child is explicit; otherwise use grouped table.
- Many series: avoid clutter; use small multiples or series toggles (do not plot >6 lines by default).
- Tables: Tabulator always for scan/sort/filter when it helps (rows > 8 or user asked for a breakdown).
- Prose/conceptual answers with no extractable dataset: still generate a widget using illustrative/mock data and label it clearly (e.g. "Example", "Illustrative", "Mock data").

Engine-specific rendering requirements:
- ECharts: option.backgroundColor must be 'transparent'; set textStyle/axis/grid colors from theme.
- Plotly: set paper_bgcolor/plot_bgcolor to 'rgba(0,0,0,0)'; set layout.font.color and layout.colorway=palette.
- D3: create SVG with responsive sizing; set tooltip styles; apply palette for strokes/fills.
"""

_ANALYTICS_DEFAULTS_RULE = """
Dashboard decision policy (data-driven; do this internally—do not output the reasoning):
1) DATASET EXTRACTION:
   - Prefer data from: (a) user request/context, (b) numeric values in <RESPONSE>.
   - If sufficient data exists, use it. If not, use illustrative/mock data — but you MUST clearly label it (e.g. "Example data", "Mock data", "Illustrative") in the widget title or a visible subtitle.
2) DATA-SHAPE DETECTION:
   - Determine shape: time-series, categorical ranking, composition, distribution, correlation, hierarchy, steps/process, or other.
3) WIDGET WARRANT:
   - Generate a widget wherever there is possibility. If extractable data exists, use it.
   - If no extractable dataset (or too few points): use illustrative/mock data and clearly label it (e.g. "Example data", "Mock data", "Illustrative"). Do NOT return empty <WIDGET></WIDGET> when a chart/calculator/table would help.
4) BI LAYOUT (only when warranted):
   - KPI row (3–6 tiles) → optional Controls row → Primary visualization → optional detail table → Insights (2–4).
5) CROSS-VIEW INTERACTION:
   - Any filter/control must update KPIs + chart + table from the SAME filtered dataset.
6) DRILLDOWN LOOP:
   - Click chart mark / legend / table row → sendPrompt('...') with clicked entity + metric + relevant time window (if present) + current filter summary.
7) INSIGHT RULE:
   - Insights must be computed from the dataset in JS (or computed from extracted values). Do not write obvious generic commentary.
"""

_COLOR_THEMING_RULE = """
Color & theming (best-in-class readability + polish):
- You may choose ANY colors, but they MUST remain readable and “enterprise clean”.
- Detect dark mode with: const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
- Create theme tokens in JS:
  - text = dark ? '#e8eaf4' : '#111318'
  - text2 = dark ? '#8d93aa' : '#5a5f72'
  - grid = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  - border = dark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)'
- Define palette in JS (hex array) and use it explicitly.
- Deterministic category coloring:
  - Build `colorMap` from category keys to palette entries (stable ordering).
  - Reuse the same `colorMap` for KPIs and chart series/marks.
- Selection/interaction states:
  - Hover: subtle opacity/brightness change
  - Selected: stronger accent (thicker stroke/line), not neon
- Grid/labels must always be visible: explicitly set label/text/grid colors for the chart engine.
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
        "Output ONLY one markdown fenced code block (triple backticks). Inside: ASCII bar chart (#) and/or "
        "aligned text columns. Do not put raw JSON in <RESPONSE>. No prose outside that single code block."
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
    widget_required: bool = True,
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

    widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
    widget_format_line = (
        "Complete self-contained HTML document for the interactive widget"
        if widget_mode != "json"
        else "JSON UI schema ONLY (no HTML) for the widget"
    )

    widget_rules_header = (
        "WIDGET RULES — for the HTML inside <WIDGET>"
        if widget_mode != "json"
        else "WIDGET RULES — for the JSON schema inside <WIDGET>"
    )

    widget_requirement_block = (
        "WIDGET REQUIRED for this user turn: return a NON-EMPTY widget."
        if widget_required
        else "WIDGET OPTIONAL for this user turn: return <WIDGET></WIDGET> if the turn is better as text-only."
    )

    widget_rules_body = (
        f"""- Hard output contract (never violate):
  - You MUST output BOTH tags exactly once: <RESPONSE>...</RESPONSE> and <WIDGET>...</WIDGET>.
  - Never omit <WIDGET> tags. For text-only turns, output `<WIDGET></WIDGET>`.
- {widget_requirement_block}
- Choose the UI based on the content in <RESPONSE> (data-driven). Do not follow any fixed template.
- IMPORTANT: In HTML mode, the content inside <WIDGET> MUST be HTML (not JSON). It must contain opening <html> and closing </html>.
- Return a COMPLETE, self-contained HTML document (opening <html> to closing </html>).
- Inline ALL CSS in <style> and ALL JS in <script>. External files: use any public CDN (cdnjs, jsdelivr, unpkg, cdn.plot.ly) for charts, tables, and other libraries.
- Visuals — use freely when they clarify the answer: <img> (https:// or data:image/...), inline <svg>, <figure>/<figcaption>, <picture>, <canvas> for drawings, and background-image in CSS (url() to https or data URIs). For diagrams/flowcharts you may embed SVG markup or use canvas/D3/Mermaid via CDN. Attribute image sources when the license requires it.
- {_LIBRARIES_RULE.strip()}
- {_ANALYTICS_DEFAULTS_RULE.strip()}
- {_COLOR_THEMING_RULE.strip()}
- No frameworks (React/Vue/jQuery). Plain HTML + CSS + JS only.
- body background must be transparent (background:transparent!important).
- No position:fixed anywhere.
- Wrap content in <div class="widget-root">.
- No markdown fences/backticks inside <WIDGET>. Use ONLY raw HTML/CSS/JS.
- Always call your main render/calc function once on page load so output is never empty (e.g., call `init()` or `render()` at the end of <script>).
- Charts/tables must be drawn from the embedded dataset immediately after the first render call.
- {_DYNAMIC_WIDGET_UX_RULE.strip()}
- Slider/input changes → local calc() only (never sendPrompt on drag).
- Slider initial values MUST match the exact numbers in your <RESPONSE>. Never invent defaults.
- Use 0.5px solid borders — never 1px solid.
- UI (HTML/CSS): use CSS variables only (no hardcoded hex/rgb). Charts (ECharts/Plotly/Chart.js): you MAY use hex colors in JS configs for palettes/series.
- NO EMOJIS — never use emojis in widgets or labels. Use text only.
- Neat and clean for any data: light backgrounds (#F5F5F5, #F2F2F2), clear typography, generous spacing. Minimal, professional layout. No decorative icons or clutter.
{_REACTIVE_RUNTIME_RULE}
{_DESIGN_SYSTEM_REMINDER}
{_SENDPROMPT_RULE}"""
        if widget_mode != "json"
        else _JSON_WIDGET_RULE.strip()
    )

    combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 7500)
    token_limit_block = f"""
TOKEN LIMIT — you have ~{combined_max_tokens} tokens total for <RESPONSE> + <WIDGET>.
- Prioritize completing the widget. Never stop mid-widget or truncate. A complete, functional widget is required.
- When space is tight: shorten <RESPONSE> (2–5 sentences), not the <WIDGET>. The widget must always be full and working.
- For comparison tables in <RESPONSE>: keep focused so the widget has room. Both must fit.
"""

    return f"""You are an expert AI assistant with rich, dynamic interactive output — widgets should feel alive, responsive, and tailored to each question (not repetitive templates).
{social_turn_banner}
Output style: No emojis. Neat, clean, professional — in both <RESPONSE> text and <WIDGET>.
{token_limit_block}
{_OUTPUT_CONTRACT_STRICT}

For every response you produce TWO sections in one generation.
The widget block may be empty for text-only turns where interactivity is not helpful.

CRITICAL — Never describe a widget you do not generate. If your <RESPONSE> mentions "the dashboard below", "interactive chart", "explore visually", or anything that implies a visualization exists, you MUST output a complete, non-empty <WIDGET>. Do NOT say "the dashboard below" if you return empty <WIDGET></WIDGET>. Either generate the full widget HTML or do not mention it in the text at all.

Only return an EMPTY widget block (<WIDGET></WIDGET>) when the turn is not “widget-worthy”:
- greetings (hi, hello, hey), acknowledgements (thanks, ok, got it), goodbyes (bye, goodbye), pure chit-chat — on these turns do NOT apply the bandit Strategy/Rule to <RESPONSE>; use a short natural reply
- conceptual Q&A with no dataset/comparison/actionable metrics
- planning/roadmap/implementation-step requests where prose is the primary output

Infer from the user's question and your <RESPONSE> content whether a widget would help. If your <RESPONSE> has structure, numbers, comparisons, or decision support, generate the best-fit widget.

Widget warrant decision checklist:
- Generate NON-EMPTY widget if the user asks for charts, dashboard, analytics, comparison, trends, KPIs, forecasting, ranking, tabular breakdown, numeric exploration, diagrams, illustrations, or images that support the answer.
- Generate NON-EMPTY widget if your response includes measurable values that benefit from visual or interactive interpretation.
- Return EMPTY widget for pure explanation/definition/planning where a chart would be decorative noise.

Understand the user's intent. When the question implies visualization, calculation, comparison, or learning, generate a NON-EMPTY <WIDGET>.

Chart/visual selection — pick the right type for the data and use case:
| Data / use case | Best widget type | Library |
|-----------------|------------------|---------|
| Time-series, trends | Line or area chart | ECharts, Plotly, Chart.js |
| Categorical comparison | Bar chart (horizontal) | ECharts, Chart.js, ApexCharts |
| Part of whole | Pie, donut, stacked bar | ECharts, Chart.js |
| Correlation, x-y | Scatter plot | ECharts, Plotly, D3 |
| Adjustable numbers, formulas | Interactive calculator with sliders | Plain JS + Chart.js/ECharts |
| Rows > 8, breakdown | Tabulator table | Tabulator |
| KPI metrics | KPI tiles + optional chart | CSS + ECharts |
| Process, flow | Diagram, sankey, funnel | D3, ECharts |
| Photo, illustration, icon | <img> / inline SVG / image block (JSON) | https or data URI |

Libraries: ECharts (cdn.jsdelivr.net/npm/echarts), Plotly (cdn.plot.ly), Chart.js (cdnjs), Tabulator (tabulator.info). Pick what fits.

One chart vs multiple: Use one chart/widget when it suffices. Add more only when each adds distinct value. Never duplicate the same data in multiple chart types.

CRITICAL — Complete the widget: Never truncate or stop mid-generation. The <WIDGET> must be a complete, functional HTML document. If space is tight, shorten <RESPONSE> — the widget must always finish.
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
CRITICAL — Primitives vs Widget (never confuse these):
- The Strategy/Rule above applies ONLY to <RESPONSE> (text format: bullets, table, prose, etc.). It does NOT constrain <WIDGET>.
- <WIDGET> is SEPARATE and INDEPENDENT. Generate a widget wherever there is possibility, based on content — never skip a widget because the text format is "table" or "prose". Widget choice (chart, calculator, table) depends on content, not on the primitive.

═══════════════════════════════════════════════════════
{widget_rules_header}
═══════════════════════════════════════════════════════
{widget_rules_body}
{widget_block}
{constraint_block}
═══════════════════════════════════════════════════════
DATA GROUNDING — most critical quality rule
═══════════════════════════════════════════════════════
Every entity, name, number, ticker, percentage shown in the widget MUST come from either:
  - the numeric values you extracted from the user request/context, OR
  - the numeric values present in your <RESPONSE>, OR
  - (for educational/concept demos only) reasonable illustrative values you choose (e.g. P=1000, r=5%, t=10 for compound interest).
- Use meaningful labels (Principal, Rate, Years). When using mock data, clearly label it (e.g. "Example", "Mock data").
- When real data is missing: you MAY use illustrative/mock data; label it clearly in the widget.
- Prefer data from <RESPONSE> or user context; when unavailable, use mock/illustrative data and label it.
- When using mock/illustrative data (dates, tickers, values), label it clearly “e.g. Example data, Mock data”.

═══════════════════════════════════════════════════════
sendPrompt specificity — always specific, never generic
═══════════════════════════════════════════════════════
GOOD: sendPrompt('What are the risks of VTI at 0.03% expense ratio?')
BAD:  sendPrompt('Tell me more')
BAD:  sendPrompt('Click for details')
"""


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

    # Extract <WIDGET>...</WIDGET>
    widget_match = re.search(r"<WIDGET>(.*?)</WIDGET>", raw, re.DOTALL | re.IGNORECASE)
    if widget_match:
        raw_widget = widget_match.group(1).strip()

        widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
        if widget_mode == "json":
            widget_payload = finalize_widget_schema_json(raw_widget)
        else:
            # HTML mode: strip markdown fences if model wrapped widget in ```...```
            if "```" in raw_widget:
                fence = re.search(r"```(?:json|html)?\s*(.*?)```", raw_widget, re.DOTALL | re.IGNORECASE)
                raw_widget = fence.group(1).strip() if fence else re.sub(r"```\w*", "", raw_widget).strip()
            if "<" in raw_widget and ">" in raw_widget:
                if "<html" not in raw_widget.lower():
                    raw_widget = f"<html><head></head><body>{raw_widget}</body></html>"
                raw_widget = re.sub(r"<!DOCTYPE[^>]*>", "", raw_widget, flags=re.IGNORECASE).strip()
                widget_payload = inject_design_system(raw_widget)

    return response_text, widget_payload
