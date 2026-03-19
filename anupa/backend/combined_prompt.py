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

import re
from typing import Tuple

from . import config
from .widget_prompt import inject_design_system

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
  - chart:
    { "type": "chart", "id": "...", "title": "...", "chart": { "kind": "line|bar", "x_label": "...", "y_label": "...", "series": [ { "name": "...", "color": "blue|orange|green|red|purple", "values": [ [x, y], ... ] }, ... ] } }
  - table:
    { "type": "table", "id": "...", "title": "...", "columns": [ ... ], "rows": [ [ ... ], ... ] }
  - action_row:
    { "type": "action_row", "id": "...", "buttons": [ { "id": "...", "label": "...", "intent": "..." }, ... ] }

Data grounding:
- Every entity/name/number used in the schema must come from the user message OR the numbers you include in <RESPONSE>.
- Do not fabricate time series. If you cannot produce real points, use KPIs + a table.

Interactivity:
- Use action_row buttons to request follow-ups via intent strings (e.g., "explain_methodology", "show_risks").
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
- MUST use ECharts as the primary visualization library.
- Allowed CDN (preferred): https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js
- ECharts background must be transparent.

Forbidden (never include):
- fetch / XMLHttpRequest / WebSocket
- eval / new Function
"""

_OUTPUT_CONTRACT_STRICT = """
OUTPUT CONTRACT (STRICT — MUST FOLLOW)
You MUST return EXACTLY two sections, in this exact order:

<RESPONSE>
...text...
</RESPONSE>
<WIDGET>
...widget...
</WIDGET>

Rules:
- NEVER omit <WIDGET>.
- NEVER return only text.
- If you are uncertain or missing data, STILL return a valid widget that:
  - clearly labels any values as approximate, and
  - includes controls + an ECharts chart driven by embedded (approximate) data, and
  - includes one or more sendPrompt() buttons asking the user for the missing data (e.g., date range / source).
If you fail to follow this contract, the system will break.
"""

_LIBRARIES_RULE = """
Allowed public libraries (CDN) for <WIDGET>:
- ECharts (BI-grade charts): https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js
- Plotly.js (high-interactivity charts): https://cdn.plot.ly/plotly-2.30.0.min.js
- D3.js (bespoke/custom visuals): https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js
- Tabulator JS (tables): https://cdn.jsdelivr.net/npm/tabulator-tables@6.2.5/dist/js/tabulator.min.js
- Tabulator CSS: https://cdn.jsdelivr.net/npm/tabulator-tables@6.2.5/dist/css/tabulator.min.css
- Chart.js (only if truly needed): https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js

Rules:
- Do NOT use any other external libraries.
- Choose exactly ONE chart engine for the main visualization: ECharts OR Plotly OR D3.
- You may use Tabulator concurrently for detail tables.
- Never mix multiple chart engines for the same chart area.

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
- Prose/conceptual answers with no extractable dataset: return <WIDGET></WIDGET> or a compact visual card (no chart/table shells).

Engine-specific rendering requirements:
- ECharts: option.backgroundColor must be 'transparent'; set textStyle/axis/grid colors from theme.
- Plotly: set paper_bgcolor/plot_bgcolor to 'rgba(0,0,0,0)'; set layout.font.color and layout.colorway=palette.
- D3: create SVG with responsive sizing; set tooltip styles; apply palette for strokes/fills.
"""

_ANALYTICS_DEFAULTS_RULE = """
Dashboard decision policy (data-driven; do this internally—do not output the reasoning):
1) DATASET EXTRACTION:
   - Extract ONE canonical dataset in JS: `const data = [...]` derived from extractable numeric anchors from this conversation:
     (a) numeric values explicitly present in the user request/context you were given, and
     (b) any numeric values you include inside <RESPONSE>.
   - Every KPI/table/chart label and every numeric value used in the widget must be present somewhere in this conversation context (either in the user request/context or in <RESPONSE>).
2) DATA-SHAPE DETECTION:
   - Determine shape: time-series, categorical ranking, composition, distribution, correlation, hierarchy, steps/process, or other.
3) WIDGET WARRANT:
   - If no extractable dataset (or too few points): return <WIDGET></WIDGET> or a compact single-card visual (no chart/table).
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

    response_rule_line = (
        "Follow this exactly for the text inside <RESPONSE>."
        if getattr(config, "STRICT_PRIMITIVES", False)
        else "Treat this as a style hint for <RESPONSE> (do not be rigid)."
    )

    um = (user_message or "").lower()
    explicit_visual = any(
        k in um
        for k in (
            "plot",
            "chart",
            "graph",
            "visualize",
            "visualisation",
            "visualization",
            "dashboard",
            "scatter",
            "line chart",
            "bar chart",
            "histogram",
            "heatmap",
            "candlestick",
        )
    )
    visual_override_block = ""
    if explicit_visual:
        visual_override_block = """
═══════════════════════════════════════════════════════
EXPLICIT VISUAL OVERRIDE (from user message)
═══════════════════════════════════════════════════════
The user explicitly asked for a visualization. Prioritize a chart/graph in <WIDGET>.
If the Strategy/Rule suggests a table-only response, you may still answer in a chart-friendly structure and should include an interactive chart in <WIDGET>.
Do not refuse a chart just because a "comparison_table" style was selected.
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

    widget_rules_body = (
        f"""- Hard output contract (never violate):
  - You MUST output BOTH tags exactly once: <RESPONSE>...</RESPONSE> and <WIDGET>...</WIDGET>.
  - Never omit <WIDGET> tags. If you choose “no widget”, output literally `<WIDGET></WIDGET>` (empty but present).
- Widget CONTENT is OPTIONAL (based on data), but the <WIDGET> tags are REQUIRED.
- Choose the UI based on the content in <RESPONSE> (data-driven). Do not follow any fixed template.
- IMPORTANT: In HTML mode, the content inside <WIDGET> MUST be HTML (not JSON). It must contain opening <html> and closing </html>.
- Return a COMPLETE, self-contained HTML document (opening <html> to closing </html>).
- Inline ALL CSS in <style> and ALL JS in <script>. The only external files allowed are the CDNs below (and Tabulator CSS via <link>).
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
- Slider/input changes → local calc() only (never sendPrompt on drag).
- Slider initial values MUST match the exact numbers in your <RESPONSE>. Never invent defaults.
- Use 0.5px solid borders — never 1px solid.
- UI (HTML/CSS): use CSS variables only (no hardcoded hex/rgb). Charts (ECharts/Plotly/Chart.js): you MAY use hex colors in JS configs for palettes/series.
{_REACTIVE_RUNTIME_RULE}
{_DESIGN_SYSTEM_REMINDER}
{_SENDPROMPT_RULE}"""
        if widget_mode != "json"
        else _JSON_WIDGET_RULE.strip()
    )

    return f"""You are an expert AI assistant with rich interactive output capabilities.

{_OUTPUT_CONTRACT_STRICT}

For every response you produce TWO sections — response text and an interactive widget — in one generation.
Default behavior: generate a NON-EMPTY <WIDGET> that turns your own <RESPONSE> into something interactive/visual.
Only return an EMPTY widget block (<WIDGET></WIDGET>) when the turn is truly not “widget-worthy”:
- greetings / acknowledgements / chit-chat
- or ultra-short answers with no structure and no numbers (≈ 1–2 sentences, no steps, no table, no formula)

Important: users should NOT have to ask for “a widget” every time. If your <RESPONSE> contains structure or numbers,
you are expected to automatically provide the best-fit widget.

Exception (must generate non-empty widget):
- If the user explicitly requests any of the following: visualization/dashboard/chart/graph/plot, **interactive** output, **widget**, **calculator**, **sliders**, or **interactive table**,
  OR provides explicit numeric arrays/series/time points in the conversation context,
  you MUST generate a NON-EMPTY <WIDGET> (not just empty tags).
  - For calculators/sliders: a compact interactive calculator widget is acceptable even without a “dataset”.
  - For plots/charts: include at least one interactive chart (ECharts/Plotly/D3) and optionally a sortable table (Tabulator).

Auto-widget triggers (based on what YOU wrote in <RESPONSE>):
- If <RESPONSE> includes any numeric values, percentages, currency, or formulas → generate an interactive calculator and/or visualization.
- If <RESPONSE> includes a list of steps or a comparison → generate an interactive checklist, decision table, or sortable table (Tabulator).
- If <RESPONSE> includes 5+ numeric points or a time series → generate an interactive chart (ECharts/Plotly/D3).
- If <RESPONSE> includes only 1–4 numeric points → generate KPI tiles + a small chart if it adds value (no fake data).
{visual_override_block}

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
Important: The chosen Rule applies ONLY inside <RESPONSE>. The <WIDGET> section must follow the WIDGET RULES (not the RESPONSE rule).

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
  - the numeric values present in your <RESPONSE>.
- Never invent data. Never use placeholder names (Item A, Value 1, Example Fund).
- If data is missing for a control, omit that control.
- Verify every chart/table label and every numeric literal used in JS appears in either your <RESPONSE> or the user-provided/context values.
- Do not create placeholder dates, tickers, or values to make the widget “look full”.

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
        # Strip markdown fences if model wrapped in ```...```
        if "```" in raw_widget:
            fence = re.search(r"```(?:json|html)?\s*(.*?)```", raw_widget, re.DOTALL | re.IGNORECASE)
            raw_widget = fence.group(1).strip() if fence else re.sub(r"```\w*", "", raw_widget).strip()

        widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
        if widget_mode == "json":
            widget_payload = raw_widget
        else:
            if "<" in raw_widget and ">" in raw_widget:
                if "<html" not in raw_widget.lower():
                    raw_widget = f"<html><head></head><body>{raw_widget}</body></html>"
                raw_widget = re.sub(r"<!DOCTYPE[^>]*>", "", raw_widget, flags=re.IGNORECASE).strip()
                widget_payload = inject_design_system(raw_widget)

    return response_text, widget_payload
