"""
widget_prompt_gpt4.py — GPT-4 optimized widget generation prompt.

GPT-4 is large enough to infer most patterns from training.
This prompt is shorter and higher-level than the earlier local-model version —
it sets intent and constraints, not step-by-step instructions.

Usage:
    from src.widget_prompt_gpt4 import build_widget_prompt, extract_widget_html, estimate_widget_height
    from openai import OpenAI

    client = OpenAI(api_key="your-key")

    prompt = build_widget_prompt(
        strategy_id=strategy_id,
        user_message=user_message,
        assistant_response=response,
        event=event,
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    html = extract_widget_html(completion.choices[0].message.content)
"""

from __future__ import annotations
import re


# ══════════════════════════════════════════════════════════════════════════════
# Design system — injected into every widget at post-processing time
# GPT-4 uses CSS variable names in its output; this supplies the values
# ══════════════════════════════════════════════════════════════════════════════

_DESIGN_SYSTEM_CSS = """<style id="__ds__">
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#ffffff;--bg2:#f7f8fa;--bg3:#eef0f4;
  --text:#111318;--text2:#5a5f72;--text3:#9098b0;
  --border:rgba(0,0,0,0.08);--border2:rgba(0,0,0,0.15);
  --accent:#378ADD;--accent-bg:rgba(55,138,221,0.09);--accent-b:rgba(55,138,221,0.35);
  --success:#1D9E75;--success-bg:rgba(29,158,117,0.09);
  --warn:#BA7517;--warn-bg:rgba(186,117,23,0.09);
  --danger:#E24B4A;--danger-bg:rgba(226,75,74,0.09);
  --radius:10px;--radius-sm:6px;--radius-pill:99px;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#13151c;--bg2:#1a1d27;--bg3:#20232f;
  --text:#e8eaf4;--text2:#8d93aa;--text3:#555b72;
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.14);
  --accent:#5ba4f5;--accent-bg:rgba(91,164,245,0.10);--accent-b:rgba(91,164,245,0.35);
}}
html,body{margin:0!important;padding:12px 14px!important;width:100%;box-sizing:border-box;
  background:transparent!important;color:var(--text);
  font-family:"Segoe UI",system-ui,sans-serif;font-size:14px;line-height:1.6}
@keyframes __fu{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.widget-root{animation:__fu .22s ease-out;width:100%;min-width:100%;box-sizing:border-box}
.card{background:var(--bg2);border:0.5px solid var(--border);border-radius:var(--radius);padding:14px 16px;margin-bottom:12px;transition:border-color .15s;width:100%;max-width:100%;box-sizing:border-box}
.card:hover{border-color:var(--border2)}
.card-title{font-size:11px;font-weight:500;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}
.raised{background:var(--bg);border:0.5px solid var(--border);border-radius:var(--radius);padding:14px;transition:border-color .15s,transform .1s;cursor:pointer}
.raised:hover{border-color:var(--accent-b);transform:translateY(-1px)}
.raised:active{transform:translateY(0) scale(.992)}
.raised.highlight{border:1.5px solid var(--accent-b)}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tab{padding:5px 13px;border-radius:var(--radius-sm);border:0.5px solid var(--border2);background:transparent;color:var(--text2);cursor:pointer;font-size:12px;transition:all .15s}
.tab:hover{background:var(--bg3);color:var(--text)}
.tab:active{transform:scale(.985)}
.tab.active{background:var(--bg3);color:var(--text);border-color:var(--accent)}
.panel{display:none}.panel.active{display:block}
table{width:100%;min-width:100%;border-collapse:collapse;font-size:14px}
th{background:var(--bg3);color:var(--text2);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:10px 12px;text-align:left;border-bottom:0.5px solid var(--border2)}
td{padding:10px 12px;border-bottom:0.5px solid var(--border);color:var(--text);vertical-align:middle;font-size:14px}
tr:last-child td{border-bottom:none}
tr.clickable:hover td{background:var(--accent-bg);cursor:pointer}
.search{width:100%;padding:8px 12px;border-radius:var(--radius-sm);border:0.5px solid var(--border2);background:var(--bg);color:var(--text);font-size:13px;outline:none;box-sizing:border-box;margin-bottom:10px;transition:border-color .15s}
.search:focus{border-color:var(--accent)}
.pills{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.pill{padding:4px 12px;border-radius:var(--radius-pill);border:0.5px solid var(--border2);background:transparent;color:var(--text2);cursor:pointer;font-size:12px;transition:all .15s}
.pill:hover{background:var(--bg3)}.pill.active{background:var(--accent-bg);color:var(--accent);border-color:var(--accent-b)}
.pill:active{transform:scale(.985)}
.ctrl-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.ctrl-lbl{font-size:12px;color:var(--text2);width:115px;flex-shrink:0}
.ctrl-val{font-size:12px;font-weight:500;color:var(--text);min-width:54px;text-align:right}
.btn-group{display:flex;gap:6px;flex-wrap:wrap}
.btn{padding:6px 13px;border-radius:var(--radius-sm);border:0.5px solid var(--border2);background:transparent;color:var(--text2);cursor:pointer;font-size:12px;transition:all .15s}
.btn:hover{background:var(--bg3);color:var(--text)}.btn.active{background:var(--accent-bg);color:var(--accent);border-color:var(--accent-b)}
.btn:active{transform:scale(.985)}
.ask-btn{display:inline-flex;align-items:center;gap:5px;margin-top:12px;padding:7px 14px;border-radius:var(--radius-sm);border:0.5px solid var(--border2);background:transparent;color:var(--text2);font-size:12px;cursor:pointer;transition:all .15s}
.ask-btn:hover{background:var(--accent-bg);color:var(--accent);border-color:var(--accent-b)}
.ask-btn:active{transform:scale(.985)}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500}
.b-blue{background:var(--accent-bg);color:var(--accent)}.b-green{background:var(--success-bg);color:var(--success)}
.b-amber{background:var(--warn-bg);color:var(--warn)}.b-red{background:var(--danger-bg);color:var(--danger)}.b-gray{background:var(--bg3);color:var(--text2)}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:14px}
.metric{background:var(--bg3);border-radius:var(--radius-sm);padding:12px}
.metric-lbl{font-size:11px;color:var(--text2);margin-bottom:4px}.metric-val{font-size:22px;font-weight:500}
.progress-wrap{background:var(--bg3);border-radius:var(--radius-pill);height:8px;overflow:hidden;margin-top:4px}
.progress-bar{height:100%;border-radius:var(--radius-pill);background:var(--accent);transition:width .4s ease-out}
.result-box{background:var(--bg2);border-radius:var(--radius-sm);padding:14px;margin-top:12px}
.result-lbl{font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.result-val{font-size:26px;font-weight:500}.result-sub{font-size:12px;color:var(--text2);margin-top:3px}
.step-row{display:flex;gap:12px;align-items:flex-start;padding:10px 8px;border-radius:var(--radius-sm);border-bottom:0.5px solid var(--border);cursor:pointer;transition:background .1s}
.step-row:last-child{border-bottom:none}.step-row:hover{background:var(--accent-bg)}
.step-num{width:28px;height:28px;border-radius:50%;background:var(--accent-bg);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0}
.step-title{font-size:14px;font-weight:500}.step-desc{font-size:12px;color:var(--text2);margin-top:2px;line-height:1.5}
.count-lbl{font-size:12px;color:var(--text2);margin-bottom:8px}
.empty{font-size:13px;color:var(--text2);padding:12px 0;text-align:center}
</style>"""

_SEND_PROMPT_BRIDGE = """<script>
if(!window.sendPrompt)window.sendPrompt=function(t){
  window.parent.postMessage({type:"streamlit:setComponentValue",value:t},"*");
};
function __postWidgetHeight(){
  try{
    const b=document.body, d=document.documentElement;
    const h=Math.max(
      b ? b.scrollHeight : 0,
      d ? d.scrollHeight : 0,
      b ? b.offsetHeight : 0,
      d ? d.offsetHeight : 0
    );
    window.parent.postMessage({type:"widget:height",value:(h+20)},"*");
  }catch(_e){}
}
if(!window.__widgetHeightBound){
  window.__widgetHeightBound=true;
  window.addEventListener("load",function(){
    __postWidgetHeight();
    setTimeout(__postWidgetHeight,120);
    setTimeout(__postWidgetHeight,420);
  });
  if(typeof ResizeObserver!=="undefined"){
    try{
      const ro=new ResizeObserver(function(){__postWidgetHeight();});
      ro.observe(document.body || document.documentElement);
    }catch(_e){}
  }
}
</script>"""


# ══════════════════════════════════════════════════════════════════════════════
# System prompt — short and high-level for GPT-4
# GPT-4 already knows HTML/JS/CSS deeply — this sets intent + constraints only
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Visualizer embedded in a chat assistant.

When given a user question and assistant response, you generate a single
self-contained interactive HTML widget — like the widgets in Claude.ai.

## Output rules
- Return ONLY raw HTML. No markdown fences. No explanation.
- Inline all CSS in <style> and JS in <script>.
- No frameworks. Plain HTML + CSS + JS only.
- Chart.js allowed: https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js
- Wrap everything in <div class="widget-root">
- No position:fixed. No hardcoded hex colors — CSS variables only.
- Always define: function sendPrompt(t){window.parent.postMessage({type:"streamlit:setComponentValue",value:t},"*");}
- Always call your main render/calc function on page load.

## Design system
A CSS design system with these variables is already injected:
--bg --bg2 --bg3 (backgrounds)
--text --text2 --text3 (text)
--border --border2 (borders)
--accent --accent-bg --accent-b (blue)
--success --success-bg (green)
--warn --warn-bg (amber)
--danger --danger-bg (red)
--radius --radius-sm --radius-pill

Pre-built classes available: .card .raised .card-title .tabs .tab .panel
table th td tr.clickable .search .pills .pill .ctrl-row .ctrl-lbl .ctrl-val
.btn-group .btn .ask-btn .badge .b-blue .b-green .b-amber .b-red .b-gray
.metric-grid .metric .metric-lbl .metric-val .progress-wrap .progress-bar
.result-box .result-lbl .result-val .result-sub .step-row .step-num
.step-title .step-desc .count-lbl .empty

For Chart.js colors (canvas can't use CSS vars):
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const tc=dark?'#8d93aa':'#5a5f72';
const gc=dark?'rgba(255,255,255,0.06)':'rgba(0,0,0,0.06)';

## Visual quality bar
- Aim for an enterprise dashboard look-and-feel: clean, minimal, high information density.
- Prefer clear structure: summary row → controls → main visualization → details / next steps.
- Use spacing and typography hierarchy instead of heavy borders.
- Group related controls and metrics in cards so the widget feels like a cohesive mini app.

## What to build — pick controls that fit the content

| Content type | Controls to use |
|---|---|
| 2+ options to compare | Comparison cards (.raised grid) or table, highlight best |
| 5+ list items | Search input + count label + clickable rows |
| Items with categories | Filter pills |
| Adjustable numbers | Range sliders (.ctrl-row) + live calc() + .ask-btn |
| Percentages/allocations | Progress bars (.progress-wrap) |
| Time series / trends | Chart.js line chart + period buttons |
| Rankings / comparisons | Chart.js bar chart |
| Step-by-step process | .step-row list with Prev/Next or all visible |
| Key stats / numbers | .metric-grid cards |
| 2+ sections / topics | Tabs (.tabs .tab .panel) |
| Calculator output | .result-box + .ask-btn with values baked in |

## Interaction rules
- Sliders → local calc() only. Never sendPrompt on drag.
- Slider initial values MUST match assistant_response exactly.
- If response says P=1000, r=5%, years=10, use those exact defaults (no generic defaults).
- Cards, rows, results → sendPrompt('specific question about [exact item]')
- Always add .ask-btn below calculator output
- Every clickable element needs a hover state
- sendPrompt text must be specific — never generic "tell me more"

## What not to do
- No emojis or decorative symbols in labels or UI
- No hardcoded colors
- No sendPrompt on slider drag
- No empty output on first render — always call calc() on load
- No placeholder or lorem ipsum data — use real content only
- No position:fixed
- No external fonts or icon libraries
- No decorative clutter — neat, minimal, professional design only"""


# ══════════════════════════════════════════════════════════════════════════════
# Content signal detector
# ══════════════════════════════════════════════════════════════════════════════

def _detect_signals(event: str, user_message: str, response: str) -> str:
    """Detect visual signals in the response and return hints for GPT-4."""
    hints: list[str] = []
    text  = response.lower()
    lines = response.split('\n')

    list_items  = sum(1 for l in lines if l.strip().startswith(('-','•','*','→'))
                      or (len(l.strip())>2 and l.strip()[0].isdigit() and l.strip()[1] in '.):'))
    num_count   = len(re.findall(r'\b\d+\.?\d*\b', response))
    pct_count   = len(re.findall(r'\d+\.?\d*\s*%', response))
    word_count  = len(response.split())
    section_cnt = len(re.findall(r'\n#{1,3}\s', response))

    if event == 'decision' or any(w in text for w in ['vs','versus','compare','pros','cons','difference']):
        hints.append("COMPARISON: comparison cards or table — highlight recommended option")

    if list_items > 5:
        hints.append(f"SEARCH: {list_items} items — add search input and count label")
    elif list_items > 1:
        hints.append(f"LIST: {list_items} items — interactive clickable list")

    if num_count >= 4:
        calc_kw = ['invest','compound','interest','return','growth','project',
                   'forecast','calculate','savings','mortgage','loan','retire']
        if any(w in text for w in calc_kw):
            hints.append("CALCULATOR: range sliders + live calc() + Chart.js line chart + ask button")
        else:
            hints.append(f"METRICS: {num_count} numbers — metric cards")

    if pct_count >= 2:
        hints.append(f"PROGRESS: {pct_count} percentages — animated progress bars")

    if sum(1 for w in ['step','first','second','then','next','finally','how to'] if w in text) >= 2:
        hints.append("STEPS: step list with click-to-ask on each step")

    time_kw = ['year','month','quarter','trend','growth','over time','history','forecast','annual']
    if sum(1 for w in time_kw if w in text) >= 2 and num_count >= 3:
        hints.append("CHART: time/trend data — Chart.js line chart with period buttons")

    if section_cnt >= 2 or word_count > 200:
        hints.append("TABS: 2+ sections detected — organize with tabs")

    if list_items > 3 and any(w in text for w in ['type','category','kind','group']):
        hints.append("FILTERS: categorical items — filter pills")

    # FAQ-like content => accordion for progressive disclosure.
    faq_markers = ['faq', 'frequently asked', 'q:', 'a:', 'question', 'answer']
    if any(w in text for w in faq_markers):
        hints.append("FAQ: accordion sections with concise expandable answers")

    # Elimination flow / shortlist decisions => elimination matrix.
    elim_markers = ['eliminate', 'shortlist', 'screen', 'must-have', 'nice to have', 'criteria']
    if any(w in text for w in elim_markers):
        hints.append("ELIMINATION_MATRIX: criteria-based keep/drop table with rationale")

    # Correlation/distribution style numeric pairs => bubble/scatter.
    scatter_markers = ['correlation', 'relationship', 'risk vs return', 'x-axis', 'y-axis', 'distribution']
    if any(w in text for w in scatter_markers):
        hints.append("SCATTER: bubble/scatter chart for pairwise numeric relationship")

    return '\n'.join(f"• {h}" for h in hints) if hints else "• SIMPLE: clean card with clickable content"


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def requires_local_interaction(user_message: str, assistant_response: str, event: str = "") -> bool:
    """Detect when widget should support local value tweaking (sliders/inputs + calc)."""
    text = f"{user_message}\n{assistant_response}".lower()
    strong_intent_words = [
        "slider",
        "sliders",
        "tweak",
        "adjust",
        "change values",
        "interactive calculator",
        "calculator",
        "projection",
        "project",
        "what if",
    ]
    finance_tweak_words = [
        "slider",
        "adjust",
        "tweak",
        "change",
        "what if",
        "calculate",
        "calculator",
        "projection",
        "project",
        "forecast",
        "invest",
        "interest",
        "returns",
        "years",
        "rate",
        "monthly",
        "sip",
        "emi",
        "loan",
        "mortgage",
    ]
    number_count = len(re.findall(r"\b\d+\.?\d*\b", text))
    if event.lower() in {"calculation", "planner"}:
        return True
    if any(w in text for w in strong_intent_words):
        return True
    if any(w in text for w in finance_tweak_words) and number_count >= 1:
        return True
    return False


def has_local_interaction_controls(html: str) -> bool:
    """Check that generated widget has true local interaction controls."""
    if not html:
        return False
    h = html.lower()
    has_input = (
        'type="range"' in h
        or "type='range'" in h
        or 'type="number"' in h
        or "type='number'" in h
    )
    has_calc = "function calc" in h or "oninput=\"calc(" in h or "oninput='calc(" in h
    return has_input and has_calc


INTENT_WIDGET_MAP = {
    # Event-style labels
    "decision": "comparison_cards + horizontal_bar + tabs",
    "information": "data_table + search + metric_cards",
    "confusion": "step_navigator + info_card",
    "summary": "metric_cards + bullet_list + tabs",
    "follow_up": "honor_previous_widget_type",
    # Strategy-style labels (used by this app currently)
    "comparison_table": "comparison_cards + data_table + filter_pills",
    "visualization": "line_or_bar_chart + metric_cards",
    "step_by_step": "step_navigator + insight_card",
    "structured_bullets": "insight_card + drill_down_cards",
    "narrative_prose": "tabs + insight_card + metric_cards",
    "concise_direct": "result_box + ask_button",
    "socratic_questions": "accordion_faq + ask_button",
}


def _intent_widget_hint(event: str, user_message: str) -> str:
    """Return an intent-first widget hint before content signal hints."""
    e = (event or "").strip().lower()
    if e in INTENT_WIDGET_MAP:
        return INTENT_WIDGET_MAP[e]

    q = (user_message or "").lower()
    if any(w in q for w in ["compare", "vs", "versus", "pros", "cons"]):
        return "comparison_cards + horizontal_bar + tabs"
    if any(w in q for w in ["calculate", "projection", "what if", "compound", "interest"]):
        return "range_slider + metric_cards + line_chart + ask_button"
    if any(w in q for w in ["list", "top", "show all", "find"]):
        return "search + filter_pills + data_table + ask_button"
    if any(w in q for w in ["how to", "steps", "guide"]):
        return "step_navigator + info_card"
    return "info_card + metric_cards"


def _extract_calc_defaults(text: str) -> dict:
    """Extract calculator defaults from assistant response text."""
    src = text or ""
    out = {
        "principal": None,
        "rate": None,
        "years": None,
        "n": None,
        "final": None,
    }

    p = re.search(r"(?:principal|p)\s*[:=]\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)", src, re.IGNORECASE)
    if p:
        out["principal"] = float(p.group(1).replace(",", ""))

    r = re.search(r"(?:rate|r)\s*[:=]\s*([0-9]+(?:\.\d+)?)\s*%?", src, re.IGNORECASE)
    if r:
        out["rate"] = float(r.group(1))

    y = re.search(r"(?:years?|t)\s*[:=]\s*([0-9]{1,3})\b", src, re.IGNORECASE)
    if y:
        out["years"] = int(y.group(1))

    n = re.search(r"\bn\s*[:=]\s*([0-9]{1,3})\b", src, re.IGNORECASE)
    if n:
        out["n"] = int(n.group(1))

    a = re.search(r"(?:final|amount|a|fv)\s*[≈~=:\s]+\$?\s*([0-9][0-9,]*(?:\.\d+)?)", src, re.IGNORECASE)
    if a:
        out["final"] = float(a.group(1).replace(",", ""))

    # Fallbacks from generic values in text if explicit labels are missing.
    if out["principal"] is None:
        m = re.search(r"\$([0-9][0-9,]*(?:\.\d+)?)", src)
        if m:
            out["principal"] = float(m.group(1).replace(",", ""))
    if out["rate"] is None:
        m = re.search(r"([0-9]+(?:\.\d+)?)\s*%", src)
        if m:
            out["rate"] = float(m.group(1))
    if out["years"] is None:
        m = re.search(r"([0-9]{1,2})\s*(?:years?|yrs?)", src, re.IGNORECASE)
        if m:
            out["years"] = int(m.group(1))

    return out

def build_widget_prompt(
    strategy_id: str,
    user_message: str,
    assistant_response: str,
    event: str = "",
    extra_context: str = "",
) -> str:
    """
    Build the user-turn prompt for GPT-4 widget generation.
    Pass SYSTEM_PROMPT as the system message separately.

    This version deliberately does NOT use keyword- or regex-based
    heuristics to choose controls. The model is responsible for
    reading the full assistant_response and user_message and deciding
    which interactive UI (tabs, charts, sliders, filters, tables,
    etc.) best serves the content.

    Args:
        strategy_id:        primitive id (e.g. 'comparison_table')
        user_message:       original user question
        assistant_response: text response from engine
        event:              classified event type
        extra_context:      optional hints from primitive_widget_map.py
    """
    defaults = _extract_calc_defaults(assistant_response)

    extra = f"\nPrimitive instructions:\n{extra_context}\n" if extra_context else ""
    grounding = ""
    if any(v is not None for v in defaults.values()):
        parts = []
        if defaults["principal"] is not None:
            parts.append(f"Principal = ${int(defaults['principal']):,}")
        if defaults["rate"] is not None:
            parts.append(f"Rate = {defaults['rate']}%")
        if defaults["years"] is not None:
            parts.append(f"Years = {defaults['years']}")
        if defaults["n"] is not None:
            parts.append(f"n = {defaults['n']}")
        if defaults["final"] is not None:
            parts.append(f"Final answer = ${defaults['final']:,.2f}")
        grounding = (
            "PRE-FILL THESE EXACT VALUES into slider defaults and initial state:\n"
            + "\n".join(f"- {p}" for p in parts)
            + "\nDo not invent alternative defaults."
        )

    guidance = (
        "Decide the widget layout and controls by understanding the meaning of the "
        "assistant_response and user_message. Do NOT rely on fixed keyword triggers. "
        "If the explanation compares options, use comparison-style UI; if it describes "
        "time or trends, use charts; if it walks through a process, use step-style UI; "
        "if it exposes tweakable numeric parameters, use sliders/inputs with live calc(). "
        "Always favor interactive controls that let the user explore the specific numbers, "
        "entities, and scenarios mentioned in the assistant_response.\n"
    )

    return (
        f"Event: {event or 'unknown'} | Strategy: {strategy_id}\n"
        f"{extra}\n"
        f"{grounding}\n\n"
        f"{guidance}\n"
        f"User asked:\n{user_message}\n\n"
        f"Assistant response to visualize:\n{assistant_response}\n\n"
        f"Generate the widget HTML now."
    )


def inject_design_system(html: str) -> str:
    """Post-process: inject design system CSS + sendPrompt bridge."""
    if not html:
        return html

    if '__ds__' not in html:
        if re.search(r'<head[^>]*>', html, re.IGNORECASE):
            html = re.sub(r'(<head[^>]*>)', r'\1\n' + _DESIGN_SYSTEM_CSS,
                          html, count=1, flags=re.IGNORECASE)
        else:
            html = _DESIGN_SYSTEM_CSS + html

    if 'streamlit:setComponentValue' not in html:
        if re.search(r'</body>', html, re.IGNORECASE):
            html = re.sub(r'(</body>)', _SEND_PROMPT_BRIDGE + r'\n\1',
                          html, count=1, flags=re.IGNORECASE)
        else:
            html += _SEND_PROMPT_BRIDGE

    # Force transparent body
    html = re.sub(
        r'(body\s*\{[^}]*?)background(?:-color)?\s*:\s*(?!transparent)[^;]+;',
        r'\1background:transparent!important;',
        html, flags=re.IGNORECASE
    )

    # Remove position:fixed
    html = re.sub(r'position\s*:\s*fixed', 'position:absolute', html, flags=re.IGNORECASE)

    # Enforce .widget-root wrapper so mount animation always fires.
    if 'class="widget-root"' not in html and "class='widget-root'" not in html:
        if re.search(r'<body[^>]*>', html, re.IGNORECASE):
            html = re.sub(r'(<body[^>]*>)', r'\1<div class="widget-root">', html, count=1, flags=re.IGNORECASE)
            if re.search(r'</body>', html, re.IGNORECASE):
                html = re.sub(r'(</body>)', r'</div>\n\1', html, count=1, flags=re.IGNORECASE)
            else:
                html += "</div>"
        else:
            html = f'<div class="widget-root">{html}</div>'

    # Refine generated UI borders to feel lighter in chat surfaces.
    html = re.sub(r'border\s*:\s*1px\s*solid', 'border: 0.5px solid', html, flags=re.IGNORECASE)
    html = re.sub(r'border-width\s*:\s*1px\b', 'border-width: 0.5px', html, flags=re.IGNORECASE)

    return html


def extract_widget_html(raw: str) -> str | None:
    """Extract and clean HTML from GPT-4 output."""
    if not raw:
        return None

    html = raw.strip()

    # Strip markdown fences (GPT-4 sometimes wraps in ```html)
    if '```' in html:
        fence = re.search(r'```html\s*(.*?)```', html, re.DOTALL | re.IGNORECASE)
        html  = fence.group(1).strip() if fence else re.sub(r'```\w*', '', html).strip()

    if '<' not in html or '>' not in html:
        return None

    if '<html' not in html.lower():
        html = f"<html><head></head><body>{html}</body></html>"

    html = re.sub(r'<!DOCTYPE[^>]*>', '', html, flags=re.IGNORECASE).strip()
    html = inject_design_system(html)

    return html


def estimate_widget_height(html: str) -> int:
    """Estimate iframe height from widget content."""
    if not html:
        return 300

    h = html.lower()
    height = 120
    height += h.count('<canvas')                                           * 240
    height += h.count('<table')                                            * 180
    height += min(h.count('<tr'), 20)                                      * 38
    height += (h.count('class="card')  + h.count("class='card"))          * 80
    height += (h.count('class="raised') + h.count("class='raised"))       * 100
    height += min(h.count('<li'), 12)                                      * 30
    height += (h.count('type="range')  + h.count("type='range"))          * 54
    height += h.count('<select')                                           * 50
    height += h.count('step-row')                                          * 56
    height += h.count('metric-grid')                                       * 100
    height += h.count('result-box')                                        * 90
    height += (h.count('class="tab')   + h.count("class='tab"))           * 10

    return max(280, min(height, 1200))


def build_local_calc_fallback(user_message: str, assistant_response: str) -> str:
    """Deterministic local calculator widget used when LLM output is static."""
    defaults = _extract_calc_defaults(assistant_response)
    nums = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", f"{user_message} {assistant_response}")[:3]]
    principal = int(defaults["principal"]) if defaults["principal"] is not None else (int(nums[0]) if len(nums) > 0 else 10000)
    rate = float(defaults["rate"]) if defaults["rate"] is not None else (float(nums[1]) if len(nums) > 1 else 8.0)
    years = int(defaults["years"]) if defaults["years"] is not None else (int(nums[2]) if len(nums) > 2 else 10)
    n = int(defaults["n"]) if defaults["n"] is not None else 1
    target = float(defaults["final"]) if defaults["final"] is not None else None

    principal = max(1000, min(principal, 2000000))
    rate = max(1.0, min(rate, 25.0))
    years = max(1, min(years, 40))
    n = max(1, min(n, 365))

    html = f"""
<html><head></head><body>
<div class="widget-root card">
  <div class="card-title">Interactive Projection</div>
  <div class="ctrl-row">
    <div class="ctrl-lbl">Principal</div>
    <input id="p" type="range" min="1000" max="2000000" step="1000" value="{principal}" oninput="calc()" style="flex:1" />
    <div class="ctrl-val" id="pv"></div>
  </div>
  <div class="ctrl-row">
    <div class="ctrl-lbl">Rate (%)</div>
    <input id="r" type="range" min="1" max="25" step="0.1" value="{rate}" oninput="calc()" style="flex:1" />
    <div class="ctrl-val" id="rv"></div>
  </div>
  <div class="ctrl-row">
    <div class="ctrl-lbl">Years</div>
    <input id="y" type="range" min="1" max="40" step="1" value="{years}" oninput="calc()" style="flex:1" />
    <div class="ctrl-val" id="yv"></div>
  </div>
  <div class="result-box">
    <div class="result-lbl">Projected Value</div>
    <div class="result-val" id="out">$0</div>
    <div class="result-sub" id="gain"></div>
    <div class="result-sub">Compounding frequency n = {n}</div>
    {"<div class='result-sub'>Response target: $" + f"{target:,.2f}" + "</div>" if target is not None else ""}
  </div>
  <button class="ask-btn" onclick="askAbout()">Ask about this ↗</button>
</div>
<script>
function sendPrompt(t) {{
  window.parent.postMessage({{type:"streamlit:setComponentValue",value:t}},"*");
}}
function fmt(n) {{
  return '$' + Math.round(n).toLocaleString();
}}
function calc() {{
  const P = +document.getElementById('p').value;
  const R = +document.getElementById('r').value / 100;
  const Y = +document.getElementById('y').value;
  const n = {n};
  const FV = P * Math.pow(1 + (R / n), n * Y);
  document.getElementById('pv').textContent = fmt(P);
  document.getElementById('rv').textContent = (R*100).toFixed(1) + '%';
  document.getElementById('yv').textContent = Y + 'y';
  document.getElementById('out').textContent = fmt(FV);
  document.getElementById('gain').textContent = 'Estimated gain: ' + fmt(FV - P);
}}
function askAbout() {{
  const P = document.getElementById('p').value;
  const R = document.getElementById('r').value;
  const Y = document.getElementById('y').value;
  sendPrompt('Explain this projection for principal $' + P + ', rate ' + R + '%, years ' + Y);
}}
calc();
</script>
</body></html>
""".strip()
    return inject_design_system(html)