"""Primitive-driven widget contracts, validator, and Claude-quality fallback templates.

Each primitive in primitives.json defines:
  - required_components: HTML signals that must appear in generated widget.
  - forbidden_components: signals that must NOT appear.
  - extra_context: injected into build_widget_prompt() to steer the model.
  - fallback_type: which deterministic template to render if model violates contract.
"""

from __future__ import annotations

import html as _html_mod
import json
import re
from pathlib import Path
from typing import Any

from .widget_prompt import inject_design_system

# ── Load primitives from JSON ──────────────────────────────────────────────

_PRIMITIVES_PATH = Path(__file__).resolve().parent / "primitives.json"

def _load_primitives() -> dict[str, Any]:
    try:
        return json.loads(_PRIMITIVES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

PRIMITIVES: dict[str, Any] = _load_primitives()

_DEFAULT_SPEC: dict[str, Any] = {
    "required_components": [],
    "forbidden_components": [],
    "extra_context": "",
    "fallback_type": "generic",
}


def get_primitive_spec(strategy_id: str) -> dict[str, Any]:
    return PRIMITIVES.get(strategy_id, _DEFAULT_SPEC)


# ── Component signal registry ──────────────────────────────────────────────

_COMPONENT_SIGNALS: dict[str, list[str]] = {
    "range_slider":       ['type="range"', "type='range'"],
    "calculator":         ["function calc(", "oninput=\"calc(", "oninput='calc("],
    "comparison_cards":   ["class=\"raised", "class='raised", "comparison", "highlight"],
    "tabs":               ["class=\"tab", "class='tab", ".panel"],
    "step_navigator":     ["step-row", "step-num", "step_row"],
    "chart":              ["<canvas", "Chart(", "new Chart"],
    "insight_cards":      ["class=\"card", "class='card", "card-title"],
    "result_box":         ["result-box", "result-val"],
    "prompt_chips":       ["pill", "sendPrompt"],
    "insight_card":       ["class=\"card", "class='card"],
    "search":             ['type="text"', 'oninput="filter', 'oninput=\'filter'],
    "progress_bars":      ["progress-bar", "progress-wrap"],
}

def _html_has(html: str, signals: list[str]) -> bool:
    return any(s.lower() in html.lower() for s in signals)


def validate_widget_html(html: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Return (valid, reason). Invalid means model violated primitive contract."""
    if not html:
        return False, "empty_html"
    for comp in spec.get("required_components", []):
        sigs = _COMPONENT_SIGNALS.get(comp, [comp])
        if not _html_has(html, sigs):
            return False, f"missing_required:{comp}"
    for comp in spec.get("forbidden_components", []):
        sigs = _COMPONENT_SIGNALS.get(comp, [comp])
        if _html_has(html, sigs):
            return False, f"forbidden_component:{comp}"
    return True, ""


# ── Escape helper ──────────────────────────────────────────────────────────

def _e(s: str) -> str:
    return _html_mod.escape(str(s or ""), quote=True)


# ── Fallback template helpers ──────────────────────────────────────────────

def _parse_bullets(response: str) -> list[dict[str, str]]:
    """Extract bullet points as {title, body} dicts."""
    lines = response.split("\n")
    items: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        for prefix in ("-", "•", "*", "→"):
            if stripped.startswith(prefix):
                text = stripped[len(prefix):].strip()
                if ":" in text:
                    parts = text.split(":", 1)
                    items.append({"title": parts[0].strip(), "body": parts[1].strip()})
                else:
                    items.append({"title": text[:50], "body": text})
                break
        if len(items) >= 6:
            break
    return items or [{"title": "Key Point", "body": response[:200]}]


def _parse_steps(response: str) -> list[dict[str, str]]:
    """Extract numbered steps as {title, body} dicts."""
    lines = response.split("\n")
    items: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if m:
            text = m.group(2).strip()
            if ":" in text:
                parts = text.split(":", 1)
                items.append({"title": parts[0].strip(), "body": parts[1].strip()})
            else:
                items.append({"title": f"Step {m.group(1)}", "body": text})
        if len(items) >= 8:
            break
    return items or [{"title": "Step 1", "body": response[:200]}]


def _parse_questions(response: str) -> list[str]:
    """Extract clarifying questions from response text."""
    lines = response.split("\n")
    qs: list[str] = []
    for line in lines:
        stripped = line.strip()
        stripped = re.sub(r"^[-•*\d.)]+\s*", "", stripped)
        if "?" in stripped and len(stripped) > 10:
            qs.append(stripped)
        if len(qs) >= 4:
            break
    return qs or ["Can you tell me more about what you're looking for?"]


def _parse_sections(response: str) -> list[dict[str, str]]:
    """Split response into 2-3 labelled sections for tabs."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", response) if p.strip()]
    if len(paragraphs) >= 3:
        return [
            {"label": "Summary", "body": paragraphs[0]},
            {"label": "Details", "body": " ".join(paragraphs[1:-1])},
            {"label": "Takeaways", "body": paragraphs[-1]},
        ]
    if len(paragraphs) == 2:
        return [
            {"label": "Overview", "body": paragraphs[0]},
            {"label": "Details", "body": paragraphs[1]},
        ]
    return [{"label": "Response", "body": response[:600]}]


def _parse_markdown_table(response: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def split_row(line: str) -> list[str]:
        txt = line.strip()
        if txt.startswith("|"):
            txt = txt[1:]
        if txt.endswith("|"):
            txt = txt[:-1]
        return [c.strip() for c in txt.split("|")]

    def is_sep(line: str) -> bool:
        cells = split_row(line)
        if not cells:
            return False
        return all(bool(re.match(r"^:?-{3,}:?$", c)) for c in cells)

    for i in range(len(lines) - 1):
        head_ln = lines[i]
        sep_ln = lines[i + 1]
        if "|" not in head_ln or not is_sep(sep_ln):
            continue
        cols = split_row(head_ln)
        if not cols:
            continue

        rows: list[list[str]] = []
        for ln in lines[i + 2:]:
            if "|" not in ln:
                break
            row = split_row(ln)
            if not row:
                break
            while len(row) < len(cols):
                row.append("")
            rows.append(row[: len(cols)])
        if rows:
            return cols, rows
    return None


# ── Fallback template builders ─────────────────────────────────────────────

def _fallback_comparison(user_message: str, response: str) -> str:
    parsed_table = _parse_markdown_table(response)
    if parsed_table:
        cols, rows = parsed_table
        table_head = "".join(f"<th>{_e(c)}</th>" for c in cols)
        body_rows = []
        option_idx = 0
        if cols:
            c0 = cols[0].strip().lower()
            if c0 in {"option", "plan", "item", "choice", "name"}:
                option_idx = 0
        for row in rows[:8]:
            item_name = row[option_idx] if option_idx < len(row) else row[0]
            cells = "".join(f"<td>{_e(v)}</td>" for v in row)
            body_rows.append(
                f"""<tr class="clickable"
 onclick="sendPrompt('Tell me more about {_e(item_name)} from this comparison table')">{cells}</tr>"""
            )
        table_html = f"""
<table>
  <thead><tr>{table_head}</tr></thead>
  <tbody>{''.join(body_rows)}</tbody>
</table>
"""
        return f"""<html><head></head><body><div class="widget-root">
<div class="tabs">
  <button class="tab active" onclick="showTab(this,'t-compare')">Compare</button>
  <button class="tab" onclick="showTab(this,'t-verdict')">Verdict</button>
</div>
<div id="t-compare" class="panel active">
  <div class="card">
    <div class="card-title">Comparison</div>
    {table_html}
  </div>
</div>
<div id="t-verdict" class="panel">
  <div class="card">
    <div class="card-title">Verdict</div>
    <div style="font-size:13px;line-height:1.7;color:var(--text)">{_e(response[:360])}</div>
  </div>
  <button class="ask-btn" onclick="sendPrompt('Based on this table, which option fits my goal best: {_e(user_message[:80])}?')">Ask about this ↗</button>
</div>
<script>
function showTab(btn,id){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
}}
</script>
</div></body></html>"""

    lines = [l.strip() for l in response.split("\n") if l.strip()]
    nouns: list[str] = []
    for line in lines:
        m = re.search(r"\*\*(.+?)\*\*", line)
        if m:
            nouns.append(m.group(1))
        if len(nouns) >= 2:
            break
    if len(nouns) < 2:
        nouns = ["Option A", "Option B"]

    bullets = _parse_bullets(response)
    rows_a = bullets[:3]
    rows_b = bullets[3:6] or bullets[:3]

    card_a = "\n".join(
        f'<div class="raised" style="margin-bottom:8px" onclick="sendPrompt(\'Tell me more about {_e(nouns[0])} — {_e(r["title"])}\')"><div style="font-size:12px;font-weight:500">{_e(r["title"])}</div><div style="font-size:11px;color:var(--text2);margin-top:3px">{_e(r["body"][:80])}</div></div>'
        for r in rows_a
    )
    card_b = "\n".join(
        f'<div class="raised" style="margin-bottom:8px" onclick="sendPrompt(\'Tell me more about {_e(nouns[1])} — {_e(r["title"])}\')"><div style="font-size:12px;font-weight:500">{_e(r["title"])}</div><div style="font-size:11px;color:var(--text2);margin-top:3px">{_e(r["body"][:80])}</div></div>'
        for r in rows_b
    )

    return f"""<html><head></head><body><div class="widget-root">
<div class="tabs">
  <button class="tab active" onclick="showTab(this,'t-compare')">Compare</button>
  <button class="tab" onclick="showTab(this,'t-verdict')">Verdict</button>
</div>
<div id="t-compare" class="panel active">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div>
      <div class="card-title">{_e(nouns[0])}</div>
      {card_a}
    </div>
    <div>
      <div class="card-title">{_e(nouns[1])}</div>
      {card_b}
    </div>
  </div>
</div>
<div id="t-verdict" class="panel">
  <div class="card">
    <div class="card-title">Verdict</div>
    <div style="font-size:13px;line-height:1.7;color:var(--text)">{_e(response[:300])}</div>
  </div>
  <button class="ask-btn" onclick="sendPrompt('Which is better for me — {_e(nouns[0])} or {_e(nouns[1])}? My goal is...')">Ask about this ↗</button>
</div>
<script>
function showTab(btn,id){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
}}
</script>
</div></body></html>"""


def _fallback_chart(user_message: str, response: str) -> str:
    nums = re.findall(r"\b(\d+\.?\d*)\b", response)
    labels_raw = re.findall(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)?)\b", response)
    values = [float(n) for n in nums[:8] if float(n) > 0][:8]
    labels = list(dict.fromkeys(labels_raw))[:len(values)]
    while len(labels) < len(values):
        labels.append(f"Item {len(labels)+1}")
    if not values:
        values = [10, 20, 15, 30, 25]
        labels = ["A", "B", "C", "D", "E"]

    js_labels = json.dumps(labels)
    js_values = json.dumps(values)

    return f"""<html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script></head>
<body><div class="widget-root">
<div class="metric-grid" style="margin-bottom:12px">
  <div class="metric"><div class="metric-lbl">Data points</div><div class="metric-val">{len(values)}</div></div>
  <div class="metric"><div class="metric-lbl">Max value</div><div class="metric-val">{max(values):.1f}</div></div>
  <div class="metric"><div class="metric-lbl">Average</div><div class="metric-val">{sum(values)/len(values):.1f}</div></div>
</div>
<div style="position:relative;height:220px">
  <canvas id="ch"></canvas>
</div>
<button class="ask-btn" style="margin-top:12px" onclick="sendPrompt('Explain the trend shown in this chart for {_e(user_message[:60])}')">Ask about this ↗</button>
<script>
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const tc=dark?'#8d93aa':'#5a5f72';
const gc=dark?'rgba(255,255,255,0.06)':'rgba(0,0,0,0.06)';
new Chart(document.getElementById('ch'),{{
  type:'bar',
  data:{{labels:{js_labels},datasets:[{{data:{js_values},backgroundColor:'rgba(55,138,221,0.7)',borderRadius:6,borderSkipped:false}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:tc}},grid:{{color:gc}}}},y:{{ticks:{{color:tc}},grid:{{color:gc}}}}}}}}
}});
</script>
</div></body></html>"""


def _fallback_steps(user_message: str, response: str) -> str:
    steps = _parse_steps(response)
    rows = "\n".join(
        f"""<div class="step-row" onclick="sendPrompt('Tell me more about step {i+1}: {_e(s['title'])}')">
  <div class="step-num">{i+1}</div>
  <div><div class="step-title">{_e(s['title'])}</div>
  <div class="step-desc">{_e(s['body'][:100])}</div></div>
</div>"""
        for i, s in enumerate(steps)
    )
    return f"""<html><head></head><body><div class="widget-root">
<div class="card">
  <div class="card-title">{len(steps)} Steps</div>
  {rows}
</div>
<button class="ask-btn" onclick="sendPrompt('I completed the steps for {_e(user_message[:60])}. What should I do next?')">What next ↗</button>
</div></body></html>"""


def _fallback_bullets(user_message: str, response: str) -> str:
    items = _parse_bullets(response)
    cards = "\n".join(
        f"""<div class="raised" onclick="sendPrompt('Tell me more about: {_e(b['title'])}')">
  <div style="font-size:11px;font-weight:600;color:var(--accent);margin-bottom:4px">#{i+1}</div>
  <div style="font-size:13px;font-weight:500">{_e(b['title'])}</div>
  <div style="font-size:12px;color:var(--text2);margin-top:4px">{_e(b['body'][:100])}</div>
</div>"""
        for i, b in enumerate(items)
    )
    return f"""<html><head></head><body><div class="widget-root">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
{cards}
</div>
<button class="ask-btn" onclick="sendPrompt('Which of these points matters most for {_e(user_message[:60])}?')">Explore further ↗</button>
</div></body></html>"""


def _fallback_narrative(user_message: str, response: str) -> str:
    sections = _parse_sections(response)
    tabs_html = "\n".join(
        f'<button class="tab{" active" if i==0 else ""}" onclick="showTab(this,\'s{i}\')">{_e(s["label"])}</button>'
        for i, s in enumerate(sections)
    )
    panels_html = "\n".join(
        f'<div id="s{i}" class="panel{" active" if i==0 else ""}"><div class="card"><div style="font-size:13px;line-height:1.75;color:var(--text)">{_e(s["body"])}</div></div></div>'
        for i, s in enumerate(sections)
    )
    return f"""<html><head></head><body><div class="widget-root">
<div class="tabs">{tabs_html}</div>
{panels_html}
<button class="ask-btn" onclick="sendPrompt('I want to explore {_e(user_message[:60])} in more detail')">Go deeper ↗</button>
<script>
function showTab(btn,id){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
}}
</script>
</div></body></html>"""


def _fallback_concise(user_message: str, response: str) -> str:
    short = response.strip()[:280]
    return f"""<html><head></head><body><div class="widget-root">
<div class="result-box">
  <div class="result-lbl">Answer</div>
  <div style="font-size:15px;font-weight:500;line-height:1.6;color:var(--text)">{_e(short)}</div>
</div>
<button class="ask-btn" onclick="sendPrompt('Can you explain this in more detail: {_e(user_message[:60])}')">Explain more ↗</button>
</div></body></html>"""


def _fallback_socratic(user_message: str, response: str) -> str:
    questions = _parse_questions(response)
    icons = ["🧮", "📈", "🎯", "💡", "🔍", "📊"]

    options = "\n".join(
        f"""
<div class="step-row" style="cursor:pointer" onclick="sendPrompt({json.dumps(q)})">
  <div class="step-num">{icons[i % len(icons)]}</div>
  <div>
    <div class="step-title">{_e(q[:80])}</div>
    <div class="step-desc" style="margin-top:3px">
      {_e("Click to answer this and I will adapt the next explanation or calculator to that choice.")}
    </div>
  </div>
</div>"""
        for i, q in enumerate(questions)
    )

    return f"""<html><head></head><body><div class="widget-root">
<div class="card">
  <div class="card-title">Help me understand</div>
  <div style="font-size:13px;color:var(--text2);margin-bottom:12px">
    Click the option that best matches your situation. I will use your choice to decide whether to show a calculator, comparison table, or step-by-step guide next.
  </div>
  {options}
</div>
</div></body></html>"""


def _fallback_generic(user_message: str, response: str) -> str:
    return _fallback_concise(user_message, response)


_FALLBACK_BUILDERS = {
    "comparison": _fallback_comparison,
    "chart":      _fallback_chart,
    "steps":      _fallback_steps,
    "bullets":    _fallback_bullets,
    "narrative":  _fallback_narrative,
    "concise":    _fallback_concise,
    "socratic":   _fallback_socratic,
    "generic":    _fallback_generic,
}


def build_primitive_fallback(strategy_id: str, user_message: str, response: str) -> str:
    """Build a deterministic Claude-quality fallback widget for the given primitive."""
    spec = get_primitive_spec(strategy_id)
    fallback_type = spec.get("fallback_type", "generic")
    builder = _FALLBACK_BUILDERS.get(fallback_type, _fallback_generic)
    raw_html = builder(user_message, response)
    return inject_design_system(raw_html)
