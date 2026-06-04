"""Response-consistency validation.

No evidence packet — the synthesized ANSWER is the source of truth. Every numeric
value in the widget must appear in the answer text (within tolerance). Numbers the
answer never stated are dropped, so the widget can't add unstated/hallucinated data.
The answer itself is grounded by the synthesizer (retrieved context only).
"""

from __future__ import annotations

import json
import re

_TOL = 0.01  # 1% tolerance for rounding/formatting differences

from .registry import allowed_chart_kinds, allowed_types


# Derived from widget-registry.json (single source of truth) at call time.
def _supported_types() -> set[str]:
    return {t.lower() for t in allowed_types()}


def _supported_kinds() -> set[str]:
    return {k.lower() for k in allowed_chart_kinds()}


def _num(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d[\d,]*\.?\d*", v.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def numbers_in_text(text: str) -> list[float]:
    out: list[float] = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _in_response(value, resp_nums: list[float]) -> tuple[bool, str]:
    got = _num(value)
    if got is None:
        return False, "value not numeric"
    for n in resp_nums:
        denom = abs(n) if abs(n) > 1e-9 else 1.0
        if abs(got - n) / denom <= _TOL:
            return True, "found in answer"
    return False, f"value {got} not stated in the answer"


def parse_widget(raw: str) -> dict:
    s = (raw or "").strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                return {"layout": []}
        return {"layout": []}


def validate(widget: dict, response: str = "") -> tuple[dict, dict]:
    """Returns (clean_widget, report). Drops any number not present in the answer."""
    resp_nums = numbers_in_text(response)
    supported_types = _supported_types()   # from widget-registry.json
    supported_kinds = _supported_kinds()   # from widget-registry.json
    clean: list[dict] = []
    checks: list[dict] = []
    passed = failed = 0

    for block in widget.get("layout", []) or []:
        t = str(block.get("type") or "").lower()

        # COMPONENT CHECK: drop block types our components can't render.
        if t not in supported_types:
            checks.append({"block": t or "unknown", "label": None, "value": None,
                           "ok": False, "detail": "no component for this block type"})
            failed += 1
            continue

        if t == "text":
            if str(block.get("content") or "").strip():
                clean.append(block)
            continue

        if t == "kpi_row":
            kept = []
            for it in block.get("items", []) or []:
                ok, why = _in_response(it.get("value"), resp_nums)
                checks.append({"block": "kpi_row", "label": it.get("label"),
                               "value": it.get("value"), "ok": ok, "detail": why})
                if ok:
                    kept.append(it); passed += 1
                else:
                    failed += 1
            if kept:
                clean.append({**block, "items": kept})
            continue

        if t == "chart":
            # COMPONENT CHECK: drop charts whose kind our chart component can't draw.
            kind = str(block.get("kind") or "").lower()
            if kind not in supported_kinds:
                checks.append({"block": "chart", "label": block.get("title"), "value": None,
                               "ok": False, "detail": f"unsupported chart kind '{kind}' — no component"})
                failed += 1
                continue

            def _check_points(points):
                out = []
                for p in points or []:
                    ok, why = _in_response(p.get("value"), resp_nums)
                    checks.append({"block": "chart", "label": p.get("label"),
                                   "value": p.get("value"), "ok": ok, "detail": why})
                    if ok:
                        out.append(p)
                        nonlocal_counts[0] += 1
                    else:
                        nonlocal_counts[1] += 1
                return out

            nonlocal_counts = [0, 0]  # [passed, failed] accumulator for this block
            if isinstance(block.get("series"), list):  # multi-series chart
                new_series = []
                for s in block["series"]:
                    kept_pts = _check_points(s.get("points"))
                    if kept_pts:
                        new_series.append({**s, "points": kept_pts})
                if new_series:
                    clean.append({**block, "series": new_series})
            else:  # single-series chart
                kept = _check_points(block.get("points"))
                if kept:
                    clean.append({**block, "points": kept})
            passed += nonlocal_counts[0]
            failed += nonlocal_counts[1]
            continue

        # unknown block type → drop

    report = {
        "total_points": passed + failed,
        "passed": passed,
        "failed": failed,
        # Consistent = nothing in the widget that the answer didn't state.
        "grounded": failed == 0,
        "has_data": passed > 0,
        "checks": checks,
    }
    return {"layout": clean}, report
