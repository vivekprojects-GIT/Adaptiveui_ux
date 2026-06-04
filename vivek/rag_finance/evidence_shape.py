"""Analyze the SYNTHESIZED RESPONSE's shape to decide which presentation formats
are feasible — so APE never picks a format the answer can't support. Works on the
answer text (no structured evidence packet)."""

from __future__ import annotations

import re

FORMATS = ["kpi_row", "line", "bar", "area", "pie", "donut", "text"]

_PERIOD_RE = re.compile(r"Q[1-4]\s+20\d{2}|FY\s?20\d{2}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_COMPOSITION_KW = ("segment", "by segment", "breakdown", "composition", "mix", "share of", "split")


def analyze(response: str) -> dict:
    text = response or ""
    low = text.lower()

    periods = sorted(set(_PERIOD_RE.findall(text)))
    numbers = _NUM_RE.findall(text)
    has_numbers = len([n for n in numbers if n not in {p[-4:] for p in periods}]) > 0  # exclude bare years

    has_time_series = len(periods) >= 3
    has_comparison = len(periods) >= 2
    has_composition = any(k in low for k in _COMPOSITION_KW)

    feasible: set[str] = {"text"}
    if has_numbers:
        feasible.add("kpi_row")
    if has_time_series:
        feasible |= {"line", "area", "bar"}
    if has_comparison:
        feasible.add("bar")
    if has_composition:
        feasible |= {"pie", "donut"}

    # Priority matches the format engine: a trend wins over an incidental segment mention.
    if has_time_series:
        context = "timeseries"
    elif has_composition:
        context = "composition"
    elif has_comparison:
        context = "comparison"
    else:
        context = "single"

    features = {
        "n_periods": len(periods),
        "n_numbers": len(numbers),
        "has_numbers": has_numbers,
        "has_time_series": has_time_series,
        "has_composition": has_composition,
        "has_comparison": has_comparison,
    }
    return {"feasible": sorted(feasible), "context": context, "features": features}
