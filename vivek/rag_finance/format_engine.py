"""Deterministic format engine — picks the widget format from the answer's shape.

No bandit, no learning, no randomness. Given the feasible formats + shape signals,
it deterministically chooses the best-fit visual:

  trend (>=3 periods)            -> line
  composition (segments, no trend) -> pie
  comparison (>=2 periods)       -> bar
  some numbers                   -> kpi_row
  otherwise                      -> text
"""

from __future__ import annotations


def pick_format(shape: dict) -> dict:
    f = shape.get("features", {})
    feasible = set(shape.get("feasible", []))

    if f.get("has_time_series") and "line" in feasible:
        chosen = "line"
    elif f.get("has_composition") and not f.get("has_time_series") and "pie" in feasible:
        chosen = "pie"
    elif f.get("has_comparison") and "bar" in feasible:
        chosen = "bar"
    elif "kpi_row" in feasible:
        chosen = "kpi_row"
    else:
        chosen = "text"

    return {
        "chosen_format": chosen,
        "context": shape.get("context"),
        "feasible": shape.get("feasible", []),
        "features": f,
        "engine": "deterministic",
    }
