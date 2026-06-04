"""APE for the RAG flow — a contextual Thompson-Sampling bandit over PRESENTATION
FORMATS (not text strategies). It learns, per evidence-shape context, which format
gets positive feedback, and only ever chooses among FEASIBLE formats for the data.

Beta-Bernoulli arms keyed by (context, format). Persisted to ape_state.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_STATE_PATH = Path(__file__).resolve().parent / "ape_state.json"


def _load() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(state: dict) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _key(context: str, fmt: str) -> str:
    return f"{context}|{fmt}"


def _arm(state: dict, context: str, fmt: str) -> dict:
    return state.get(_key(context, fmt), {"alpha": 1.0, "beta": 1.0})


def select_format(feasible: list[str], context: str) -> dict:
    """Thompson Sampling over feasible formats within this context."""
    state = _load()
    if not feasible:
        feasible = ["text"]
    samples = {}
    posteriors = {}
    best_fmt, best_theta = feasible[0], -1.0
    for fmt in feasible:
        arm = _arm(state, context, fmt)
        a, b = float(arm["alpha"]), float(arm["beta"])
        theta = float(np.random.beta(a, b))
        samples[fmt] = theta
        posteriors[fmt] = {
            "alpha": a, "beta": b,
            "mean": round(a / (a + b), 3),
            "trials": int(a + b - 2),
        }
        if theta > best_theta:
            best_theta, best_fmt = theta, fmt
    return {"chosen": best_fmt, "context": context, "feasible": feasible,
            "sampled": {k: round(v, 3) for k, v in samples.items()}, "posteriors": posteriors}


def update(context: str, fmt: str, reward: float) -> dict:
    """reward in [0,1] (1 = 👍, 0 = 👎). Beta update on the chosen arm."""
    state = _load()
    key = _key(context, fmt)
    arm = state.get(key, {"alpha": 1.0, "beta": 1.0})
    if reward >= 0.5:
        arm["alpha"] = float(arm["alpha"]) + 1.0
    else:
        arm["beta"] = float(arm["beta"]) + 1.0
    state[key] = arm
    _save(state)
    return {"context": context, "format": fmt, "alpha": arm["alpha"], "beta": arm["beta"]}
