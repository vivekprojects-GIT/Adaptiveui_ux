"""Utility helpers: math, response enforcement, and valence heuristics.

This module provides:
- sigmoid + uncertainty helpers
- lightweight valence detection (heuristic)
- format override + explore trigger detection
- response post-processing to enforce selected format
"""

import json
import re
import math
import numpy as np


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -500, 500))
    return 1.0 / (1.0 + math.exp(-x))


def mean_uncertainty(sigma_inv: np.ndarray) -> float:
    prec = np.diag(sigma_inv)
    return float(np.mean(1.0 / np.clip(prec, 1e-8, None)))


# --- Valence signals ---
_POS = re.compile(
    r"\b(thank|thanks|great|perfect|awesome|love|helpful|useful|exactly|makes sense|clear|brilliant|nice|good|yes|correct|right)\b",
    re.I,
)
_NEG = re.compile(
    r"\b(wrong|incorrect|confused|confusing|not what|still don.t|don't understand|that.s not|try again|again\b|useless|unhelpful|bad|nope\b|nah\b|wtf\b|huh\??|noooo+)\b",
    re.I,
)
_REPHRASE = re.compile(r"\b(what I mean|let me rephrase|I said|as I mentioned|again|once more)\b", re.I)


def fast_valence(message: str, prev_response: str) -> dict:
    """Return heuristic valence and a human-readable reason string."""
    if not prev_response:
        return {"pos": 0.5, "neg": 0.1, "reason": "first message"}
    pos_hits = len(_POS.findall(message))
    neg_hits = len(_NEG.findall(message))
    rephr = 1 if _REPHRASE.search(message) else 0
    len_ratio = len(message) / max(len(prev_response), 1)
    brevity_neg = 0.3 if (len_ratio < 0.08 and len(message) < 15) else 0.0

    pos = float(np.clip(0.3 + 0.3 * pos_hits - 0.1 * neg_hits, 0.0, 1.0))
    neg = float(np.clip(0.1 + 0.3 * neg_hits + 0.2 * rephr + brevity_neg, 0.0, 1.0))

    reasons = []
    if pos_hits:
        reasons.append(f"{pos_hits} positive signal(s)")
    if neg_hits:
        reasons.append(f"{neg_hits} negative signal(s)")
    if rephr:
        reasons.append("rephrase")
    if brevity_neg:
        reasons.append("very short reply")
    return {"pos": pos, "neg": neg, "reason": ", ".join(reasons) or "neutral"}


# --- Explicit format override detection (user mentions what they want) ---
_OVERRIDE_PATTERNS = [
    ("structured_bullets", re.compile(r"\b(bullets?|bullet points?|list it|in bullets)\b", re.I)),
    ("step_by_step", re.compile(r"\b(step by step|steps?|walk me through|procedure)\b", re.I)),
    ("concise_direct", re.compile(r"\b(concise|short|tl;dr|tldr|in 1-3 sentences)\b", re.I)),
    ("narrative_prose", re.compile(r"\b(paragraph|narrative|in prose|explain like a story)\b", re.I)),
    ("socratic_questions", re.compile(r"\b(ask me|ask questions|clarifying questions?)\b", re.I)),
    ("comparison_table", re.compile(r"\b(table|comparison table|pros and cons|compare|vs\.?|versus)\b", re.I)),
    ("visualization", re.compile(r"\b(chart|plot|graph|visuali[sz]e|visualization|bar chart|pie chart)\b", re.I)),
]


def detect_format_override(message: str, available: list[str]) -> str | None:
    m = (message or "").strip().lower()
    for strat, pat in _OVERRIDE_PATTERNS:
        if strat in available and pat.search(m):
            return strat
    return None


# --- Explore triggers (force trying a different format) ---
_EXPLORE_TRIG = re.compile(r"\b(try again|different|another way|not this|still the same|nope\b|nah\b|noooo+)\b", re.I)


def detect_explore_trigger(message: str) -> bool:
    return bool(_EXPLORE_TRIG.search((message or "").strip()))


def negative_strength(ev: dict) -> float:
    """Map heuristic valence to a [0,1] strength scalar."""
    if not ev:
        return 0.0
    return float(np.clip(ev.get("neg", 0.0), 0.0, 1.0))


# --- Response enforcement ---
def enforce_response(strategy: str, text: str) -> str:
    """Post-process model output to strongly encourage the chosen format."""
    t = (text or "").strip()

    if strategy == "structured_bullets":
        parts = re.split(r"[\n]+", t)
        if len(parts) <= 2 and len(t) > 160:
            parts = re.split(r"(?<=[.!])\s+", t)
        cleaned = []
        for p in parts:
            p = p.strip().lstrip("-•* ").strip()
            if not p:
                continue
            if "?" in p:
                continue
            cleaned.append(p)
        cleaned = cleaned[:5]
        if len(cleaned) < 3:
            cleaned = cleaned or [t.replace("?", "").strip()]
        return "\n".join(["- " + c for c in cleaned[:5]])

    if strategy == "step_by_step":
        lines = [ln.strip() for ln in re.split(r"[\n]+", t) if ln.strip()]
        items = []
        for ln in lines:
            ln = re.sub(r"^([\-*•]|\d+[.)])\s*", "", ln).strip()
            if ln:
                items.append(ln)
        items = items[:6] or [t]
        return "\n".join([f"{i+1}. {it}" for i, it in enumerate(items[:6])])

    if strategy == "concise_direct":
        sents = re.split(r"(?<=[.!?])\s+", t)
        return " ".join(sents[:3]).strip()

    if strategy == "socratic_questions":
        qs = re.findall(r"[^\n?]*\?", t)
        if qs:
            qs = [q.strip() for q in qs if q.strip()][:2]
            return "Got it.\n" + "\n".join(["- " + q for q in qs])
        return "Got it.\n- What outcome do you want?\n- Any constraints or example input/output?"

    if strategy == "comparison_table":
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and isinstance(obj.get("columns"), list) and isinstance(obj.get("rows"), list):
                return json.dumps(obj)
        except Exception:
            pass

        lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]
        if len(table_lines) >= 2:
            if not any(re.match(r"^\|?\s*[:-]{2,}", ln) for ln in table_lines):
                header = table_lines[0]
                cols = [c.strip() for c in header.strip("|").split("|")]
                sep = "|" + "|".join(["---"] * len(cols)) + "|"
                return "\n".join([header, sep] + table_lines[1:6])
            return "\n".join(table_lines[:8])

        fallback = {
            "columns": ["Option", "Pros", "Cons", "Best for"],
            "rows": [
                ["A", "", "", ""],
                ["B", "", "", ""],
            ],
        }
        return json.dumps(fallback)

    if strategy == "visualization":
        try:
            obj = json.loads(t)
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("type"), str)
                and isinstance(obj.get("labels"), list)
                and isinstance(obj.get("values"), list)
            ):
                return json.dumps(obj)
        except Exception:
            pass

        m = re.search(r"```[\s\S]*?```", t)
        if m:
            return m.group(0)

        fallback = {
            "type": "bar",
            "title": "Visualization",
            "labels": ["A", "B"],
            "values": [1, 1],
            "x_label": "Category",
            "y_label": "Value",
        }
        return json.dumps(fallback)

    return t
