"""Single-call RAG pipeline:

  query -> retrieve (ChromaDB, relevance-gated)
        -> ONE LLM call: <RESPONSE> (grounded) + <WIDGET> (JSON built from the response)
        -> registry-match + component check + answer check
        -> render with our components

The model picks the widget format inline; the validator drops any block type /
chart kind we don't have a component for, and any number not stated in the answer.
"""

from __future__ import annotations

from .combined import generate
from .retriever import retrieve
from .validator import parse_widget, validate

_NO_DATA_MSG = (
    "I couldn't find matching filings in the database for this question, so I won't show a chart. "
    "The dataset covers Apple, Microsoft, Alphabet, Amazon, NVIDIA, Tesla, and Meta "
    "(roughly FY2021-FY2023/24 annual results)."
)

_SUPPORTED = ["line", "bar", "area", "pie", "donut", "kpi_row", "text"]


def _derive_format(widget: dict) -> str:
    layout = widget.get("layout", []) or []
    for b in layout:
        if str(b.get("type")).lower() == "chart":
            return str(b.get("kind") or "chart").lower()
    for b in layout:
        if str(b.get("type")).lower() == "kpi_row":
            return "kpi_row"
    return "text"


def answer(query: str, k: int = 6) -> dict:
    r = retrieve(query, k=k)

    # RELEVANCE GATE: nothing close enough → refuse, no widget.
    if not r["relevant"]:
        return {
            "query": query,
            "response": _NO_DATA_MSG,
            "retrieved_docs": [],
            "evidence_facts": 0,
            "format": {"chosen_format": "none", "context": "no_data", "feasible": [], "features": {}},
            "widget": {"layout": []},
            "verification": {"total_points": 0, "passed": 0, "failed": 0,
                             "grounded": True, "has_data": False, "checks": []},
            "best_distance": r.get("best_distance"),
        }

    # ONE call: response + widget JSON (widget generated from the response).
    response, widget_raw = generate(query, r["chunks"])

    # Registry-match + component check + answer check.
    widget = parse_widget(widget_raw)
    clean_widget, report = validate(widget, response=response)

    return {
        "query": query,
        "response": response,
        "retrieved_docs": r["doc_ids"],
        "evidence_facts": len(r["chunks"]),
        "format": {
            "chosen_format": _derive_format(clean_widget),
            "context": "model",
            "feasible": _SUPPORTED,
            "features": {},
        },
        "widget": clean_widget,
        "verification": report,
    }
