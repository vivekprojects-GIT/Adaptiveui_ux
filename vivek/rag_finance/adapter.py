"""Adapt the evidence-bound widget (points + source_ref) into the schema the
existing Vue renderer understands (series / items), so RAG widgets render with
your real components. Verification (source_refs) is reported separately."""

from __future__ import annotations


def _fmt(value, unit: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "USD_millions":
        return f"${v:,.1f}M"
    return f"{v:,.1f}"


def to_renderer_schema(evidence_widget: dict) -> dict:
    layout: list[dict] = []
    for b in evidence_widget.get("layout", []) or []:
        t = str(b.get("type") or "").lower()

        if t == "text":
            layout.append({"type": "text", "content": b.get("content", "")})

        elif t == "kpi_row":
            items = [
                {"label": it.get("label"), "value": _fmt(it.get("value"), it.get("unit", "")), "tone": "neutral"}
                for it in (b.get("items") or [])
            ]
            if items:
                layout.append({"type": "kpi_row", "items": items})

        elif t == "chart":
            kind = str(b.get("kind") or "line").lower()
            # Normalize to a list of series (handle both single `points` and multi `series`).
            if isinstance(b.get("series"), list) and b["series"]:
                series_in = b["series"]
            else:
                series_in = [{"name": b.get("title") or "Value", "kind": None, "points": b.get("points") or []}]
            if not any((s.get("points") or []) for s in series_in):
                continue
            first_pts = series_in[0].get("points") or []

            if kind in ("pie", "donut", "funnel", "gauge"):
                chart: dict = {"kind": kind, "items": [{"label": p.get("label"), "value": p.get("value")} for p in first_pts]}
                if b.get("max") is not None:
                    chart["max"] = b.get("max")
                layout.append({"type": "chart", "title": b.get("title"), "chart": chart})
            else:
                # cartesian (line/bar/hbar/area/scatter/stacked/combo) + radar — shared labels, N series
                cats = [p.get("label") for p in first_pts]
                rseries = []
                for i, s in enumerate(series_in):
                    pts = s.get("points") or []
                    entry: dict = {"name": s.get("name") or f"Series {i + 1}",
                                   "values": [[j, p.get("value")] for j, p in enumerate(pts)]}
                    if s.get("kind"):
                        entry["kind"] = s.get("kind")
                    rseries.append(entry)
                layout.append({
                    "type": "chart", "title": b.get("title"),
                    "chart": {"kind": kind, "x_categories": cats, "series": rseries},
                })

    return {"version": "1.0", "layout": layout}
