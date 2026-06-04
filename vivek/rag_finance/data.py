"""Synthetic finance dataset for the RAG app.

Generates fictional company filings (so nothing is mistaken for real audited data)
with quarterly financials. Every numeric fact gets a stable `source_ref` so the
widget validator can prove each chart/table value is grounded in evidence.

Two products:
  - documents:      retrievable text chunks (narrative + table) for ChromaDB
  - evidence_index: source_ref -> {value, unit, metric, company, period, label}
"""

from __future__ import annotations

import random

# Fictional companies (NOT real — avoids confusion with audited filings).
COMPANIES = [
    {"id": "LUM", "name": "Lumina Systems", "sector": "Software",
     "segments": ["Cloud Platform", "Licenses", "Services"], "base_rev": 820.0, "growth": 0.06, "margin": 78.0},
    {"id": "APX", "name": "Apex Financial", "sector": "Financial Services",
     "segments": ["Lending", "Wealth Management", "Payments"], "base_rev": 1340.0, "growth": 0.03, "margin": 61.0},
    {"id": "VTX", "name": "Vertex Energy", "sector": "Energy",
     "segments": ["Renewables", "Grid Services", "Storage"], "base_rev": 2110.0, "growth": 0.04, "margin": 44.0},
    {"id": "MER", "name": "Meridian Retail", "sector": "Consumer Retail",
     "segments": ["E-commerce", "Stores", "Wholesale"], "base_rev": 1750.0, "growth": 0.02, "margin": 37.0},
]

QUARTERS = [(y, q) for y in (2023, 2024) for q in (1, 2, 3, 4)]  # 8 quarters


def _round(x: float, n: int = 1) -> float:
    return round(x, n)


def generate_documents() -> list[dict]:
    """Deterministic synthetic filings (seeded, so re-runs are identical)."""
    rng = random.Random(42)
    docs: list[dict] = []

    for co in COMPANIES:
        rev = co["base_rev"]
        for (year, q) in QUARTERS:
            # seasonal + growth + small noise
            season = 1.0 + (0.12 if q == 4 else (-0.04 if q == 1 else 0.0))
            rev = rev * (1 + co["growth"])
            quarter_rev = _round(rev * season * (1 + rng.uniform(-0.02, 0.02)))
            gross_margin = _round(co["margin"] + rng.uniform(-1.5, 1.5))
            opex = _round(quarter_rev * (0.30 + rng.uniform(-0.03, 0.03)))
            net_income = _round(quarter_rev * (gross_margin / 100.0) - opex)
            # segment split
            weights = [rng.uniform(0.5, 1.0) for _ in co["segments"]]
            wsum = sum(weights)
            seg_rev = {s: _round(quarter_rev * (w / wsum)) for s, w in zip(co["segments"], weights)}

            period = f"Q{q} {year}"
            doc_id = f"{co['id']}_{year}Q{q}"

            seg_lines = "\n".join(f"  - {s}: ${v:,.1f}M" for s, v in seg_rev.items())
            narrative = (
                f"{co['name']} ({co['id']}) — {period} Financial Summary ({co['sector']} sector).\n"
                f"Total revenue for {period} was ${quarter_rev:,.1f}M, with a gross margin of {gross_margin:.1f}%. "
                f"Net income was ${net_income:,.1f}M after operating expenses of ${opex:,.1f}M.\n"
                f"Revenue by segment:\n{seg_lines}\n"
                f"Source: {co['name']} internal management report {doc_id}."
            )

            metrics = {
                "revenue": quarter_rev,
                "net_income": net_income,
                "gross_margin_pct": gross_margin,
                "operating_expenses": opex,
            }
            docs.append({
                "doc_id": doc_id,
                "company": co["name"],
                "company_id": co["id"],
                "sector": co["sector"],
                "period": period,
                "year": year,
                "quarter": q,
                "metrics": metrics,
                "segments": seg_rev,
                "text": narrative,
            })
    return docs


def build_evidence_index(docs: list[dict]) -> dict[str, dict]:
    """source_ref -> evidence record. Mirrors every number in the documents."""
    idx: dict[str, dict] = {}
    metric_labels = {
        "revenue": ("Total Revenue", "USD_millions"),
        "net_income": ("Net Income", "USD_millions"),
        "gross_margin_pct": ("Gross Margin", "percent"),
        "operating_expenses": ("Operating Expenses", "USD_millions"),
    }
    for d in docs:
        for metric, value in d["metrics"].items():
            label, unit = metric_labels[metric]
            ref = f"{d['doc_id']}/income/{metric}"
            idx[ref] = {
                "source_ref": ref, "value": value, "unit": unit, "metric": metric,
                "company": d["company"], "company_id": d["company_id"],
                "period": d["period"], "year": d["year"], "quarter": d["quarter"],
                "label": f"{d['company']} {d['period']} {label}",
            }
        for seg, value in d["segments"].items():
            ref = f"{d['doc_id']}/segments/{seg.replace(' ', '_')}"
            idx[ref] = {
                "source_ref": ref, "value": value, "unit": "USD_millions", "metric": "segment_revenue",
                "segment": seg, "company": d["company"], "company_id": d["company_id"],
                "period": d["period"], "year": d["year"], "quarter": d["quarter"],
                "label": f"{d['company']} {d['period']} {seg} revenue",
            }
    return idx


if __name__ == "__main__":
    docs = generate_documents()
    idx = build_evidence_index(docs)
    print(f"Generated {len(docs)} documents, {len(idx)} evidence facts.")
    print("\nExample document:\n")
    print(docs[0]["text"])
    print("\nExample evidence refs:")
    for ref in list(idx)[:4]:
        print(" ", ref, "->", idx[ref]["value"], idx[ref]["unit"])
