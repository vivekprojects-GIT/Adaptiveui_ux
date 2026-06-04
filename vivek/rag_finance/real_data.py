"""Real public-company financial data as plain TEXT CHUNKS for the RAG corpus.

No structured evidence index — just documents. Figures are real reported public
financials (approximate, from training knowledge; for a production system you would
ingest the actual 10-K/10-Q filings). One chunk per company-fiscal-year.
"""

from __future__ import annotations

# (company, ticker, sector, fiscal_year_note, revenue_$B, net_income_$B, gross_margin_%, segments {name: $B}, note)
_RECORDS = [
    # ---- Apple (fiscal year ends September) ----
    ("Apple", "AAPL", "Technology", "FY2023 (ended Sep 2023)", 383.3, 97.0, 44.1,
     {"iPhone": 200.6, "Services": 85.2, "Wearables, Home & Accessories": 39.8, "Mac": 29.4, "iPad": 28.3},
     "Services revenue reached an all-time high; total revenue declined about 3% year over year."),
    ("Apple", "AAPL", "Technology", "FY2022 (ended Sep 2022)", 394.3, 99.8, 43.3,
     {"iPhone": 205.5, "Services": 78.1, "Mac": 40.2, "Wearables, Home & Accessories": 41.2, "iPad": 29.3},
     "Record annual revenue driven by iPhone and Services."),
    ("Apple", "AAPL", "Technology", "FY2021 (ended Sep 2021)", 365.8, 94.7, 41.8,
     {"iPhone": 192.0, "Services": 68.4, "Wearables, Home & Accessories": 38.4, "Mac": 35.2, "iPad": 31.9},
     "Strong rebound year with double-digit growth across most segments."),

    # ---- Microsoft (fiscal year ends June) ----
    ("Microsoft", "MSFT", "Technology", "FY2023 (ended Jun 2023)", 211.9, 72.4, 69.0,
     {"Intelligent Cloud": 87.9, "Productivity & Business Processes": 69.3, "More Personal Computing": 54.7},
     "Azure and cloud services led growth in the Intelligent Cloud segment."),
    ("Microsoft", "MSFT", "Technology", "FY2022 (ended Jun 2022)", 198.3, 72.7, 68.4,
     {"Intelligent Cloud": 75.3, "Productivity & Business Processes": 63.4, "More Personal Computing": 59.7},
     "Broad-based growth across all three segments."),
    ("Microsoft", "MSFT", "Technology", "FY2021 (ended Jun 2021)", 168.1, 61.3, 68.9,
     {"Intelligent Cloud": 60.1, "More Personal Computing": 54.1, "Productivity & Business Processes": 53.9},
     "Accelerated digital transformation drove cloud demand."),

    # ---- Alphabet (calendar year) ----
    ("Alphabet", "GOOGL", "Technology", "FY2023", 307.4, 73.8, 56.6,
     {"Google advertising": 237.9, "Google Cloud": 33.1, "Other": 36.4},
     "Google Cloud turned profitable; advertising reaccelerated."),
    ("Alphabet", "GOOGL", "Technology", "FY2022", 282.8, 60.0, 55.4,
     {"Google advertising": 224.5, "Google Cloud": 26.3, "Other": 32.0},
     "Advertising growth slowed amid macro headwinds."),
    ("Alphabet", "GOOGL", "Technology", "FY2021", 257.6, 76.0, 56.9,
     {"Google advertising": 209.5, "Google Cloud": 19.2, "Other": 28.9},
     "Record profit on strong advertising demand."),

    # ---- Amazon (calendar year) ----
    ("Amazon", "AMZN", "Consumer / Cloud", "FY2023", 574.8, 30.4, None,
     {"North America": 352.8, "International": 131.2, "AWS": 90.8},
     "Return to profitability; AWS remained the primary profit driver."),
    ("Amazon", "AMZN", "Consumer / Cloud", "FY2022", 514.0, -2.7, None,
     {"North America": 315.9, "International": 118.0, "AWS": 80.1},
     "Net loss for the year amid heavy investment and a Rivian markdown."),
    ("Amazon", "AMZN", "Consumer / Cloud", "FY2021", 469.8, 33.4, None,
     {"North America": 279.8, "International": 127.8, "AWS": 62.2},
     "Strong pandemic-era growth across all segments."),

    # ---- NVIDIA (fiscal year ends ~late January) ----
    ("NVIDIA", "NVDA", "Semiconductors", "FY2024 (ended Jan 2024)", 60.9, 29.8, 72.7,
     {"Data Center": 47.5, "Gaming": 10.4, "Professional Visualization": 1.6, "Automotive": 1.1},
     "Data Center revenue more than tripled on AI/accelerated-computing demand."),
    ("NVIDIA", "NVDA", "Semiconductors", "FY2023 (ended Jan 2023)", 27.0, 4.4, 56.9,
     {"Data Center": 15.0, "Gaming": 9.1, "Professional Visualization": 1.5, "Automotive": 0.9},
     "Gaming declined while Data Center grew."),
    ("NVIDIA", "NVDA", "Semiconductors", "FY2022 (ended Jan 2022)", 26.9, 9.8, 64.9,
     {"Gaming": 12.5, "Data Center": 10.6, "Professional Visualization": 2.1, "Automotive": 0.6},
     "Record results led by Gaming and Data Center."),

    # ---- Tesla (calendar year) ----
    ("Tesla", "TSLA", "Automotive / Energy", "FY2023", 96.8, 15.0, 18.2,
     {"Automotive": 82.4, "Energy generation & storage": 6.0, "Services & other": 8.3},
     "Net income boosted by a one-time deferred tax benefit; automotive margins compressed on price cuts."),
    ("Tesla", "TSLA", "Automotive / Energy", "FY2022", 81.5, 12.6, 25.6,
     {"Automotive": 71.5, "Energy generation & storage": 3.9, "Services & other": 6.1},
     "Record deliveries and strong automotive margins."),
    ("Tesla", "TSLA", "Automotive / Energy", "FY2021", 53.8, 5.5, 25.3,
     {"Automotive": 47.2, "Energy generation & storage": 2.8, "Services & other": 3.8},
     "First full year of sustained GAAP profitability."),

    # ---- Meta (calendar year) ----
    ("Meta", "META", "Technology", "FY2023", 134.9, 39.1, 80.8,
     {"Family of Apps": 133.0, "Reality Labs": 1.9},
     "Reality Labs posted an operating loss of about $16.1B; advertising rebounded strongly."),
    ("Meta", "META", "Technology", "FY2022", 116.6, 23.2, 80.0,
     {"Family of Apps": 114.5, "Reality Labs": 2.2},
     "First annual revenue decline; heavy Reality Labs investment."),
    ("Meta", "META", "Technology", "FY2021", 117.9, 39.4, 80.8,
     {"Family of Apps": 115.7, "Reality Labs": 2.3},
     "Record revenue and profit before the 2022 slowdown."),
]


def generate_documents() -> list[dict]:
    docs: list[dict] = []
    for i, (name, ticker, sector, fy, rev, ni, gm, segs, note) in enumerate(_RECORDS):
        ni_str = f"a net loss of ${abs(ni):,.1f}B" if ni < 0 else f"net income of ${ni:,.1f}B"
        gm_str = f", and a gross margin of {gm:.1f}%" if gm is not None else ""
        seg_lines = "; ".join(f"{s} ${v:,.1f}B" for s, v in segs.items())
        text = (
            f"{name} ({ticker}) — {fy} Financial Summary ({sector}).\n"
            f"{name} reported total revenue of ${rev:,.1f}B, with {ni_str}{gm_str}.\n"
            f"Revenue by segment: {seg_lines}.\n"
            f"{note}"
        )
        docs.append({
            "doc_id": f"{ticker}_{fy.split()[0]}",
            "company": name,
            "company_id": ticker,
            "sector": sector,
            "period": fy.split()[0],          # e.g. "FY2023"
            "text": text,
        })
    return docs


if __name__ == "__main__":
    docs = generate_documents()
    print(f"{len(docs)} real-data chunks")
    print(docs[0]["text"])
