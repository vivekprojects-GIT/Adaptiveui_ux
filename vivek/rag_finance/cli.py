"""CLI: ask a finance question, see the grounded answer, the evidence-bound widget,
and the verification proving every visual number matches the source DB.

Usage:
  python -m rag_finance.cli "How has Lumina Systems revenue trended over 2023-2024?"
"""

from __future__ import annotations

import json
import sys

from .pipeline import answer


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "Show Lumina Systems quarterly revenue trend for 2023-2024."
    res = answer(query)

    print("=" * 70)
    print("QUERY:", res["query"])
    print("=" * 70)
    print("\nRETRIEVED DOCS:", ", ".join(res["retrieved_docs"]))
    print("EVIDENCE FACTS AVAILABLE:", res["evidence_facts"])
    print("\n--- GROUNDED ANSWER ---\n")
    print(res["response"])
    print("\n--- EVIDENCE-BOUND WIDGET ---\n")
    print(json.dumps(res["widget"], ensure_ascii=False, indent=2))

    v = res["verification"]
    print("\n--- VERIFICATION (visual values vs source DB) ---")
    print(f"points: {v['total_points']}  passed: {v['passed']}  failed: {v['failed']}  "
          f"GROUNDED: {'YES' if v['grounded'] else 'NO'}")
    for c in v["checks"]:
        mark = "OK " if c["ok"] else "XX "
        print(f"  {mark} {c['block']:8} {str(c['label'])[:28]:28} value={c['value']} ref={c['source_ref']}  {c['detail']}")


if __name__ == "__main__":
    main()
