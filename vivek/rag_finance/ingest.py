"""Load the finance corpus into a persistent ChromaDB collection as plain text
chunks. No evidence index — the new flow grounds the widget in the synthesized
answer, not a structured evidence packet."""

from __future__ import annotations

from pathlib import Path

import chromadb

from .real_data import generate_documents

DIR = Path(__file__).resolve().parent
DB_PATH = DIR / "chroma_db"
COLLECTION = "finance_filings"


def ingest() -> dict:
    docs = generate_documents()

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    col.add(
        documents=[d["text"] for d in docs],
        ids=[d["doc_id"] for d in docs],
        metadatas=[
            {"company": d["company"], "company_id": d["company_id"], "sector": d["sector"], "period": d["period"]}
            for d in docs
        ],
    )
    return {"documents": len(docs), "collection": COLLECTION, "db_path": str(DB_PATH)}


if __name__ == "__main__":
    stats = ingest()
    print(f"Ingested {stats['documents']} real-data chunks -> '{stats['collection']}'")
    print(f"DB: {stats['db_path']}")
