"""Step 1 — synthesize a grounded answer from retrieved filings ONLY."""

from __future__ import annotations

from .llm import complete

_SYS = (
    "You are a financial analyst assistant. Answer the question USING ONLY the provided "
    "context (retrieved company filings). Quote figures exactly as they appear. If the "
    "context does not contain the answer, say so plainly. NEVER invent numbers, companies, "
    "periods, or facts that are not in the context."
)


def synthesize(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    user = (
        f"CONTEXT (retrieved filings):\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer concisely (3-6 sentences) using only the context above. "
        "Reference specific companies, periods, and figures where relevant."
    )
    return complete(_SYS, user, max_tokens=600)
