"""Retriever: busca protocolos e devolve a fonte (requisito #7)."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.embeddings import embed_query
from src.rag.store import get_collection


@dataclass
class RetrievedDoc:
    text: str
    source: str
    score: float
    source_url: str = ""


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """Busca por similaridade de cosseno e devolve trechos com a fonte."""
    top_k = top_k or _top_k()
    collection = get_collection()
    count = collection.count()
    if count == 0 or not query.strip():
        return []
    res = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, count),
    )
    docs = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        docs.append(
            RetrievedDoc(
                text=text,
                source=str(meta.get("source", "desconhecido")),
                score=float(1 - dist),
                source_url=str(meta.get("source_url", "")),
            )
        )
    return docs


def _top_k() -> int:
    from src.config import settings

    return settings.rag.top_k


def format_context(docs: list[RetrievedDoc]) -> str:
    """Monta o bloco de contexto do prompt, com a fonte de cada trecho."""
    if not docs:
        return "(nenhum documento encontrado na base de conhecimento)"
    parts = []
    for i, doc in enumerate(docs, 1):
        citation = doc.source
        if doc.source_url:
            citation += f" — {doc.source_url}"
        parts.append(f"[{i}] (fonte: {citation}, relevância {doc.score:.2f})\n{doc.text}")
    return "\n\n".join(parts)
