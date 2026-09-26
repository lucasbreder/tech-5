"""Retriever: busca documentos relevantes e preserva a fonte (requisito #7)."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.ingest import _embed
from src.rag.store import get_collection


@dataclass
class RetrievedDoc:
    text: str
    source: str
    score: float


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """Busca por similaridade de cosseno no Chroma e retorna chunks + fonte."""
    top_k = top_k or settings_top_k()
    collection = get_collection()
    if collection.count() == 0:
        return []
    res = collection.query(query_embeddings=_embed([query]), n_results=min(top_k, collection.count()))
    docs = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        docs.append(RetrievedDoc(text=text, source=meta.get("source", "desconhecido"), score=1 - dist))
    return docs


def settings_top_k() -> int:
    from src.config import settings

    return settings.rag.top_k


def format_context(docs: list[RetrievedDoc]) -> str:
    """Monta o bloco de contexto p/ o prompt, citando as fontes."""
    if not docs:
        return "(nenhum documento encontrado na base de conhecimento)"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] (fonte: {d.source})\n{d.text}")
    return "\n\n".join(parts)
