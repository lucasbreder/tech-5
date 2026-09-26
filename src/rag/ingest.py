"""Chunking + embeddings de protocolos/cartilhas públicas (requisito #7).

Coloque PDFs/Markdown de protocolos (ex.: cartilhas SUS de pré-natal ou de
atenção à mulher em situação de violência) em `data/knowledge/` e rode
`python main.py rag-ingest`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from src.config import settings
from src.rag.store import get_collection


def _embed(texts: list[str]) -> list[list[float]]:
    """Gera embeddings via LangChain (OpenAI/Azure), com fallback determinístico."""
    try:
        from src.llm.client import get_embeddings

        return get_embeddings().embed_documents(texts)
    except Exception as exc:  # sem chave de API → permite desenvolver o pipeline offline
        logger.warning(f"Embeddings de API indisponíveis ({exc}); usando hash local (apenas dev).")
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:64]] for t in texts]


def ingest_documents(directory: str | Path | None = None) -> int:
    """Lê .md/.txt de `data/knowledge/`, chunka e indexa. Retorna nº de chunks."""
    directory = Path(directory or settings.path(settings.knowledge_dir))
    collection = get_collection()
    total = 0
    for file in sorted(directory.glob("**/*")):
        if file.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(_chunk(text)):
            doc_id = f"{file.name}:{i}"
            collection.upsert(
                ids=[doc_id],
                documents=[chunk],
                embeddings=_embed([chunk]),
                metadatas=[{"source": file.name, "chunk": i}],
            )
            total += 1
    logger.info(f"RAG: {total} chunks indexados em '{settings.rag.collection_name}'.")
    return total


def _chunk(text: str) -> list[str]:
    size, overlap = settings.rag.chunk_size, settings.rag.chunk_overlap
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + size]
        if piece.strip():
            chunks.append(piece)
        start += size - overlap
    return chunks
