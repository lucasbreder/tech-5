"""Chunking + indexação dos protocolos de apoio (requisito #7)."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from src.config import settings
from src.rag.embeddings import embed_documents, write_backend
from src.rag.store import reset_collection


def ingest_documents(directory: str | Path | None = None) -> int:
    """Lê .md/.txt de `data/knowledge/`, recria o índice e devolve o nº de chunks."""
    directory = Path(directory or settings.path(settings.knowledge_dir))
    files = [
        file
        for file in sorted(directory.glob("**/*"))
        if file.is_file() and file.suffix.lower() in {".md", ".txt"} and ".chroma" not in file.parts
    ]
    chunks: list[tuple[str, str, str, str]] = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        source_url = _source_url(text)
        for i, chunk in enumerate(_chunk(text)):
            chunks.append((f"{file.name}:{i}", chunk, file.name, source_url))

    if not chunks:
        logger.warning(f"Nenhum .md/.txt em {directory}.")
        return 0

    vectors, backend = embed_documents([chunk for _, chunk, _, _ in chunks])
    write_backend(backend)
    collection = reset_collection()
    collection.upsert(
        ids=[item[0] for item in chunks],
        documents=[item[1] for item in chunks],
        embeddings=vectors,
        metadatas=[
            {"source": item[2], "source_url": item[3], "chunk": i}
            for i, item in enumerate(chunks)
        ],
    )
    logger.info(f"RAG: {len(chunks)} chunks indexados com backend '{backend}'.")
    return len(chunks)


def _source_url(text: str) -> str:
    match = re.search(r"^Fonte oficial:\s*(https?://\S+)", text, flags=re.MULTILINE)
    return match.group(1).rstrip(".,)") if match else ""


def _chunk(text: str) -> list[str]:
    size, overlap = settings.rag.chunk_size, settings.rag.chunk_overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += max(size - overlap, 1)
    return chunks
