"""Vector store (Chroma) para a base de conhecimento (requisito #7)."""

from __future__ import annotations

import chromadb

from src.config import settings


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.path(settings.rag.persist_dir)))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.rag.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
