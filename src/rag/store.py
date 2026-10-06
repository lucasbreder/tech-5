"""Vector store (Chroma) para a base de conhecimento (requisito #7)."""

from __future__ import annotations

import chromadb

from src.config import settings


def get_client() -> chromadb.ClientAPI:
    path = settings.path(settings.rag.persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.rag.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(settings.rag.collection_name)
    except Exception:
        pass
    return get_collection()
