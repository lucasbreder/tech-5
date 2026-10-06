"""Embeddings locais do RAG, sem chave e sem envio de texto a terceiros."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer

from src.config import settings

BACKEND_FILE = ".embed_backend"
LOCAL_BACKEND = "hashing"
N_FEATURES = 512

_PT_STOP = [
    "a", "o", "os", "as", "um", "uma", "de", "da", "do", "das", "dos",
    "e", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem",
    "que", "se", "ou", "ao", "à", "seu", "sua", "não", "nao",
]


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words=_PT_STOP,
        lowercase=True,
    )


def local_embed(texts: list[str]) -> list[list[float]]:
    return _vectorizer().transform(texts).toarray().tolist()


def backend_path() -> Path:
    return settings.path(settings.knowledge_dir) / BACKEND_FILE


def read_backend() -> str:
    path = backend_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or LOCAL_BACKEND
    return LOCAL_BACKEND


def write_backend(name: str) -> None:
    path = backend_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")


def embed_documents(texts: list[str]) -> tuple[list[list[float]], str]:
    return local_embed(texts), LOCAL_BACKEND


def embed_query(text: str) -> list[float]:
    backend = read_backend()
    if backend != LOCAL_BACKEND:
        raise RuntimeError(
            f"Índice RAG usa backend '{backend}', mas a aplicação usa '{LOCAL_BACKEND}'. "
            "Rode `python main.py rag-ingest` para reconstruir o índice local."
        )
    return local_embed([text])[0]
