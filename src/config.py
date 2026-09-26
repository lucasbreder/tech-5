"""Configuração centralizada da Guardiã AI (Fase 5).

Padrão herdado do tech-4: Pydantic Settings + .env + caminhos resolvidos
a partir da raiz do projeto. Adaptado de Azure Cognitive Services para
LLM + RAG (OpenAI / Azure OpenAI + Chroma).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(BaseSettings):
    """Provedor de LLM (OpenAI ou Azure OpenAI)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: Literal["openai", "azure"] = Field(default="openai")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: str = Field(default="gpt-4o-mini", alias="AZURE_OPENAI_DEPLOYMENT")

    embeddings_model: str = Field(default="text-embedding-3-small", alias="EMBEDDINGS_MODEL")

    @property
    def is_configured(self) -> bool:
        if self.provider == "azure":
            return bool(self.azure_openai_api_key and self.azure_openai_endpoint)
        return bool(self.openai_api_key)


class RAGConfig(BaseSettings):
    """Base de conhecimento / vector store."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    persist_dir: str = Field(default="data/knowledge/.chroma", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="protocolos")
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)
    top_k: int = Field(default=4)


class MLConfig(BaseSettings):
    """Modelos de Machine Learning tabular."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    random_state: int = Field(default=42, alias="RANDOM_STATE")
    test_size: float = Field(default=0.2, alias="TEST_SIZE")
    target_column: str = Field(default="risk_level", description="Coluna alvo da classificação")


class Settings(BaseSettings):
    """Configuração global agrupada."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_raw_dir: str = Field(default="data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field(default="data/processed", alias="DATA_PROCESSED_DIR")
    knowledge_dir: str = Field(default="data/knowledge", alias="KNOWLEDGE_DIR")
    models_dir: str = Field(default="models", alias="MODELS_DIR")
    audit_dir: str = Field(default="data/reports/audits", alias="AUDIT_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu", alias="DEVICE")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    def path(self, relative: str) -> Path:
        return (PROJECT_ROOT / relative).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
