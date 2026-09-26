"""Cliente LLM / embeddings (requisito #6). Suporta OpenAI e Azure OpenAI."""

from __future__ import annotations

from src.config import settings


def get_chat_model(temperature: float = 0.0):
    """Retorna um BaseChatModel do LangChain conforme o provedor configurado."""
    llm = settings.llm
    if not llm.is_configured:
        raise RuntimeError("Nenhuma chave de LLM configurada. Defina OPENAI_API_KEY no .env.")

    if llm.provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=llm.azure_openai_deployment,
            api_version=llm.azure_openai_api_version,
            temperature=temperature,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=llm.openai_model, temperature=temperature, api_key=llm.openai_api_key)


def get_embeddings():
    """Embeddings para o RAG (OpenAI / Azure OpenAI)."""
    llm = settings.llm
    if not llm.is_configured:
        raise RuntimeError("Nenhuma chave de LLM configurada para embeddings.")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=llm.embeddings_model, api_key=llm.openai_api_key)
