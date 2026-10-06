"""Cliente LLM / embeddings (requisito #6).

O padrão fala com o Ollama em 127.0.0.1. A biblioteca pede um api_key nesse
cliente compatível com a API da OpenAI; o valor não sai da máquina.
"""

from __future__ import annotations

from src.config import settings


def get_chat_model(temperature: float = 0.0):
    """Retorna um BaseChatModel do LangChain conforme o provedor configurado."""
    llm = settings.llm
    if llm.provider == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=llm.ollama_model,
            base_url=f"{llm.ollama_base_url.rstrip('/')}/v1",
            api_key="ollama",
            temperature=temperature,
            max_tokens=700,
        )

    if not llm.has_remote_api:
        raise RuntimeError(
            "Provedor remoto sem chave. Use LLM_PROVIDER=ollama ou defina OPENAI_API_KEY."
        )

    if llm.provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=llm.azure_openai_deployment,
            azure_endpoint=llm.azure_openai_endpoint,
            api_key=llm.azure_openai_api_key,
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

    if llm.provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=llm.embeddings_model,
            azure_endpoint=llm.azure_openai_endpoint,
            api_key=llm.azure_openai_api_key,
            api_version=llm.azure_openai_api_version,
        )
    return OpenAIEmbeddings(model=llm.embeddings_model, api_key=llm.openai_api_key)
