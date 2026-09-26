"""Cadeias LangChain (requisito #6 + #8) — gerar explicação a partir do contexto."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.llm.client import get_chat_model
from src.llm.prompts import EXPLAIN_TEMPLATE, SYSTEM_PROMPT


def explain_chain():
    """Runnable: prompt -> LLM -> resposta. Usa contexto RAG e resultado ML."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", EXPLAIN_TEMPLATE)]
    )
    return prompt | get_chat_model(temperature=0.0)


def build_explanation(
    patient_summary: str,
    ml_prediction: str,
    ml_probability: str,
    top_features: str,
    rag_context: str,
) -> str:
    """Atalho prático para chamar a cadeia de explicação."""
    chain = explain_chain()
    result = chain.invoke(
        {
            "patient_summary": patient_summary,
            "ml_prediction": ml_prediction,
            "ml_probability": ml_probability,
            "top_features": top_features,
            "rag_context": rag_context,
        }
    )
    return result.content
