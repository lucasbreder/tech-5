"""Cadeia de explicação (LangChain) com texto local quando não há chave."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config import settings
from src.llm.prompts import EXPLAIN_TEMPLATE, SYSTEM_PROMPT

SAFETY_FOOTER = (
    "\n\n---\n**Limite obrigatório:** esta análise apoia a revisão profissional; "
    "não é diagnóstico, prescrição nem decisão automática de segurança."
)


def explain_chain():
    """Runnable prompt → LLM. Exige chave configurada."""
    from src.llm.client import get_chat_model

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", EXPLAIN_TEMPLATE)]
    )
    return prompt | get_chat_model(temperature=0.0)


def build_explanation(**fields: str) -> tuple[str, str]:
    """Devolve (texto, modo). Modo é ``llm`` ou ``local``."""
    payload = {key: str(value) for key, value in fields.items()}
    prompt_keys = (
        "patient_summary",
        "report_text",
        "narrative_signals",
        "ml_prediction",
        "ml_probability",
        "probabilities",
        "top_features",
        "anomaly_summary",
        "rag_context",
    )
    if settings.llm.is_configured:
        mode = "ollama" if settings.llm.provider == "ollama" else "llm"
        try:
            result = explain_chain().invoke({key: payload.get(key, "") for key in prompt_keys})
            return _with_safety_footer(_as_text(result.content)), mode
        except Exception as exc:
            logger.warning(f"LLM indisponível; usando nota determinística: {exc}")
            local = _local_explanation(payload)
            hint = ""
            if settings.llm.provider == "ollama":
                hint = (
                    f" Confira se o Ollama está no ar (`ollama serve`) e se o modelo "
                    f"`{settings.llm.ollama_model}` existe (`ollama list`)."
                )
            message = (
                local
                + f"\n\n_O modelo local não respondeu.{hint} "
                "O texto acima foi montado pela aplicação._"
            )
            return _with_safety_footer(message), "local"
    return _with_safety_footer(_local_explanation(payload)), "local"


def _with_safety_footer(text: str) -> str:
    """Acrescenta o limite fora da geração, mesmo se o modelo ignorar o prompt."""
    return text.strip() + SAFETY_FOOTER


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(part for part in parts if part)
    return str(content)


def _local_explanation(fields: dict[str, str]) -> str:
    """Mesma estrutura do prompt, sem inventar conduta que o protocolo não trouxe."""
    return "\n".join(
        [
            "Nota de apoio montada localmente (nenhuma LLM foi chamada nesta análise).",
            "",
            "1. Resumo do atendimento",
            fields.get("patient_summary", "—"),
            f"Relato: {fields.get('report_text') or 'não informado.'}",
            f"Leitura do relato: {fields.get('narrative_signals', '—')}",
            "",
            "2. Leitura do modelo",
            (
                f"O classificador `{fields.get('model', 'modelo')}` indicou risco "
                f"**{fields.get('ml_prediction', '—')}** "
                f"(probabilidade da classe prevista: {fields.get('ml_probability', '—')}). "
                f"Distribuição: {fields.get('probabilities', '—')}. "
                f"Efeitos locais em relação à mediana do treino: "
                f"{fields.get('top_features', '—')}. "
                "Se a classe alta for um falso negativo, o profissional deixa de revisar a tempo "
                "um atendimento que pedia atenção. Se for um falso alarme, a equipe gasta tempo "
                "com um caso que os sinais não sustentam. A conferência é humana."
            ),
            "",
            "3. Pontos para o profissional revisar",
            fields.get("anomaly_summary", "—"),
            "Trechos recuperados da base (a fonte está no cabeçalho de cada bloco):",
            _clip(fields.get("rag_context", "—"), 900),
            "",
            "4. Limitações",
            (
                "Esta nota não é diagnóstico, não prescreve e não decide medida de segurança. "
                "O modelo foi treinado em um conjunto público de risco materno, com poucas "
                "variáveis, e não conhece exame, história obstétrica nem contexto do relato. "
                "Termos destacados no texto não confirmam violência."
            ),
        ]
    )
