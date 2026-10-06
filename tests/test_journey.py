"""Jornada completa em modo local (sem chave de LLM)."""

from __future__ import annotations

from src.graph.workflow import analyze_case
from src.ml.train import train_all
from src.ml.registry import load_metadata
from src.rag.ingest import ingest_documents
from src.rag.retriever import retrieve


def test_train_retrieve_and_explain_locally(monkeypatch):
    # Teste determinístico: a suíte não depende de um processo Ollama externo.
    from src.config import settings

    monkeypatch.setattr(settings.llm, "provider", "openai")
    monkeypatch.setattr(settings.llm, "openai_api_key", "")

    table = train_all()
    assert "recall_alto" in table.columns
    assert table.index[0] == load_metadata()["best_model"]

    indexed = ingest_documents()
    assert indexed >= 2
    docs = retrieve("pressão arterial glicemia temperatura pré-natal")
    assert docs
    assert any("sinais_atencao_prenatal.md" in doc.source for doc in docs)

    result = analyze_case(
        {
            "age": 34,
            "systolic_bp": 148,
            "diastolic_bp": 96,
            "blood_sugar": 168,
            "body_temp": 36.8,
            "heart_rate": 92,
        },
        report_text="Tenho medo. Ele ameaçou e pediu para ninguém ligar.",
    )
    assert result["prediction"] in {"baixo", "moderado", "alto"}
    assert result["explanation_mode"] == "local"
    assert result["explanation"].strip()
    assert "Limite obrigatório" in result["explanation"]
    assert "escuta_relato_atendimento.md" in result["rag_sources"]
    assert all(item["source_url"].startswith("https://") for item in result["rag_excerpts"])
    assert result["narrative"]["signals"]
    assert "não diagnostica" in result["disclaimer"].lower()
    assert result["model_results"]
    assert result["local_contributions"]
