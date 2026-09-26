"""Orquestração do fluxo com LangGraph (requisito #8).

Fluxo simples do edital:
    receber dados -> executar modelo -> consultar documentos -> gerar explicação
    -> retornar resultado.
"""

from __future__ import annotations

import joblib
import pandas as pd
from loguru import logger

from src.config import settings
from src.llm.chains import build_explanation
from src.ml.explain import feature_importance
from src.rag.retriever import format_context, retrieve


def _run_model(features: dict, model_name: str) -> tuple[str, float]:
    model = joblib.load(settings.path(f"{settings.models_dir}/{model_name}.joblib"))
    X = pd.DataFrame([features])
    pred = str(model.predict(X)[0])
    prob = 1.0
    if hasattr(model, "predict_proba"):
        prob = float(max(model.predict_proba(X)[0]))
    return pred, prob


def _top_features(features: dict, model_name: str, k: int = 3) -> str:
    try:
        model = joblib.load(settings.path(f"{settings.models_dir}/{model_name}.joblib"))
        imp = feature_importance(model, list(features.keys()))
        return ", ".join(f"{name}={features[name]:g} (imp {imp[name]:.2f})" for name in imp.head(k).index)
    except Exception as exc:
        logger.warning(f"Sem feature importance: {exc}")
        return "não disponível"


def analyze_case(
    features: dict,
    report_text: str | None = None,
    model_name: str = "random_forest",
    query: str | None = None,
) -> dict:
    """Executa a jornada completa e retorna dict explicável/auditável."""
    prediction, probability = _run_model(features, model_name)

    q = query or " ".join(f"{k}: {v}" for k, v in features.items())
    docs = retrieve(q)
    rag_context = format_context(docs)
    sources = sorted({d.source for d in docs})

    explanation = build_explanation(
        patient_summary=str(features),
        ml_prediction=prediction,
        ml_probability=f"{probability:.2f}",
        top_features=_top_features(features, model_name),
        rag_context=rag_context,
    )

    return {
        "features": features,
        "prediction": prediction,
        "probability": probability,
        "model": model_name,
        "rag_sources": sources,
        "explanation": explanation,
    }
