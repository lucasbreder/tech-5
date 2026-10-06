"""Orquestração com LangGraph (requisito #8).

receber dados → modelo → anomalia e relato → protocolos → explicação.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.data.preprocessing import FEATURE_LABELS
from src.llm.chains import build_explanation
from src.ml.anomaly import describe_flags, reference_range_flags
from src.ml.explain import feature_importance, local_probability_contributions
from src.ml.registry import ANOMALY_NAME, available_classifiers, feature_frame, load_metadata, load_model
from src.rag.retriever import RetrievedDoc, format_context, retrieve
from src.services.narrative import scan_report


def _top_k() -> int:
    from src.config import settings

    return settings.rag.top_k


def _merge_docs(groups: list[list[RetrievedDoc]], limit: int) -> list[RetrievedDoc]:
    """Garante o melhor trecho de cada consulta e completa até `limit`."""
    picked: list[RetrievedDoc] = []
    seen: set[tuple[str, str]] = set()

    def _add(doc: RetrievedDoc) -> None:
        key = (doc.source, doc.text[:120])
        if key in seen:
            return
        seen.add(key)
        picked.append(doc)

    for docs in groups:
        if docs:
            _add(max(docs, key=lambda doc: doc.score))
    rest = [doc for docs in groups for doc in docs]
    for doc in sorted(rest, key=lambda item: item.score, reverse=True):
        if len(picked) >= limit:
            break
        _add(doc)
    return picked[:limit]


class CaseState(TypedDict, total=False):
    features: dict
    report_text: str
    model_name: str
    query: str
    prediction: str
    probability: float
    probabilities: dict
    model_results: list
    top_features: str
    importance: dict
    local_contributions: dict
    anomaly: dict
    narrative: dict
    rag_sources: list
    rag_excerpts: list
    rag_context: str
    explanation: str
    explanation_mode: str
    anomaly_summary: str


def _predict_one(model_name: str, frame) -> dict:
    model = load_model(model_name)
    pred = str(model.predict(frame)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        classes = [str(c) for c in model.classes_]
        probabilities = {label: float(value) for label, value in zip(classes, proba)}
    return {
        "model": model_name,
        "prediction": pred,
        "probability": float(probabilities.get(pred, 0.0)),
        "probabilities": probabilities,
    }


def _node_ml(state: CaseState) -> dict:
    frame = feature_frame(state["features"])
    chosen = state.get("model_name") or load_metadata().get("best_model") or "random_forest"
    names = available_classifiers() or [chosen]
    results = []
    for name in names:
        try:
            results.append(_predict_one(name, frame))
        except FileNotFoundError:
            continue
    if not results:
        raise FileNotFoundError("Nenhum modelo treinado. Rode `python main.py train`.")

    primary = next((item for item in results if item["model"] == chosen), results[0])
    model = load_model(primary["model"])
    metadata = load_metadata()
    try:
        series = feature_importance(model, list(frame.columns))
        importance = {str(k): float(v) for k, v in series.items()}
        local = local_probability_contributions(
            model,
            frame,
            primary["prediction"],
            metadata.get("feature_baselines") or {},
        )
        local_contributions = {str(k): float(v) for k, v in local.items()}
        top = ", ".join(
            f"{FEATURE_LABELS.get(name, name)}={state['features'][name]:g} "
            f"(efeito local {local[name]:+.2f})"
            for name in local.head(3).index
        ) or "não disponível"
    except Exception as exc:
        logger.warning(f"Sem explicação de variáveis: {exc}")
        importance = {}
        local_contributions = {}
        top = "não disponível"
    return {
        "model_name": primary["model"],
        "prediction": primary["prediction"],
        "probability": primary["probability"],
        "probabilities": primary["probabilities"],
        "model_results": results,
        "importance": importance,
        "local_contributions": local_contributions,
        "top_features": top,
    }


def _node_context(state: CaseState) -> dict:
    features = state["features"]
    flags = reference_range_flags(features)
    anomaly = {
        "available": False,
        "is_outlier": False,
        "score": None,
        "range_flags": describe_flags(flags),
    }
    try:
        detector = load_model(ANOMALY_NAME.replace(".joblib", ""))
        frame = feature_frame(features)
        anomaly["available"] = True
        anomaly["is_outlier"] = int(detector.predict(frame)[0]) == -1
        anomaly["score"] = float(detector.decision_function(frame)[0])
    except FileNotFoundError:
        logger.info("Isolation Forest ainda não treinado.")

    narrative = scan_report(state.get("report_text"))
    return {"anomaly": anomaly, "narrative": narrative}


def _node_retrieve(state: CaseState) -> dict:
    features = state["features"]
    clinical = " ".join(f"{FEATURE_LABELS.get(k, k)} {features[k]:g}" for k in features)
    clinical = f"{clinical} risco {state.get('prediction', '')} pré-natal pressão glicemia"
    report = (state.get("report_text") or "").strip()
    custom = (state.get("query") or "").strip()
    groups = [retrieve(custom or clinical)]
    if report:
        groups.append(retrieve(report))
    if (state.get("narrative") or {}).get("signals"):
        groups.append(retrieve(f"{report} escuta sigilo relato ameaça atendimento"))
    docs = _merge_docs(groups, limit=max(_top_k(), len(groups)))
    excerpts = [
        {
            "source": doc.source,
            "source_url": doc.source_url,
            "text": doc.text.strip(),
            "score": round(doc.score, 3),
        }
        for doc in docs
    ]
    return {
        "rag_sources": sorted({doc.source for doc in docs}),
        "rag_excerpts": excerpts,
        "rag_context": format_context(docs),
    }


def _node_explain(state: CaseState) -> dict:
    anomaly = state.get("anomaly") or {}
    flags = anomaly.get("range_flags") or []
    if anomaly.get("available"):
        combo = (
            "a combinação dos sinais é incomum no conjunto de treino"
            if anomaly.get("is_outlier")
            else "a combinação dos sinais aparece no conjunto de treino"
        )
    else:
        combo = "o detector de combinação incomum não está disponível"
    if flags:
        anomaly_summary = f"{combo}. Fora da faixa de atenção da aplicação: {', '.join(flags)}."
    else:
        anomaly_summary = f"{combo}. Nenhum sinal fora da faixa de atenção cadastrada."

    narrative = state.get("narrative") or {}
    signals = narrative.get("signals") or []
    narrative_signals = narrative.get("note", "—")
    if signals:
        narrative_signals = f"{narrative_signals} Termos: {', '.join(signals)}."

    probabilities = state.get("probabilities") or {}
    prob_txt = ", ".join(f"{k}={v:.2f}" for k, v in probabilities.items()) or "—"
    features = state.get("features") or {}
    patient_summary = "\n".join(
        f"- {FEATURE_LABELS.get(key, key)}: {value:g}" for key, value in features.items()
    )
    explanation, mode = build_explanation(
        patient_summary=patient_summary,
        report_text=state.get("report_text") or "não informado",
        narrative_signals=narrative_signals,
        ml_prediction=state.get("prediction"),
        ml_probability=f"{state.get('probability', 0):.2f}",
        probabilities=prob_txt,
        top_features=state.get("top_features"),
        anomaly_summary=anomaly_summary,
        rag_context=state.get("rag_context"),
        model=state.get("model_name"),
    )
    return {"explanation": explanation, "explanation_mode": mode, "anomaly_summary": anomaly_summary}


def build_graph():
    graph = StateGraph(CaseState)
    graph.add_node("ml", _node_ml)
    graph.add_node("context", _node_context)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("explain", _node_explain)
    graph.set_entry_point("ml")
    graph.add_edge("ml", "context")
    graph.add_edge("context", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


@lru_cache
def get_graph():
    return build_graph()


def analyze_case(
    features: dict,
    report_text: str | None = None,
    model_name: str | None = None,
    query: str | None = None,
) -> dict:
    """Executa a jornada e devolve um dicionário pronto para a tela e a auditoria."""
    meta = load_metadata()
    chosen = model_name or meta.get("best_model") or "random_forest"
    state = get_graph().invoke(
        {
            "features": {k: float(v) for k, v in features.items()},
            "report_text": report_text or "",
            "model_name": chosen,
            "query": query or "",
        }
    )
    return {
        "features": state["features"],
        "report_text": state.get("report_text") or "",
        "prediction": state.get("prediction"),
        "probability": state.get("probability"),
        "probabilities": state.get("probabilities") or {},
        "model": state.get("model_name"),
        "model_results": state.get("model_results") or [],
        "importance": state.get("importance") or {},
        "local_contributions": state.get("local_contributions") or {},
        "top_features": state.get("top_features"),
        "anomaly": state.get("anomaly") or {},
        "narrative": state.get("narrative") or {},
        "rag_sources": state.get("rag_sources") or [],
        "rag_excerpts": state.get("rag_excerpts") or [],
        "explanation": state.get("explanation"),
        "explanation_mode": state.get("explanation_mode"),
        "disclaimer": (
            "Sistema de apoio à decisão. Não diagnostica, não prescreve e não "
            "decide medida de segurança. A avaliação final é do profissional."
        ),
    }
