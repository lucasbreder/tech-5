"""Interface Streamlit da jornada de atendimento (requisito #9)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.preprocessing import FEATURE_LABELS
from src.graph.workflow import analyze_case
from src.ml.evaluate import METRIC_RATIONALE
from src.ml.registry import available_classifiers, load_metadata, load_metrics
from src.services.audit_log import log_analysis, read_logs
from src.utils.report_generator import save_report

st.set_page_config(page_title="Guardiã AI", page_icon="🛡️", layout="wide")

FIELDS = {
    "systolic_bp": {"label": "Pressão sistólica (mmHg)", "min": 70, "max": 220, "step": 1.0, "value": 148.0},
    "diastolic_bp": {"label": "Pressão diastólica (mmHg)", "min": 40, "max": 140, "step": 1.0, "value": 96.0},
    "blood_sugar": {"label": "Glicemia (mg/dL)", "min": 40, "max": 400, "step": 1.0, "value": 168.0},
    "body_temp": {"label": "Temperatura (°C)", "min": 35.0, "max": 41.0, "step": 0.1, "value": 36.8},
    "age": {"label": "Idade (anos)", "min": 12, "max": 70, "step": 1.0, "value": 34.0},
    "heart_rate": {"label": "Frequência cardíaca (bpm)", "min": 40, "max": 180, "step": 1.0, "value": 92.0},
}

PRESETS = {
    "Sinais estáveis": {
        "features": {
            "age": 26,
            "systolic_bp": 110,
            "diastolic_bp": 70,
            "blood_sugar": 95,
            "body_temp": 36.6,
            "heart_rate": 78,
        },
        "report": "Gestante refere bem-estar, sem queixa nova. Pré-natal em acompanhamento.",
    },
    "Pressão e glicemia em atenção": {
        "features": {
            "age": 34,
            "systolic_bp": 148,
            "diastolic_bp": 96,
            "blood_sugar": 168,
            "body_temp": 36.8,
            "heart_rate": 92,
        },
        "report": "Queixa de dor de cabeça e inchaço nas mãos nesta semana.",
    },
    "Alterações e relato sensível": {
        "features": {
            "age": 22,
            "systolic_bp": 150,
            "diastolic_bp": 100,
            "blood_sugar": 180,
            "body_temp": 38.2,
            "heart_rate": 110,
        },
        "report": (
            "Disse que tem medo de ir para casa porque o parceiro a ameaçou e já houve agressão. "
            "Pediu para ninguém ligar para ele."
        ),
    },
}

DISCLAIMER = (
    "Apoio ao profissional. Esta aplicação não faz diagnóstico, não prescreve "
    "e não toma decisão de segurança. A avaliação final é sempre humana."
)


def _init_state() -> None:
    base = PRESETS["Pressão e glicemia em atenção"]
    for key, spec in FIELDS.items():
        st.session_state.setdefault(key, float(base["features"][key]))
    st.session_state.setdefault("report_text", base["report"])


def _apply_preset(name: str) -> None:
    preset = PRESETS[name]
    for key, value in preset["features"].items():
        st.session_state[key] = float(value)
    st.session_state["report_text"] = preset["report"]


_init_state()

st.title("Guardiã AI")
st.caption("Triagem de apoio no atendimento à gestante, com leitura do relato.")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Modelos")
    meta = load_metadata()
    trained = available_classifiers()
    if trained:
        default = meta.get("best_model") if meta.get("best_model") in trained else trained[0]
        model_name = st.selectbox(
            "Modelo da nota",
            trained,
            index=trained.index(default),
            help="A nota usa este modelo. Os outros também rodam, para comparação.",
        )
    else:
        model_name = None
        st.info("Nenhum modelo treinado. No terminal: `python main.py train`.")
    st.markdown(METRIC_RATIONALE)
    metrics = load_metrics()
    if metrics is not None:
        st.dataframe(metrics.round(3))

st.subheader("Atendimento")
st.session_state.setdefault("preset", "Pressão e glicemia em atenção")


def _on_preset() -> None:
    _apply_preset(st.session_state["preset"])


st.selectbox(
    "Exemplo de atendimento",
    list(PRESETS),
    key="preset",
    on_change=_on_preset,
)

left, right = st.columns(2)
features: dict[str, float] = {}
items = list(FIELDS.items())
for index, (key, spec) in enumerate(items):
    column = left if index % 2 == 0 else right
    features[key] = column.number_input(
        spec["label"],
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        step=float(spec["step"]),
        key=key,
    )
report_text = st.text_area(
    "Relato do atendimento (opcional)",
    key="report_text",
    height=120,
    help="Texto informado no atendimento. A aplicação organiza termos de atenção; não conclui o fato.",
)

analyze = st.button("Analisar atendimento", type="primary", disabled=not trained)

if analyze:
    with st.spinner(f"Modelo, protocolos e nota local ({settings.llm.ollama_model})..."):
        try:
            result = analyze_case(features, report_text=report_text or None, model_name=model_name)
        except FileNotFoundError as exc:
            st.error(f"Modelos ainda não treinados. Rode `python main.py train`.\n\n{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Falha na análise do atendimento: {exc}")
            st.error(
                "Não foi possível concluir a análise. Confira o terminal da "
                "aplicação e tente novamente."
            )
            st.stop()
    log_analysis(result)
    path = save_report(result)
    st.session_state["last_result"] = result
    st.session_state["last_report"] = str(path)

result = st.session_state.get("last_result")
if result:
    st.subheader("Resultado para o profissional")
    c1, c2, c3 = st.columns(3)
    c1.metric("Risco indicado", str(result["prediction"]))
    c2.metric("Probabilidade da classe", f"{float(result['probability']):.0%}")
    c3.metric("Modelo da nota", str(result["model"]))
    st.caption(
        "A probabilidade é a saída interna do classificador e não foi calibrada "
        "como probabilidade clínica individual."
    )

    probs = result.get("probabilities") or {}
    if probs:
        st.caption("Distribuição de probabilidade do modelo escolhido")
        st.bar_chart(pd.Series(probs, name="probabilidade"))

    others = pd.DataFrame(result.get("model_results") or [])
    if not others.empty:
        st.markdown("**Os três classificadores neste atendimento**")
        view = others[["model", "prediction", "probability"]].copy()
        view["probability"] = view["probability"].map(lambda v: f"{float(v):.0%}")
        st.dataframe(view, hide_index=True, width="stretch")

    importance = result.get("importance") or {}
    if importance:
        st.markdown("**Peso global das variáveis no modelo da nota**")
        series = pd.Series(importance)
        series.index = [FEATURE_LABELS.get(i, i) for i in series.index]
        st.bar_chart(series.sort_values(ascending=False))

    local_contributions = result.get("local_contributions") or {}
    if local_contributions:
        st.markdown("**Efeito local neste atendimento**")
        local_series = pd.Series(local_contributions)
        local_series.index = [FEATURE_LABELS.get(i, i) for i in local_series.index]
        st.bar_chart(local_series)
        st.caption(
            "Variação na probabilidade da classe indicada ao trocar uma variável "
            "pela mediana do treino. Positivo aumenta a classe; negativo reduz."
        )

    anomaly = result.get("anomaly") or {}
    narrative = result.get("narrative") or {}
    a, b = st.columns(2)
    with a:
        st.markdown("**Faixa de atenção**")
        flags = anomaly.get("range_flags") or []
        st.write(", ".join(flags) if flags else "Nenhum sinal fora da faixa cadastrada.")
        if anomaly.get("available"):
            st.write(
                "Combinação incomum no treino: "
                + ("sim." if anomaly.get("is_outlier") else "não.")
            )
    with b:
        st.markdown("**Relato**")
        st.write(narrative.get("note") or "—")
        signals = narrative.get("signals") or []
        if signals:
            st.write("Termos destacados: " + ", ".join(signals) + ".")

    mode = result.get("explanation_mode")
    st.markdown("**Nota de apoio**")
    if mode == "ollama":
        st.caption(
            f"Texto gerado na sua máquina pelo modelo {settings.llm.ollama_model} (Ollama), sem chave de API."
        )
    elif mode == "local":
        st.caption("Texto montado na aplicação porque o modelo local não respondeu.")
    else:
        st.caption("Texto gerado pela LLM a partir do modelo e dos protocolos recuperados.")
    st.markdown(result.get("explanation") or "_sem nota_")

    st.markdown("**Protocolos consultados**")
    excerpts = result.get("rag_excerpts") or []
    if not excerpts:
        st.info("Nenhum protocolo indexado. Rode `python main.py rag-ingest`.")
    for excerpt in excerpts:
        with st.expander(f"{excerpt['source']}  ·  relevância {excerpt['score']:.2f}"):
            if excerpt.get("source_url"):
                st.markdown(f"[Abrir documento oficial]({excerpt['source_url']})")
            st.write(excerpt["text"])

    report_path = st.session_state.get("last_report")
    if report_path:
        st.download_button(
            "Baixar relatório (Markdown)",
            Path(report_path).read_text(encoding="utf-8"),
            file_name=Path(report_path).name,
        )

with st.expander("Registro das análises (auditoria)"):
    logs = read_logs()
    if not logs:
        st.caption("Ainda não há análises registradas neste ambiente.")
    else:
        recent = list(reversed(logs[-8:]))
        rows = [
            {
                "quando": item.get("timestamp", ""),
                "risco": item.get("prediction"),
                "modelo": item.get("model"),
                "fontes": ", ".join(item.get("rag_sources") or []),
            }
            for item in recent
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
