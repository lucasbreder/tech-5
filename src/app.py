"""Interface Streamlit da Guardiã AI (requisito #9).

Jornada completa: entrada de dados de atendimento -> ML -> RAG -> LLM -> resultado.
Rodar: streamlit run src/app.py  (ou `python main.py demo`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.workflow import analyze_case  # noqa: E402
from src.services.audit_log import log_analysis  # noqa: E402
from src.utils.report_generator import save_report  # noqa: E402

st.set_page_config(page_title="Guardiã AI", page_icon="🛡️", layout="wide")

# Campos de exemplo (triagem de risco materno). Ajustar ao dataset escolhido.
DEFAULT_FIELDS = {
    "age": 28.0,
    "systolic_bp": 138.0,
    "diastolic_bp": 88.0,
    "blood_sugar": 145.0,
    "body_temp": 37.4,
    "heart_rate": 96.0,
}

st.title("🛡️ Guardiã AI — Apoio à Saúde e Segurança da Mulher")
st.caption(
    "Sistema de **apoio à decisão**. Não faz diagnóstico definitivo; a "
    "avaliação final é sempre do profissional responsável."
)

with st.form("atendimento"):
    cols = st.columns(2)
    features: dict[str, float] = {}
    for i, (field, default) in enumerate(DEFAULT_FIELDS.items()):
        features[field] = cols[i % 2].number_input(field.replace("_", " ").title(), value=default)
    report_text = st.text_area("Relato do atendimento (opcional — interpretado pela LLM)")
    submitted = st.form_submit_button("Analisar caso")

if submitted:
    with st.spinner("Executando ML → RAG → LLM..."):
        try:
            result = analyze_case(features, report_text=report_text or None)
        except FileNotFoundError as exc:
            st.error(f"Modelos ainda não treinados. Rode `python main.py train`.\n\n{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha na análise: {exc}")
            st.stop()

    st.subheader("Resultado da análise")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predição de risco", result["prediction"])
    c2.metric("Confiança", f"{result['probability']:.2f}")
    c3.metric("Modelo", result["model"])

    st.markdown("### Explicação (LLM)")
    st.markdown(result["explanation"])

    if result.get("rag_sources"):
        st.markdown("### Fontes consultadas (RAG)")
        for s in result["rag_sources"]:
            st.write(f"- 📄 {s}")

    log_analysis(result)
    path = save_report(result)
    st.download_button("Baixar relatório (Markdown)", path.read_text(), file_name=path.name)
