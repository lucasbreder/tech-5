"""Relatório Markdown de cada análise — base da auditoria legível."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings

DISCLAIMER = (
    "> **Guardiã AI** é um sistema de **apoio à decisão**. Não realiza "
    "diagnóstico definitivo, não prescreve e não decide medida de segurança. "
    "A avaliação final cabe ao profissional responsável pelo atendimento."
)


def render_markdown(analysis: dict[str, Any]) -> str:
    sources = analysis.get("rag_sources") or []
    srcs = ", ".join(sources) or "—"
    narrative = analysis.get("narrative") or {}
    anomaly = analysis.get("anomaly") or {}
    flags = ", ".join(anomaly.get("range_flags") or []) or "nenhum"
    signals = ", ".join(narrative.get("signals") or []) or "nenhum"
    others = analysis.get("model_results") or []
    lines = [
        f"# Análise de atendimento — {datetime.now():%d/%m/%Y %H:%M}",
        "",
        DISCLAIMER,
        "",
        "## Dados de entrada",
        "```json",
        json.dumps(analysis.get("features", {}), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Relato",
        analysis.get("report_text") or "_não informado_",
        "",
        f"- Sinais da lista de atenção: {signals}",
        f"- Nota: {narrative.get('note', '—')}",
        "",
        "## Resultado do modelo",
        f"- **Modelo usado na nota:** {analysis.get('model')}",
        f"- **Predição de risco:** {analysis.get('prediction')}",
        f"- **Probabilidade da classe prevista:** {float(analysis.get('probability') or 0):.2f}",
        f"- **Distribuição:** {analysis.get('probabilities')}",
        f"- **Efeitos locais vs. mediana do treino:** {analysis.get('local_contributions') or {}}",
        "",
        "### Comparação neste atendimento",
    ]
    for item in others:
        lines.append(
            f"- {item.get('model')}: {item.get('prediction')} ({float(item.get('probability') or 0):.2f})"
        )
    lines.extend(
        [
            "",
            "## Faixa de atenção e combinação incomum",
            f"- Fora da faixa: {flags}",
            f"- Combinação incomum no treino: {'sim' if anomaly.get('is_outlier') else 'não'}",
            "",
            "## Fontes consultadas (RAG)",
            f"- {srcs}",
            "",
            "## Explicação",
            f"_Modo: {analysis.get('explanation_mode', '—')}._",
            "",
            analysis.get("explanation", "_sem resposta_"),
            "",
        ]
    )
    return "\n".join(lines)


def save_report(analysis: dict[str, Any], name: str | None = None) -> Path:
    out_dir = settings.path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"analise_{datetime.now():%Y%m%d_%H%M%S}"
    (out_dir / f"{name}.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    md_path = out_dir / f"{name}.md"
    md_path.write_text(render_markdown(analysis), encoding="utf-8")
    return md_path
