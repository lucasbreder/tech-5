"""Geração de relatórios (JSON + Markdown) — reuso do padrão tech-4.

A saída Markdown é a base para o relatório técnico em PDF (requisito #13):
    md2pdf relatorio.md -o relatorio.pdf
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings

DISCLAIMER = (
    "> ⚠️ **Guardiã AI** é um sistema de **apoio à decisão**. Não realiza "
    "diagnóstico definitivo, não prescreve e a avaliação final cabe sempre ao "
    "profissional responsável pelo atendimento."
)


def render_markdown(analysis: dict[str, Any]) -> str:
    srcs = ", ".join(analysis.get("rag_sources", [])) or "—"
    md = [
        f"# Análise de Atendimento — {datetime.now():%d/%m/%Y %H:%M}",
        "",
        DISCLAIMER,
        "",
        "## Dados de entrada",
        "```json",
        json.dumps(analysis.get("features", {}), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Resultado do modelo",
        f"- **Modelo:** {analysis.get('model')}",
        f"- **Predição de risco:** {analysis.get('prediction')}",
        f"- **Confiança:** {analysis.get('probability', 0):.2f}",
        "",
        "## Fontes consultadas (RAG)",
        f"- {srcs}",
        "",
        "## Explicação (LLM)",
        analysis.get("explanation", "_sem resposta_"),
        "",
    ]
    return "\n".join(md)


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
