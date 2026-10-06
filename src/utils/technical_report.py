"""Geração do relatório técnico de entrega em PDF."""

from __future__ import annotations

import html
from pathlib import Path

from src.config import PROJECT_ROOT


def generate_technical_report(
    source: str | Path = "relatorio/RELATORIO_TECNICO.md",
    destination: str | Path = "relatorio/relatorio_tecnico.pdf",
) -> Path:
    """Converte o relatório Markdown simples em PDF sem dependências de sistema."""
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    source_path = _resolve(source)
    destination_path = _resolve(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitlePt",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyPt",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletPt",
            parent=styles["BodyPt"],
            leftIndent=14,
            firstLineIndent=-8,
        )
    )

    story = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.12 * cm))
        elif line == "---":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), styles["TitlePt"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"• {_inline(line[2:])}", styles["BulletPt"]))
        elif line.startswith("> "):
            story.append(Paragraph(_inline(line[2:]), styles["Italic"]))
        elif line.startswith("```"):
            continue
        else:
            story.append(Paragraph(_inline(line), styles["BodyPt"]))

    document = SimpleDocTemplate(
        str(destination_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Guardiã AI — Relatório Técnico",
        author="Equipe Guardiã AI",
    )
    document.build(story)
    return destination_path


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _inline(text: str) -> str:
    escaped = html.escape(text)
    while "**" in escaped:
        escaped = escaped.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return escaped.replace("`", "")
