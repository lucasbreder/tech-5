"""Registro de auditoria das análises (requisito #11).

Padrão adaptado do alert_service do tech-4: em vez de e-mail, persiste cada
análise em JSONL para rastreabilidade (quais dados, qual resultado, quais
documentos RAG e qual resposta da LLM).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings


def log_analysis(record: dict[str, Any]) -> Path:
    """Anexa um registro de análise ao log de auditoria (JSONL)."""
    out_dir = settings.path(settings.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"audits_{datetime.now():%Y%m%d}.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
        "disclaimer": "Sistema de apoio à decisão. A avaliação final é do profissional.",
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Auditoria registrada em {path.name}")
    return path


def read_logs(date: str | None = None) -> list[dict[str, Any]]:
    """Lê registros do dia (YYYYMMDD) ou de todos os logs."""
    out_dir = settings.path(settings.audit_dir)
    files = [out_dir / f"audits_{date}.jsonl"] if date else sorted(out_dir.glob("audits_*.jsonl"))
    records: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"Registro de auditoria inválido ignorado em "
                        f"{path.name}:{line_number}: {exc}"
                    )
    return records
