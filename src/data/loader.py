"""Carregamento do dataset tabular de saúde da mulher."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega CSV/Parquet de `data/raw/`.

    Args:
        path: arquivo específico. Se None, usa o primeiro CSV/Parquet em raw/.
    """
    raw = settings.path(settings.data_raw_dir)
    if path is None:
        candidates = sorted(raw.glob("*.csv")) + sorted(raw.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"Nenhum dataset em {raw}. Coloque um .csv/.parquet em data/raw/."
            )
        path = candidates[0]
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Formato não suportado: {path.suffix}")
