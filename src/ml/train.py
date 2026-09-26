"""Treinamento dos modelos + comparação (requisitos #3 e #4)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import settings
from src.data.loader import load_dataset
from src.data.preprocessing import preprocess, split
from src.ml.evaluate import compare_models
from src.ml.models import get_models


def train_all(target: str | None = None, save: bool = True) -> pd.DataFrame:
    """Treina todos os candidatos e retorna o DataFrame comparativo de métricas."""
    target = target or settings.ml.target_column
    df = preprocess(load_dataset())
    if target not in df.columns:
        raise KeyError(f"Coluna alvo '{target}' ausente. Colunas: {list(df.columns)}")

    Xtr, Xte, ytr, yte = split(df, target)
    models = get_models(settings.ml.random_state)

    fitted: dict[str, object] = {}
    for name, est in models.items():
        est.fit(Xtr, ytr)
        fitted[name] = est
        if save:
            out = settings.path(settings.models_dir)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, out / f"{name}.joblib")

    return compare_models(fitted, Xte, yte)
