"""Preparação de dados: limpeza, encoding, escala e split (requisito #2)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings


def preprocess(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Remove duplicatas/nulos e ordena colunas (coluna alvo por último)."""
    out = df.copy()
    out = out.drop_duplicates()
    numeric = out.select_dtypes("number")
    out[numeric.columns] = out[numeric.columns].fillna(numeric.median(numeric_only=True))
    out = out.dropna()
    target = target or settings.ml.target_column
    if target in out.columns:
        cols = [c for c in out.columns if c != target] + [target]
        out = out[cols]
    return out


def split(
    df: pd.DataFrame,
    target: str,
    scale_numeric: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split estratificado com padronização opcional."""
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if y.nunique() <= 10 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=settings.ml.test_size,
        random_state=settings.ml.random_state,
        stratify=stratify,
    )
    if scale_numeric:
        scaler = StandardScaler()
        num = Xtr.select_dtypes("number").columns
        Xtr[num] = scaler.fit_transform(Xtr[num])
        Xte[num] = scaler.transform(Xte[num])
    return Xtr, Xte, ytr, yte
