"""Sinais fora da faixa de atenção e combinações incomuns.

Isolation Forest + faixas clínicas. Não classifica risco e não dispara
encaminhamento automático: só destaca o que o profissional deve reler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Faixa de atenção da aplicação — não é critério diagnóstico.
VITAL_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp": (90.0, 139.0),
    "diastolic_bp": (60.0, 89.0),
    "heart_rate": (60.0, 100.0),
    "body_temp": (36.0, 37.7),
    "blood_sugar": (70.0, 139.0),
}

RANGE_LABELS = {
    "systolic_bp": "pressão sistólica",
    "diastolic_bp": "pressão diastólica",
    "heart_rate": "frequência cardíaca",
    "body_temp": "temperatura",
    "blood_sugar": "glicemia",
}


def zscore_flags(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Marca valores fora de `threshold` desvios-padrão por coluna."""
    out = df.copy()
    for col in columns:
        mean, std = out[col].mean(), out[col].std()
        out[f"{col}_zscore"] = np.abs((out[col] - mean) / std) if std else 0.0
        out[f"{col}_flag"] = out[f"{col}_zscore"] > threshold
    return out


def fit_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
) -> IsolationForest:
    cols = list(df.select_dtypes("number").columns)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(df[cols])
    return model


def reference_range_flags(row: dict) -> list[str]:
    """Nomes das variáveis fora da faixa de atenção."""
    flags = []
    for signal, (lo, hi) in VITAL_RANGES.items():
        value = row.get(signal)
        if value is None:
            continue
        if not (lo <= float(value) <= hi):
            flags.append(signal)
    return flags


def describe_flags(flags: list[str]) -> list[str]:
    return [RANGE_LABELS.get(flag, flag) for flag in flags]
