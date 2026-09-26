"""Detecção de anomalias em sinais vitais — reuso do anomaly_agent do tech-4.

Pode funcionar como UM DOS modelos de risco (Isolation Forest + Z-Score)
e também como checagem de consistência dos dados de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Faixas de referência (gestante) — adaptadas do tech-4.
VITAL_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp": (90.0, 140.0),
    "diastolic_bp": (60.0, 90.0),
    "heart_rate": (60.0, 100.0),
    "temperature": (36.1, 37.2),
    "oxygen_saturation": (95.0, 100.0),
    "glucose": (70.0, 140.0),
}


def zscore_flags(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Marca valores fora de `threshold` desvios-padrão por coluna."""
    out = df.copy()
    for col in columns:
        mean, std = out[col].mean(), out[col].std()
        out[f"{col}_zscore"] = np.abs((out[col] - mean) / std) if std else 0.0
        out[f"{col}_flag"] = out[f"{col}_zscore"] > threshold
    return out


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> IsolationForest:
    cols = list(df.select_dtypes("number").columns)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(df[cols])
    return model


def reference_range_flags(row: dict) -> list[str]:
    """Lista de sinais fora da faixa clínica de referência."""
    flags = []
    for signal, (lo, hi) in VITAL_RANGES.items():
        value = row.get(signal)
        if value is not None and not (lo <= float(value) <= hi):
            flags.append(signal)
    return flags
