"""Leitura dos artefatos salvos pelo treino."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from src.config import settings
from src.data.preprocessing import FEATURES

METADATA_NAME = "metadata.json"
ANOMALY_NAME = "isolation_forest.joblib"


def models_dir() -> Path:
    return settings.path(settings.models_dir)


def metadata_path() -> Path:
    return models_dir() / METADATA_NAME


def load_metadata() -> dict:
    path = metadata_path()
    if not path.exists():
        return {"features": FEATURES, "best_model": "random_forest"}
    return json.loads(path.read_text(encoding="utf-8"))


def feature_columns() -> list[str]:
    return list(load_metadata().get("features") or FEATURES)


def feature_frame(features: dict) -> pd.DataFrame:
    cols = feature_columns()
    missing = [c for c in cols if c not in features or features[c] is None]
    if missing:
        raise KeyError(f"Campos ausentes no atendimento: {missing}")
    row = {c: float(features[c]) for c in cols}
    invalid = [name for name, value in row.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Valores não finitos no atendimento: {invalid}")
    return pd.DataFrame([row], columns=cols)


def model_path(name: str) -> Path:
    return models_dir() / f"{name}.joblib"


def load_model(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo '{name}' não encontrado em {path}. Rode `python main.py train`."
        )
    return joblib.load(path)


def available_classifiers() -> list[str]:
    names = []
    for path in sorted(models_dir().glob("*.joblib")):
        if path.name == ANOMALY_NAME:
            continue
        names.append(path.stem)
    return names


def load_metrics() -> pd.DataFrame | None:
    path = models_dir() / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)
