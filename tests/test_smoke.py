"""Smoke test da configuração e do pré-processamento."""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.data.preprocessing import preprocess


def test_settings_loaded():
    assert settings.device in {"cpu", "cuda", "mps"}
    assert 0 < settings.ml.test_size < 1


def test_preprocess_drops_dupes_and_nulls():
    df = pd.DataFrame(
        {
            "age": [30, 30, 40, None],
            "systolic_bp": [120, 120, 150, 130],
            "diastolic_bp": [80, 80, 95, 85],
            "blood_sugar": [100, 100, 180, 110],
            "body_temp": [36.7, 36.7, 38.0, 36.8],
            "heart_rate": [75, 75, 95, 80],
            "risk_level": ["baixo", "baixo", "alto", "baixo"],
        }
    )
    out = preprocess(df, target="risk_level")
    assert len(out) <= len(df)
    assert not out.isna().any().any()
    assert out.columns[-1] == "risk_level"
