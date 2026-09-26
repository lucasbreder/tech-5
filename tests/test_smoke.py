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
            "risk_level": ["low", "low", "high", "low"],
        }
    )
    out = preprocess(df, target="risk_level")
    assert len(out) <= len(df)
    assert not out.isna().any().any()
    assert out.columns[-1] == "risk_level"
