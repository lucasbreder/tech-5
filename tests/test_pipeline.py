"""Preparação, métricas, relato e jornada sem chave de LLM."""

from __future__ import annotations

import pandas as pd

from src.data.preprocessing import prepare_dataset, preprocess
from src.ml.anomaly import reference_range_flags
from src.ml.registry import feature_frame
from src.services.narrative import scan_report


def test_prepare_converts_uci_units_and_labels():
    raw = pd.DataFrame(
        {
            "Age": [25, 35],
            "SystolicBP": [120, 150],
            "DiastolicBP": [80, 100],
            "BS": [7.0, 11.0],
            "BodyTemp": [98.6, 102.0],
            "HeartRate": [80, 7],
            "RiskLevel": ["low risk", "high risk"],
        }
    )
    out = prepare_dataset(raw)
    assert len(out) == 1
    assert out.loc[0, "risk_level"] == "baixo"
    assert abs(out.loc[0, "blood_sugar"] - 126.0) < 0.2
    assert 36.5 < out.loc[0, "body_temp"] < 37.5


def test_preprocess_drops_duplicate_after_preparation():
    raw = pd.DataFrame(
        {
            "Age": [30, 30, 40],
            "SystolicBP": [120, 120, 150],
            "DiastolicBP": [80, 80, 95],
            "BS": [6.5, 6.5, 12.0],
            "BodyTemp": [98.0, 98.0, 101.0],
            "HeartRate": [76, 76, 88],
            "RiskLevel": ["low risk", "low risk", "high risk"],
        }
    )
    out = preprocess(raw)
    assert len(out) == 2
    assert set(out["risk_level"]) == {"baixo", "alto"}
    assert out.columns[-1] == "risk_level"


def test_preprocess_removes_same_features_with_conflicting_labels():
    raw = pd.DataFrame(
        {
            "Age": [30, 30, 40],
            "SystolicBP": [120, 120, 150],
            "DiastolicBP": [80, 80, 95],
            "BS": [6.5, 6.5, 12.0],
            "BodyTemp": [98.0, 98.0, 101.0],
            "HeartRate": [76, 76, 88],
            "RiskLevel": ["low risk", "high risk", "high risk"],
        }
    )
    out = preprocess(raw)
    assert len(out) == 1
    assert out.iloc[0]["risk_level"] == "alto"


def test_reference_flags_use_clinical_units():
    flags = reference_range_flags(
        {
            "systolic_bp": 150,
            "diastolic_bp": 96,
            "blood_sugar": 180,
            "body_temp": 36.6,
            "heart_rate": 80,
        }
    )
    assert "systolic_bp" in flags
    assert "blood_sugar" in flags
    assert "heart_rate" not in flags


def test_narrative_lists_attention_terms_without_concluding():
    found = scan_report("Tenho medo de ir para casa. Ele ameaçou e pediu para ninguém ligar.")
    assert "expressão de medo" in found["signals"]
    assert "menção a ameaça" in found["signals"]
    assert "pedido para não contatar terceiros" in found["signals"]
    assert "não confirma" in found["note"]

    empty = scan_report("Pré-natal em dia, sem queixa.")
    assert empty["signals"] == []
    assert scan_report("  ")["present"] is False


def test_feature_frame_rejects_non_finite_values():
    values = {
        "age": 30,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "blood_sugar": float("nan"),
        "body_temp": 36.7,
        "heart_rate": 78,
    }
    try:
        feature_frame(values)
    except ValueError as exc:
        assert "blood_sugar" in str(exc)
    else:
        raise AssertionError("NaN deveria ser rejeitado")
