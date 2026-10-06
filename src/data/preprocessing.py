"""Preparação de dados: limpeza, unidades clínicas e split (requisito #2).

O conjunto público UCI *Maternal Health Risk* registra temperatura em
Fahrenheit e glicemia em mmol/L. A aplicação fala com o profissional em
°C e mg/dL. Duplicatas são removidas antes do split para não vazar o
mesmo atendimento entre treino e teste.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings

FEATURES: list[str] = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar",
    "body_temp",
    "heart_rate",
]
TARGET = "risk_level"

RAW_RENAME = {
    "Age": "age",
    "SystolicBP": "systolic_bp",
    "DiastolicBP": "diastolic_bp",
    "BS": "blood_sugar",
    "BodyTemp": "body_temp",
    "HeartRate": "heart_rate",
    "RiskLevel": "risk_level",
}

LABEL_MAP = {
    "low risk": "baixo",
    "mid risk": "moderado",
    "high risk": "alto",
}

FEATURE_LABELS = {
    "age": "idade (anos)",
    "systolic_bp": "pressão sistólica (mmHg)",
    "diastolic_bp": "pressão diastólica (mmHg)",
    "blood_sugar": "glicemia (mg/dL)",
    "body_temp": "temperatura (°C)",
    "heart_rate": "frequência cardíaca (bpm)",
}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o CSV público para o esquema usado no treino e na tela.

    Remove frequência cardíaca incompatível com um registro clínico
    (valor conhecido do dataset: 7 bpm).
    """
    out = df.copy()
    out = out.rename(columns={k: v for k, v in RAW_RENAME.items() if k in out.columns})

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Colunas ausentes após normalização: {missing}")

    out = out[required]
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fahrenheit costuma ficar acima de 45; mmol/L de glicemia, abaixo de 30.
    if out["body_temp"].median(skipna=True) > 45:
        out["body_temp"] = (out["body_temp"] - 32.0) * 5.0 / 9.0
    if out["blood_sugar"].median(skipna=True) < 30:
        out["blood_sugar"] = out["blood_sugar"] * 18.0

    out[TARGET] = out[TARGET].astype(str).str.strip().str.lower().replace(LABEL_MAP)
    out = out[out["heart_rate"].between(40, 220)]
    out = out.dropna(subset=required)
    return out.reset_index(drop=True)


def preprocess(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Remove duplicatas, conflitos de rótulo e deixa a coluna alvo por último.

    O UCI contém vetores de sinais idênticos associados a riscos diferentes.
    Sem história clínica para desempatar, escolher um rótulo seria arbitrário.
    Todos os grupos ambíguos são removidos e essa perda é registrada no treino.
    """
    out = prepare_dataset(df) if _looks_raw(df) or target is None else df.copy()
    if TARGET not in out.columns and "risk_level" not in out.columns:
        out = prepare_dataset(df)

    target = target or settings.ml.target_column
    missing = [column for column in FEATURES + [target] if column not in out.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

    out = out.drop_duplicates()
    if all(column in out.columns for column in FEATURES + [TARGET]):
        label_counts = out.groupby(FEATURES, dropna=False)[TARGET].transform("nunique")
        out = out[label_counts == 1]
    numeric = out.select_dtypes("number")
    out[numeric.columns] = out[numeric.columns].fillna(numeric.median(numeric_only=True))
    out = out.dropna()

    if target in out.columns:
        cols = [c for c in FEATURES if c in out.columns and c != target]
        extra = [c for c in out.columns if c not in cols and c != target]
        out = out[cols + extra + [target]]
    return out.reset_index(drop=True)


def _looks_raw(df: pd.DataFrame) -> bool:
    return "RiskLevel" in df.columns or "SystolicBP" in df.columns


def split(
    df: pd.DataFrame,
    target: str,
    scale_numeric: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split estratificado.

    A escala fica dentro do pipeline dos modelos que precisam dela
    (regressão logística). Escalar aqui quebraria a predição: a tela
    envia valores clínicos, não z-scores.
    """
    if scale_numeric:
        raise ValueError(
            "Não escale no split. A regressão logística padroniza no próprio pipeline; "
            "a interface envia unidades clínicas."
        )
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=settings.ml.test_size,
        random_state=settings.ml.random_state,
        stratify=stratify,
    )
