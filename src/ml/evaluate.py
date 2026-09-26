"""Métricas de avaliação e comparação de modelos (requisito #4).

Não basta accuracy: em risco materno/segurança, **recall** (não deixar
passar um caso crítico — falso negativo) costuma ser a métrica-chave.
A justificativa vai para o relatório.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_one(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="weighted", zero_division=0),
        "recall": recall_score(y, pred, average="weighted", zero_division=0),
        "f1": f1_score(y, pred, average="weighted", zero_division=0),
    }


def compare_models(models: dict[str, object], X, y) -> pd.DataFrame:
    """Tabela comparativa ordenada por recall decrescente."""
    rows = {name: evaluate_one(m, X, y) for name, m in models.items()}
    df = pd.DataFrame(rows).T.sort_values("recall", ascending=False)
    return df


def confusion(model, X, y):
    return confusion_matrix(y, model.predict(X))
