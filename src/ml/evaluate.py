"""Métricas de comparação (requisito #4).

Accuracy sozinha esconde o erro que mais importa neste cenário: classificar
como baixo um atendimento de risco alto (falso negativo). A tabela prioriza
o recall da classe ``alto`` e também guarda precisão e F1.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


def evaluate_one(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    labels = sorted(pd.Series(y).astype(str).unique())
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    recall_by_class = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    precision_by_class = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    for label, rec, prec in zip(labels, recall_by_class, precision_by_class):
        metrics[f"recall_{label}"] = float(rec)
        metrics[f"precision_{label}"] = float(prec)
    return metrics


def compare_models(models: dict[str, object], X, y) -> pd.DataFrame:
    """Tabela comparativa. Ordena pelo recall da classe de maior gravidade."""
    rows = {name: evaluate_one(m, X, y) for name, m in models.items()}
    df = pd.DataFrame(rows).T
    if "recall_alto" in df.columns:
        cols = ["recall_alto", "precision_alto", "f1"]
    else:
        cols = ["recall", "f1"]
    return df.sort_values(cols, ascending=False)


def cross_validate_models(
    models: dict[str, object],
    X,
    y,
    folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara modelos por validação cruzada sem tocar no conjunto de teste."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows: dict[str, dict[str, float]] = {}
    for name, estimator in models.items():
        fold_metrics: list[dict[str, float]] = []
        for train_index, valid_index in splitter.split(X, y):
            fitted = clone(estimator)
            fitted.fit(X.iloc[train_index], y.iloc[train_index])
            fold_metrics.append(evaluate_one(fitted, X.iloc[valid_index], y.iloc[valid_index]))
        frame = pd.DataFrame(fold_metrics)
        rows[name] = {
            f"{column}_mean": float(frame[column].mean())
            for column in frame.columns
        } | {
            f"{column}_std": float(frame[column].std(ddof=1))
            for column in frame.columns
        }
    result = pd.DataFrame(rows).T
    return result.sort_values(
        ["recall_alto_mean", "precision_alto_mean", "f1_mean"],
        ascending=False,
    )


def confusion(model, X, y) -> pd.DataFrame:
    labels = sorted(pd.Series(y).astype(str).unique())
    matrix = confusion_matrix(y, model.predict(X), labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


METRIC_RATIONALE = (
    "Na triagem de risco materno, deixar passar um caso alto (falso negativo) "
    "atrasa a revisão do profissional. A seleção usa a média de cinco folds e "
    "ordena pelo recall da classe alto. Se dois modelos empatam, fica o de maior "
    "precisão na mesma classe: o falso alarme ocupa tempo de equipe, e só "
    "desempatamos por ele quando a chance de silenciar um caso alto é a mesma. "
    "A tabela abaixo é a avaliação final no teste; accuracy não decide sozinha."
)
