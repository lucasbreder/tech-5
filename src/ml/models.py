"""Definição de candidatos de modelos de Machine Learning (requisito #3).

Ao menos DOIS modelos treinados e comparados.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Retorna {nome: estimator} — pipelines prontos para X tabular."""
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=random_state),
    }
