"""Candidatos de classificação de risco materno (requisito #3).

Três modelos supervisionados são treinados e comparados. A detecção de
anomalia (Isolation Forest) entra à parte, como leitura de combinação
incomum — não como mais uma classe de risco.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models(random_state: int = 42) -> dict[str, object]:
    """Retorna {nome: estimator} prontos para os sinais vitais do atendimento."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }
