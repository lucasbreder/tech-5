"""Interpretabilidade: SHAP e feature importance (requisito #5)."""

from __future__ import annotations

import pandas as pd


def feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Importância nativa (estimadores sklearn com .feature_importances_)."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return pd.Series(abs(clf.coef_).ravel(), index=feature_names).sort_values(ascending=False)
    raise AttributeError("Modelo sem atributo de importância acessível — use shap_values().")


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Retorna matriz de valores SHAP (explicação de predições individuais).

    Requer `shap` instalado. Use para justificar ao profissional QUAL variável
    puxou o risco para cima/baixo naquela paciente.
    """
    import shap  # import tardio: shap é pesado

    explainer = shap.Explainer(model, X)
    return pd.DataFrame(explainer(X).values, columns=X.columns)
