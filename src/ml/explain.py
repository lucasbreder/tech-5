"""Interpretabilidade: importância das variáveis e SHAP (requisito #5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _final_estimator(model):
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        return model.named_steps["clf"]
    return model


def feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Importância global do modelo (ganho das árvores ou |coef| médio)."""
    est = _final_estimator(model)
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.abs(np.asarray(est.coef_, dtype=float))
        values = coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
    else:
        raise AttributeError("Modelo sem importância acessível — use shap_values().")
    if len(values) != len(feature_names):
        raise ValueError(
            f"Importância com {len(values)} valores para {len(feature_names)} colunas."
        )
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def local_probability_contributions(
    model,
    X: pd.DataFrame,
    predicted_label: str,
    baselines: dict[str, float],
) -> pd.Series:
    """Efeito local por substituição: P(caso) - P(feature na mediana do treino).

    Valor positivo indica que o valor informado aumenta a probabilidade da
    classe prevista em relação à mediana; negativo indica redução. É uma
    explicação local equivalente, não um valor SHAP aditivo.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Modelo sem predict_proba para contribuição local.")
    classes = [str(value) for value in model.classes_]
    class_index = classes.index(str(predicted_label))
    base_probability = float(model.predict_proba(X)[0][class_index])
    effects: dict[str, float] = {}
    for column in X.columns:
        if column not in baselines:
            continue
        counterfactual = X.copy()
        counterfactual.loc[counterfactual.index[0], column] = float(baselines[column])
        changed_probability = float(model.predict_proba(counterfactual)[0][class_index])
        effects[column] = base_probability - changed_probability
    return pd.Series(effects).reindex(
        sorted(effects, key=lambda name: abs(effects[name]), reverse=True)
    )


def shap_mean_abs(model, X: pd.DataFrame) -> pd.Series:
    """Média do |SHAP| por variável, agregando as classes quando houver."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        stacked = np.mean([np.abs(np.asarray(item)) for item in values], axis=0)
    else:
        array = np.abs(np.asarray(values))
        if array.ndim == 3 and array.shape[-1] == X.shape[1]:
            stacked = array.mean(axis=1)
        elif array.ndim == 3:
            stacked = array.mean(axis=-1)
        else:
            stacked = array
    return pd.Series(np.asarray(stacked).mean(axis=0), index=X.columns).sort_values(ascending=False)


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Matriz SHAP. Em modelo multiclasse, devolve a classe de maior |SHAP| médio.

    O valor explica a contribuição de cada variável naquela linha, em relação
    ao que o modelo viu no treino.
    """
    import shap

    explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model, X)
    raw = explainer(X).values
    array = np.asarray(raw)
    if array.ndim == 3:
        # (n_amostras, n_classes, n_features) ou (n_amostras, n_features, n_classes)
        if array.shape[1] == X.shape[1]:
            array = array[:, :, int(np.abs(array).mean(axis=(0, 1)).argmax())]
        else:
            class_index = int(np.abs(array).mean(axis=(0, 2)).argmax())
            array = array[:, class_index, :]
    return pd.DataFrame(array, columns=X.columns, index=X.index)
