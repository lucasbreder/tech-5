"""Treinamento dos classificadores, anomalia e artefatos (requisitos #3 e #4)."""

from __future__ import annotations

import json
import hashlib
import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from loguru import logger

from src.config import settings
from src.data.loader import load_dataset
from src.data.preprocessing import FEATURES, TARGET, prepare_dataset, preprocess
from src.ml.anomaly import fit_isolation_forest
from src.ml.evaluate import compare_models, confusion, cross_validate_models
from src.ml.explain import feature_importance, shap_mean_abs
from src.ml.models import get_models
from src.ml.registry import ANOMALY_NAME, METADATA_NAME


def train_all(target: str | None = None, save: bool = True) -> pd.DataFrame:
    """Treina os classificadores e devolve a tabela de métricas no teste."""
    target = target or settings.ml.target_column
    raw = load_dataset()
    df = preprocess(raw, target=None)
    if target not in df.columns:
        raise KeyError(f"Coluna alvo '{target}' ausente. Colunas: {list(df.columns)}")

    from src.data.preprocessing import split

    Xtr, Xte, ytr, yte = split(df, target, scale_numeric=False)
    models = get_models(settings.ml.random_state)
    cv_table = cross_validate_models(
        models,
        Xtr,
        ytr,
        random_state=settings.ml.random_state,
    )
    best = str(cv_table.index[0])

    fitted: dict[str, object] = {}
    for name, est in models.items():
        est.fit(Xtr, ytr)
        fitted[name] = est

    table = compare_models(fitted, Xte, yte).reindex(cv_table.index)

    if save:
        out = settings.path(settings.models_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, est in fitted.items():
            joblib.dump(est, out / f"{name}.joblib")
            _save_importance(est, list(Xtr.columns), out / f"importance_{name}.csv")
            confusion(est, Xte, yte).to_csv(out / f"confusion_{name}.csv")

        anomaly = fit_isolation_forest(Xtr, random_state=settings.ml.random_state)
        joblib.dump(anomaly, out / ANOMALY_NAME)
        table.to_csv(out / "metrics.csv")
        cv_table.to_csv(out / "cv_metrics.csv")
        _save_processed(df)
        shap_model = _save_shap(fitted, Xte, out / "shap_mean_abs.csv")
        prepared = prepare_dataset(raw).drop_duplicates()
        conflicting = prepared.groupby(FEATURES)[TARGET].transform("nunique") > 1
        meta = {
            "dataset": "UCI Maternal Health Risk Data Set",
            "dataset_url": "https://archive.ics.uci.edu/dataset/863/maternal+health+risk",
            "dataset_sha256": _dataset_sha256(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "rows_raw": int(len(raw)),
            "rows_after_exact_deduplication": int(len(prepared)),
            "rows_removed_label_conflict": int(conflicting.sum()),
            "rows_prepared": int(len(df)),
            "rows_train": int(len(Xtr)),
            "rows_test": int(len(Xte)),
            "features": list(Xtr.columns),
            "target": target,
            "labels": sorted(df[target].astype(str).unique()),
            "best_model": best,
            "selection_rule": (
                "5-fold CV no treino: recall_alto_mean, depois "
                "precision_alto_mean, depois f1_mean"
            ),
            "shap_model": shap_model,
            "units": {
                "age": "anos",
                "systolic_bp": "mmHg",
                "diastolic_bp": "mmHg",
                "blood_sugar": "mg/dL",
                "body_temp": "C",
                "heart_rate": "bpm",
            },
            "feature_baselines": {
                column: float(Xtr[column].median()) for column in Xtr.columns
            },
            "class_counts": df[target].value_counts().astype(int).to_dict(),
        }
        (out / METADATA_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Treino concluído. Modelo escolhido: {best}.")

    return table


def _save_importance(model, columns: list[str], path) -> None:
    try:
        feature_importance(model, columns).to_csv(path, header=["importance"])
    except AttributeError as exc:
        logger.warning(f"Sem importância para {path.name}: {exc}")


def _save_shap(fitted: dict[str, object], Xte: pd.DataFrame, path) -> str | None:
    """SHAP no primeiro modelo de árvore que o explicador aceita.

    Gradient boosting multiclasse não é coberto pelo TreeExplainer atual;
    a random forest é a alternativa usada no artefato.
    """
    sample = Xte.head(80)
    order = ["random_forest", *[name for name in fitted if name != "random_forest"]]
    for name in order:
        model = fitted.get(name)
        if model is None or not hasattr(model, "estimators_"):
            continue
        try:
            shap_mean_abs(model, sample).to_csv(path, header=["mean_abs_shap"])
            return name
        except Exception as exc:
            logger.warning(f"SHAP indisponível em {name}: {exc}")
    return None


def _save_processed(df: pd.DataFrame) -> None:
    dest = settings.path(settings.data_processed_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ordered = df[[c for c in FEATURES if c in df.columns] + [TARGET]]
    ordered.to_csv(dest / "maternal_health_risk.csv", index=False)


def _dataset_sha256() -> str:
    candidates = sorted(settings.path(settings.data_raw_dir).glob("*.csv"))
    if not candidates:
        return ""
    return hashlib.sha256(candidates[0].read_bytes()).hexdigest()
