"""Entraînement des modèles (reprend les choix du notebook Capstone_project)."""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import Settings
from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET_CLF,
    TARGET_REG,
    add_features,
)

REGRESSION_ALGOS = ("random_forest", "ridge", "lightgbm")
CLASSIFICATION_ALGOS = ("lightgbm", "random_forest", "logistic_regression")

REGRESSOR_FILE = "regressor.joblib"
CLASSIFIER_FILE = "classifier.joblib"
METADATA_FILE = "metadata.json"

REQUIRED_COLUMNS = RAW_FEATURES + [TARGET_REG]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering + imputation/standardisation + one-hot (section 2.3)."""
    prep_num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    prep_cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        [("num", prep_num, NUMERIC_FEATURES), ("cat", prep_cat, CATEGORICAL_FEATURES)]
    )
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns)])


def _make_regressor(algo: str, rs: int):
    if algo == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=rs)
    if algo == "ridge":
        return Ridge(alpha=1.0)
    if algo == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=rs, verbose=-1
        )
    raise ValueError(f"Algorithme de régression inconnu : {algo}")


def _make_classifier(algo: str, rs: int, params: dict[str, Any] | None = None):
    if algo == "lightgbm":
        base = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)
        base.update(params or {})
        return lgb.LGBMClassifier(**base, is_unbalance=True, n_jobs=-1, random_state=rs, verbose=-1)
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=rs
        )
    if algo == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=rs)
    raise ValueError(f"Algorithme de classification inconnu : {algo}")


def _optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int, rs: int) -> dict[str, Any]:
    """Optimisation Optuna du LightGBM (espace de recherche et métrique du notebook : PR-AUC)."""
    import optuna  # import local : dépendance lourde, utile seulement si n_trials > 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        }
        pipe = Pipeline([("prep", build_preprocessor()), ("model", _make_classifier("lightgbm", rs, params))])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
        # n_jobs=1 côté CV : LightGBM parallélise déjà en interne (évite la sur-souscription, surtout sous Windows)
        return cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": threshold,
    }


def _save_atomic(obj: Any, path: Path) -> None:
    """Écriture atomique : un chargement concurrent ne voit jamais un fichier tronqué."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def train(
    settings: Settings,
    tasks: tuple[str, ...] = ("regression", "classification"),
    regression_algorithm: str = "random_forest",
    classification_algorithm: str = "lightgbm",
    optuna_trials: int = 0,
) -> dict[str, Any]:
    """Entraîne les modèles demandés, les évalue sur le jeu de test, les sauvegarde.

    Reprend le protocole du notebook : split 80/20 stratifié sur `at_risk`,
    random_state=42. Retourne les métadonnées (aussi écrites dans metadata.json).
    """
    rs = settings.random_state
    df = load_dataset(settings.data_path)
    df[TARGET_CLF] = (df[TARGET_REG] < settings.risk_cutoff).astype(int)

    X = df[RAW_FEATURES]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, df[TARGET_REG], df[TARGET_CLF], test_size=settings.test_size, random_state=rs, stratify=df[TARGET_CLF]
    )

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = settings.models_dir / METADATA_FILE
    metadata: dict[str, Any] = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {
        "trained_at": now,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "dataset": str(settings.data_path),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
    }

    if "regression" in tasks:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", _make_regressor(regression_algorithm, rs))])
        pipe.fit(X_tr, yr_tr)
        metrics = _regression_metrics(yr_te, pipe.predict(X_te))
        _save_atomic(pipe, settings.models_dir / REGRESSOR_FILE)
        metadata["regression"] = {"algorithm": regression_algorithm, "metrics": metrics, **common}

    if "classification" in tasks:
        best_params: dict[str, Any] = {}
        if optuna_trials > 0 and classification_algorithm == "lightgbm":
            best_params = _optimize_lgbm(X_tr, yc_tr, optuna_trials, rs)
        pipe = Pipeline(
            [("prep", build_preprocessor()), ("model", _make_classifier(classification_algorithm, rs, best_params))]
        )
        pipe.fit(X_tr, yc_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        metrics = _classification_metrics(yc_te, proba, settings.risk_threshold)
        _save_atomic(pipe, settings.models_dir / CLASSIFIER_FILE)
        metadata["classification"] = {
            "algorithm": classification_algorithm,
            "metrics": metrics,
            "best_params": best_params,
            "risk_cutoff": settings.risk_cutoff,
            **common,
        }

    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), "utf-8")
    os.replace(meta_tmp, meta_path)
    return metadata
