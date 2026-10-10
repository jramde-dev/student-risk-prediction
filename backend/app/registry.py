"""Chargement/rechargement des modèles en mémoire (thread-safe)."""
from __future__ import annotations

from importlib.metadata import metadata
import json
import logging
import threading
from typing import Any

import joblib
import lightgbm
import pandas as pd
import sklearn

from .config import Settings
from .features import RAW_FEATURES
from .training import CLASSIFIER_FILE, METADATA_FILE, REGRESSOR_FILE

log = logging.getLogger("student-risk-api")

class ModelStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.regressor = None
        self.classifier = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        """(Re)charge depuis le disque ce qui existe ; ignore ce qui manque."""
        d = self.settings.models_dir
        with self._lock:
            self.regressor = joblib.load(d / REGRESSOR_FILE) if (d / REGRESSOR_FILE).exists() else None
            self.classifier = joblib.load(d / CLASSIFIER_FILE) if (d / CLASSIFIER_FILE).exists() else None
            meta = d / METADATA_FILE
            self.metadata = json.loads(meta.read_text("utf-8")) if meta.exists() else {}
        _warn_on_version_mismatch(self.metadata)

    @property
    def status(self) -> dict[str, bool]:
        return {"regression": self.regressor is not None, "classification": self.classifier is not None}

    def predict(self, rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
        with self._lock:
            reg, clf = self.regressor, self.classifier
        if reg is None and clf is None:
            raise LookupError("Aucun modèle entraîné : appelez d'abord POST /train.")
        X = pd.DataFrame(rows)[RAW_FEATURES]
        scores = reg.predict(X) if reg is not None else [None] * len(X)
        probas = clf.predict_proba(X)[:, 1] if clf is not None else [None] * len(X)
        out = []
        for s, p in zip(scores, probas):
            out.append(
                {
                    "mental_health_score": None if s is None else round(float(s), 2),
                    "at_risk_probability": None if p is None else round(float(p), 4),
                    "at_risk": None if p is None else bool(p >= threshold),
                    "threshold": threshold,
                }
            )
        return out
    
def _warn_on_version_mismatch(metadata: dict[str, Any]) -> None:
    """Un modèle pickle n'est garanti que pour les versions de scikit-learn/LightGBM utilisées à l'entraînement."""
    runtime = {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}
    for task in ("regression", "classification"):
        trained = metadata.get(task, {}).get("versions", {})
        for lib, now in runtime.items():
            if trained.get(lib) and trained[lib] != now:
                log.warning("%s : entraîné avec %s %s, exécuté avec %s %s", task, lib, trained[lib], lib, now)
