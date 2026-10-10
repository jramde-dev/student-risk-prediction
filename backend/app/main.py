"""API FastAPI : entraînement et déploiement des modèles « étudiants à risque »."""
from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware


from . import training
from .config import Settings
from .registry import ModelStore
from .schemas import BatchRequest, BatchResponse, JobStatus, Prediction, StudentInput, TrainRequest

log = logging.getLogger("student-risk-api")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = ModelStore(settings)
    jobs: dict[str, dict] = {}
    train_lock = threading.Lock()  # un seul entraînement à la fois

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.load()
        log.info("Modèles chargés : %s", store.status)
        yield

    app = FastAPI(
        title="Student Risk API",
        description="Entraînement et prédiction du score de santé mentale / du risque chez les étudiants.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store, app.state.jobs, app.state.settings = store, jobs, settings

# ... juste après la création de `app` (après app.state...) :
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # ------------------------------------------------------------------ santé
    @app.get("/health", tags=["Système"])
    def health():
        return {"status": "ok", "models_loaded": store.status, "training_enabled": settings.training_enabled}

    # ------------------------------------------------------------ entraînement
    def _run_training(job_id: str, req: TrainRequest) -> None:
        try:
            training.train(
                settings,
                tasks=tuple(req.tasks),
                regression_algorithm=req.regression_algorithm,
                classification_algorithm=req.classification_algorithm,
                optuna_trials=req.optuna_trials,
            )
            store.load()  # bascule à chaud sur les nouveaux modèles
            jobs[job_id].update(status="succeeded")
        except Exception as exc:  # noqa: BLE001 - on remonte l'erreur via le statut du job
            log.exception("Échec de l'entraînement %s", job_id)
            jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            jobs[job_id]["finished_at"] = _now()
            train_lock.release()


    @app.get("/health", tags=["Système"])
    def health():
        return {"status": "ok", "models_loaded": store.status, "training_enabled": settings.training_enabled}
   
    @app.post("/train", status_code=202, response_model=JobStatus, tags=["Entraînement"])
    def start_training(req: TrainRequest, background: BackgroundTasks):
        """Lance un entraînement en arrière-plan ; suivre l'avancement via GET /train/{job_id}."""
        if not settings.training_enabled:
            raise HTTPException(403, "Entraînement désactivé sur ce déploiement (APP_TRAINING_ENABLED=false).")
        if not settings.data_path.exists():
            raise HTTPException(422, f"Dataset introuvable : {settings.data_path}")      
        if not train_lock.acquire(blocking=False):
            raise HTTPException(409, "Un entraînement est déjà en cours.")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"job_id": job_id, "status": "running", "started_at": _now(), "request": req}
        background.add_task(_run_training, job_id, req)
        return jobs[job_id]

    @app.get("/train/{job_id}", response_model=JobStatus, tags=["Entraînement"])
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Job inconnu.")
        return jobs[job_id]

    # ------------------------------------------------------------------ modèles
    @app.get("/models", tags=["Modèles"])
    def models_info():
        """Algorithme, métriques de test et versions des modèles actuellement servis."""
        return {"loaded": store.status, "metadata": store.metadata}

    # --------------------------------------------------------------- prédiction
    def _predict(rows: list[dict], threshold: float | None) -> list[dict]:
        try:
            return store.predict(rows, settings.risk_threshold if threshold is None else threshold)
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc

    ThresholdQ = Query(None, ge=0, le=1, description="Seuil de décision (défaut : APP_RISK_THRESHOLD)")

    @app.post("/predict", response_model=Prediction, tags=["Prédiction"])
    def predict(student: StudentInput, threshold: float | None = ThresholdQ):
        return _predict([student.model_dump()], threshold)[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["Prédiction"])
    def predict_batch(req: BatchRequest, threshold: float | None = ThresholdQ):
        return {"predictions": _predict([s.model_dump() for s in req.students], threshold)}

    return app


app = create_app()
