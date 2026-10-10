"""Schémas Pydantic (entrées/sorties de l'API)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .training import CLASSIFICATION_ALGOS, REGRESSION_ALGOS

Task = Literal["regression", "classification"]
RegAlgo = Literal[REGRESSION_ALGOS]  # type: ignore[valid-type]
ClfAlgo = Literal[CLASSIFICATION_ALGOS]  # type: ignore[valid-type]


class StudentInput(BaseModel):
    """Plages = celles du dataset d'entraînement (hors plage, le modèle extrapole mal)."""

    Age: int = Field(ge=13, le=25)
    Gender: Literal["Male", "Female", "Non-binary"]
    Education_Level: Literal["High School", "College", "University"]
    Daily_Social_Media_Hours: float = Field(ge=0, le=24)
    Daily_AI_Tool_Usage_Hours: float = Field(ge=0, le=24)
    Sleep_Hours: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Physical_Health_Score: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _total_hours(self):
        if self.Sleep_Hours + self.Daily_Social_Media_Hours + self.Daily_AI_Tool_Usage_Hours + self.Physical_Activity_Hours > 24:
            raise ValueError("La somme des heures quotidiennes dépasse 24 h")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 17,
                "Gender": "Female",
                "Education_Level": "High School",
                "Daily_Social_Media_Hours": 8.0,
                "Daily_AI_Tool_Usage_Hours": 2.9,
                "Sleep_Hours": 4.0,
                "Physical_Activity_Hours": 0.0,
                "Physical_Health_Score": 70.0,
            }
        }
    }


class Prediction(BaseModel):
    mental_health_score: float | None = Field(None, description="Score prédit (0-100), null si le modèle de régression est absent")
    at_risk_probability: float | None = Field(None, description="P(score < 60), null si le classifieur est absent")
    at_risk: bool | None = Field(None, description="at_risk_probability >= threshold")
    threshold: float


class BatchRequest(BaseModel):
    students: list[StudentInput] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class TrainRequest(BaseModel):
    tasks: list[Task] = Field(default=["regression", "classification"], min_length=1)
    regression_algorithm: RegAlgo = "random_forest"
    classification_algorithm: ClfAlgo = "lightgbm"
    optuna_trials: int = Field(0, ge=0, le=200, description="0 = pas d'optimisation ; >0 = essais Optuna (LightGBM uniquement)")


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "succeeded", "failed"]
    started_at: str
    finished_at: str | None = None
    request: TrainRequest
    error: str | None = None
