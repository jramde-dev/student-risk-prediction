"""Configuration de l'application (variables d'environnement préfixées APP_)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    data_path: Path = Path("data/AI_SocialMedia_Student_Dataset.csv")
    models_dir: Path = Path("models")
    random_state: int = 42
    test_size: float = 0.2
    # Un étudiant est « à risque » si Mental_Health_Score < risk_cutoff (cf. notebook)
    risk_cutoff: float = 60.0
    # Seuil de décision par défaut sur la probabilité (modifiable par requête)
    risk_threshold: float = 0.5

    # Dans .env : APP_CORS_ORIGINS=["http://localhost:5173"]  (JSON)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Mettre à false en production : un entraînement public écraserait le modèle (et le disque est éphémère)
    training_enabled: bool = True
