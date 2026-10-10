import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _synthetic(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    social = rng.uniform(0, 12, n)
    sleep = rng.uniform(3, 10, n)
    score = 85 - 2.5 * social + 3 * (sleep - 6) + rng.normal(0, 6, n)
    return pd.DataFrame(
        {
            "Student_ID": [f"STU_{i:05d}" for i in range(n)],
            "Age": rng.integers(13, 26, n),
            "Gender": rng.choice(["Male", "Female", "Non-binary"], n),
            "Education_Level": rng.choice(["High School", "College", "University"], n),
            "Daily_Social_Media_Hours": social,
            "Daily_AI_Tool_Usage_Hours": rng.uniform(0, 6, n),
            "Sleep_Hours": sleep,
            "Physical_Activity_Hours": rng.uniform(0, 4, n),
            "Mental_Health_Score": score.clip(0, 100),
            "Physical_Health_Score": rng.uniform(50, 100, n),
        }
    )


@pytest.fixture()
def client(tmp_path):
    data = tmp_path / "data.csv"
    _synthetic().to_csv(data, index=False)
    settings = Settings(data_path=data, models_dir=tmp_path / "models")
    with TestClient(create_app(settings)) as c:  # `with` => lifespan exécuté
        yield c


@pytest.fixture()
def student():
    return {
        "Age": 17, "Gender": "Female", "Education_Level": "High School",
        "Daily_Social_Media_Hours": 8.0, "Daily_AI_Tool_Usage_Hours": 2.9,
        "Sleep_Hours": 4.0, "Physical_Activity_Hours": 0.0, "Physical_Health_Score": 70.0,
    }
