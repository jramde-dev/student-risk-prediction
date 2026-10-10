"""Feature engineering du notebook, intégré au pipeline pour que le modèle
sauvegardé accepte directement les colonnes brutes.

ATTENTION : joblib sérialise `add_features` par référence (app.features).
Le modèle doit donc être rechargé avec ce module importable.
"""
import pandas as pd

RAW_NUMERIC = [
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Physical_Health_Score",
]
RAW_CATEGORICAL = ["Gender", "Education_Level"]
RAW_FEATURES = ["Age", "Gender", "Education_Level"] + RAW_NUMERIC[1:]

ENGINEERED = [
    "Total_Screen_Time",
    "Digital_Balance",
    "Sleep_Deficit",
    "Activity_Ratio",
    "Screen_Sleep_Ratio",
    "Heavy_Social_Media",
    "Sleep_Deprived",
]
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED
CATEGORICAL_FEATURES = RAW_CATEGORICAL

TARGET_REG = "Mental_Health_Score"
TARGET_CLF = "at_risk"


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Reproduit exactement la section 2.2 du notebook (sans Age_Group, écartée)."""
    X = X.copy()
    social, ai = X["Daily_Social_Media_Hours"], X["Daily_AI_Tool_Usage_Hours"]
    sleep, activity = X["Sleep_Hours"], X["Physical_Activity_Hours"]

    X["Total_Screen_Time"] = social + ai
    X["Digital_Balance"] = social - ai
    X["Sleep_Deficit"] = (8 - sleep).clip(lower=0)
    X["Activity_Ratio"] = activity / (sleep + 1)
    X["Screen_Sleep_Ratio"] = X["Total_Screen_Time"] / (sleep + 0.1)
    X["Heavy_Social_Media"] = (social > 6).astype(int)
    X["Sleep_Deprived"] = (sleep < 6).astype(int)
    return X
