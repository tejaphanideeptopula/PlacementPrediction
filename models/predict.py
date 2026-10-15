
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "placement_model.joblib"
)

def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run python -m models.train_model first."
        )
    return joblib.load(MODEL_PATH)


def get_feature_info():
    package = load_prediction_model()

    return {
        "numeric_cols": package["numeric_cols"],
        "categorical_cols": package["categorical_cols"],
        "categories": package["categories"]
    }


def predict_student(student_data):
    package = load_prediction_model()

    model = package["model"]
    numeric_cols = package["numeric_cols"]
    feature_names = package["feature_names"]

    row = {}

    for feature in feature_names:
        value = student_data.get(feature, "")

        if feature in numeric_cols:
            row[feature] = (
                float(value) if value != "" else float("nan")
            )
        else:
            row[feature] = value if value != "" else None

    sample = pd.DataFrame([row], columns=feature_names)

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "prediction": str(prediction),
        "confidence": confidence
    }
