from pathlib import Path
import json
import joblib
import numpy as np
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "crop_model.pkl"

FEATURE_NAMES = [
    "crop ID",
    "soil_type",
    "Seedling Stage",
    "MOI",
    "temp",
    "humidity",
]


def load_artifact(filename: str) -> Any:
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}. Run src/train.py first.")
    return joblib.load(path)


def load_model() -> tuple[Any, dict[str, Any], Any | None]:
    model = load_artifact("crop_model.pkl")
    encoders = {}
    for name in ["crop_ID", "soil_type", "Seedling_Stage"]:
        artifact_name = f"le_{name}.pkl"
        artifact_path = MODEL_DIR / artifact_name
        if artifact_path.exists():
            encoders[name] = load_artifact(artifact_name)

    target_encoder = None
    target_path = MODEL_DIR / "le_target.pkl"
    if target_path.exists():
        target_encoder = load_artifact("le_target.pkl")

    return model, encoders, target_encoder


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _create_shap_explainer(model):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for XAI explanations. Install with `pip install shap`."
        ) from exc
    return shap.TreeExplainer(model)


def _format_shap_values(shap_values, prediction_index=0):
    values = getattr(shap_values, "values", None)
    if values is None:
        values = np.array(shap_values).flatten()

    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            if values.shape[0] == 1:
                values = values[0, :, prediction_index]
            else:
                values = values[prediction_index][0]
        elif values.ndim == 2:
            values = values[0]
        values = np.array(values).flatten()
    else:
        values = np.array(values).flatten()

    return values


def _build_input_df(crop: str, soil: str, stage: str, moi: float, temp: float, humidity: float) -> pd.DataFrame:
    model, encoders, target_encoder = load_model()

    mapped_values = {}
    for label, value in [("crop_ID", crop), ("soil_type", soil), ("Seedling_Stage", stage)]:
        encoder = encoders.get(label)
        if encoder is None:
            raise ValueError(f"Missing encoder for {label}. Ensure training completed successfully.")

        if str(value) not in encoder.classes_.astype(str):
            raise ValueError(f"Unknown category '{value}' for {label}.")

        mapped_values[label] = int(encoder.transform([str(value)])[0])

    return pd.DataFrame([
        {
            "crop ID": mapped_values["crop_ID"],
            "soil_type": mapped_values["soil_type"],
            "Seedling Stage": mapped_values["Seedling_Stage"],
            "MOI": float(moi),
            "temp": float(temp),
            "humidity": float(humidity),
        }
    ])


def predict_crop(crop: str, soil: str, stage: str, moi: float, temp: float, humidity: float, explain: bool = False) -> str | dict[str, Any]:
    input_df = _build_input_df(crop, soil, stage, moi, temp, humidity)

    model, _, target_encoder = load_model()
    prediction_raw = model.predict(input_df)[0]
    prediction = prediction_raw
    if target_encoder is not None:
        prediction = target_encoder.inverse_transform([prediction_raw])[0]

    if explain:
        probabilities = None
        try:
            probabilities = [float(x) for x in model.predict_proba(input_df)[0]]
        except Exception:
            probabilities = None

        explainer = _create_shap_explainer(model)
        shap_values = explainer(input_df)
        values = _format_shap_values(
            shap_values,
            prediction_index=prediction_raw if isinstance(prediction_raw, int) else 0,
        )

        explanation = [
            {
                "feature": FEATURE_NAMES[i],
                "impact": float(values[i]),
                "direction": "Positive" if values[i] >= 0 else "Negative",
            }
            for i in range(len(FEATURE_NAMES))
        ]
        explanation = sorted(explanation, key=lambda item: -abs(item["impact"]))

        return {
            "prediction": str(prediction),
            "water_needed": "Water needed" if str(prediction).strip().lower() in {"1", "yes", "true", "irrigation needed"} else "No water needed",
            "input": {
                "crop": crop,
                "soil": soil,
                "stage": stage,
                "moi": float(moi),
                "temp": float(temp),
                "humidity": float(humidity),
            },
            "probabilities": probabilities,
            "shap_values": explanation,
        }

    if str(prediction) in {"1", "yes", "true", "Irrigation Needed", "irrigation needed"}:
        return "irrigation needed"
    return "irrigation no need"


def explain_crop(crop: str, soil: str, stage: str, moi: float, temp: float, humidity: float) -> dict[str, Any]:
    result = predict_crop(
        crop=crop,
        soil=soil,
        stage=stage,
        moi=moi,
        temp=temp,
        humidity=humidity,
        explain=True,
    )
    return result  # type: ignore[return-value]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict irrigation need for a crop sample.")
    parser.add_argument("--crop", default="Wheat", help="Crop name, e.g. Wheat")
    parser.add_argument("--soil", default="Black Soil", help="Soil type, e.g. Black Soil")
    parser.add_argument("--stage", default="Germination", help="Seedling stage, e.g. Germination")
    parser.add_argument("--moi", type=float, default=3.0, help="MOI value")
    parser.add_argument("--temp", type=float, default=27.0, help="Temperature value")
    parser.add_argument("--humidity", type=float, default=75.0, help="Humidity value")
    parser.add_argument("--explain", action="store_true", help="Return SHAP explanation for the input.")
    args = parser.parse_args()

    if args.explain:
        explanation = explain_crop(
            crop=args.crop,
            soil=args.soil,
            stage=args.stage,
            moi=args.moi,
            temp=args.temp,
            humidity=args.humidity,
        )
        print(json.dumps(explanation, indent=2))
    else:
        prediction = predict_crop(
            crop=args.crop,
            soil=args.soil,
            stage=args.stage,
            moi=args.moi,
            temp=args.temp,
            humidity=args.humidity,
        )
        print(prediction)


if __name__ == "__main__":
    main()
