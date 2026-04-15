import os
import joblib
from typing import Any

import pandas as pd

MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")


def load_artifact(filename: str) -> Any:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}. Run src/train.py first.")
    return joblib.load(path)


def load_model() -> tuple[Any, dict[str, Any], Any | None]:
    model = load_artifact("crop_model.pkl")
    encoders = {}
    for name in ["crop_ID", "soil_type", "Seedling_Stage"]:
        artifact_name = f"le_{name}.pkl"
        if os.path.exists(os.path.join(MODEL_DIR, artifact_name)):
            encoders[name] = load_artifact(artifact_name)

    target_encoder = None
    if os.path.exists(os.path.join(MODEL_DIR, "le_target.pkl")):
        target_encoder = load_artifact("le_target.pkl")

    return model, encoders, target_encoder


def predict_crop(crop: str, soil: str, stage: str, moi: float, temp: float, humidity: float) -> str:
    model, encoders, target_encoder = load_model()

    mapped_values = {}
    for label, value in [("crop_ID", crop), ("soil_type", soil), ("Seedling_Stage", stage)]:
        encoder = encoders.get(label)
        if encoder is None:
            raise ValueError(f"Missing encoder for {label}. Ensure training completed successfully.")

        if str(value) not in encoder.classes_.astype(str):
            raise ValueError(f"Unknown category '{value}' for {label}.")

        mapped_values[label] = int(encoder.transform([str(value)])[0])

    input_df = pd.DataFrame([
        {
            "crop ID": mapped_values["crop_ID"],
            "soil_type": mapped_values["soil_type"],
            "Seedling Stage": mapped_values["Seedling_Stage"],
            "MOI": float(moi),
            "temp": float(temp),
            "humidity": float(humidity),
        }
    ])

    prediction = model.predict(input_df)[0]
    if target_encoder is not None:
        prediction = target_encoder.inverse_transform([prediction])[0]

    if str(prediction) in {"1", "yes", "true", "Irrigation Needed", "irrigation needed"}:
        return "irrigation needed"
    return "irrigation no need"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict irrigation need for a crop sample.")
    parser.add_argument("--crop", default="Wheat", help="Crop name, e.g. Wheat")
    parser.add_argument("--soil", default="Black Soil", help="Soil type, e.g. Black Soil")
    parser.add_argument("--stage", default="Germination", help="Seedling stage, e.g. Germination")
    parser.add_argument("--moi", type=float, default=3.0, help="MOI value")
    parser.add_argument("--temp", type=float, default=27.0, help="Temperature value")
    parser.add_argument("--humidity", type=float, default=75.0, help="Humidity value")
    args = parser.parse_args()

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
