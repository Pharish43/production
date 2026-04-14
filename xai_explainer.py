import pickle
from pathlib import Path
import numpy as np

FEATURE_NAMES = [
    "N", "P", "K", "pH", "EC", "OC",
    "S", "Zn", "Fe", "Cu", "Mn", "B"
]

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.pkl"
SCALER_PATH = Path(__file__).resolve().parent / "models" / "scaler.pkl"
ENCODER_PATH = Path(__file__).resolve().parent / "models" / "label_encoder.pkl"


def load_artifacts():
    """Load the saved model, scaler, and label encoder."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


def _create_shap_explainer(model):
    """Create a SHAP explainer for the saved tree model."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for XAI explanations. Install with `pip install shap`"
        ) from exc

    return shap.TreeExplainer(model)


CLASS_LABELS = {
    0: "Good",
    1: "Bad",
    2: "Medium",
}


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def explain_input(model, scaler, label_encoder, input_data):
    """Return prediction, probability, and SHAP-based feature explanation."""
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = _to_python_scalar(label_encoder.inverse_transform(prediction)[0])
    if isinstance(predicted_label, int) and predicted_label in CLASS_LABELS:
        predicted_label = CLASS_LABELS[predicted_label]

    probabilities = None
    try:
        probabilities = [float(x) for x in model.predict_proba(input_scaled)[0]]
    except Exception:
        probabilities = None

    explainer = _create_shap_explainer(model)
    shap_values = explainer(input_scaled)

    values = getattr(shap_values, "values", None)
    if values is None:
        raise RuntimeError("Could not extract SHAP values from the explainer output.")

    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            if values.shape[0] == 1:
                values = values[0, :, prediction[0]]
            else:
                values = values[prediction[0]][0]
        elif values.ndim == 2:
            values = values[0]
        else:
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}")
    else:
        values = np.array(values).flatten()

    explanation = []
    for name, value in zip(FEATURE_NAMES, values):
        explanation.append({
            "feature": name,
            "impact": float(value),
            "direction": "Positive" if value >= 0 else "Negative"
        })

    explanation = sorted(explanation, key=lambda item: -abs(item["impact"]))

    return {
        "prediction": predicted_label,
        "probabilities": probabilities,
        "shap_values": explanation,
    }


def format_explanation_report(result):
    """Format a human-readable explanation summary for display."""
    summary = []
    summary.append(f"Prediction: {result['prediction']}")
    if result["probabilities"] is not None:
        proba_lines = [
            f"  Class {idx}: {prob:.4f}" for idx, prob in enumerate(result["probabilities"])
        ]
        summary.append("Prediction probabilities:")
        summary.extend(proba_lines)

    summary.append("\nTop feature impacts:")
    for item in result["shap_values"][:8]:
        summary.append(
            f"  {item['feature']}: {item['direction']} impact ({item['impact']:+.4f})"
        )

    return "\n".join(summary)


def explain_values(input_values):
    """Run prediction and SHAP explanation for a single numeric input row."""
    if len(input_values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(input_values)}."
        )

    model, scaler, label_encoder = load_artifacts()
    result = explain_input(model, scaler, label_encoder, np.array([input_values]))
    return result
