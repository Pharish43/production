from pathlib import Path
import sys

# Ensure root project path is importable from api/
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from xai_explainer import FEATURE_NAMES, load_artifacts, _create_shap_explainer
from model2 import load_production_model as load_model2

CLASS_LABELS = {
    0: "Good",
    1: "Bad",
    2: "Medium",
}

FEATURE_RANGES = {
    "N": (0, 1000),
    "P": (0, 200),
    "K": (0, 1000),
    "pH": (0, 14),
    "EC": (0, 5),
    "OC": (0, 10),
    "S": (0, 100),
    "Zn": (0, 10),
    "Fe": (0, 50),
    "Cu": (0, 10),
    "Mn": (0, 50),
    "B": (0, 5),
}

CLASS_FEEDBACK = {
    0: "Good soil: nutrient levels are healthy and balanced.",
    1: "Bad soil: nutrient levels are poor and require immediate amendment.",
    2: "Medium soil: moderate condition with room for improvement.",
}

MEDIUM_CLASS_RANGES = {
    "N": (88, 383),
    "P": (5.3, 125),
    "K": (317, 887),
    "pH": (7.02, 11.15),
    "EC": (0.37, 1.27),
    "OC": (0.1, 1.07),
    "S": (2.5, 31),
    "Zn": (0.26, 2.08),
    "Fe": (0.31, 9.03),
    "Cu": (0.10, 1.55),
    "Mn": (0.26, 31),
    "B": (0.19, 2.12),
}

def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

app = Flask(__name__)
CORS(app)

model, scaler, encoder = load_artifacts()
explainer = _create_shap_explainer(model)


def _format_shap_values(shap_values, prediction_index):
    values = shap_values.values
    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            # SHAP returns shape (1, n_features, n_classes) for multiclass tree models
            if values.shape[0] == 1:
                values = values[0, :, prediction_index]
            else:
                # fallback for other SHAP layouts
                values = values[prediction_index][0]
        elif values.ndim == 2:
            values = values[0]
    else:
        values = np.array(values).flatten()
    return values


def _validate_medium_soil(input_array):
    issues = []
    for i, value in enumerate(input_array[0]):
        feature_name = FEATURE_NAMES[i]
        low, high = MEDIUM_CLASS_RANGES[feature_name]
        if not (low <= value <= high):
            issues.append(
                f"{feature_name}={value} is outside the expected Medium range ({low} - {high})."
            )
    if issues:
        return {
            "status": "warning",
            "message": "The predicted soil class is Medium, but some inputs fall outside the typical Medium distribution.",
            "issues": issues,
        }
    return {
        "status": "ok",
        "message": "The input values are consistent with Medium soil conditions.",
        "issues": [],
    }


def _load_model2():
    model2_model, model2_scaler, model2_encoder = load_model2()
    if model2_model is None:
        raise RuntimeError("Failed to load model2 artifacts.")
    return model2_model, model2_scaler, model2_encoder


def _format_model2_response(features):
    try:
        model2_model, model2_scaler, model2_encoder = _load_model2()
    except Exception as exc:
        return None, None, None, str(exc)

    try:
        input_array = np.array([features], dtype=float)
    except ValueError:
        return None, None, None, "All feature values must be numeric."

    try:
        if hasattr(model2_scaler, 'n_features_in_') and input_array.shape[1] == model2_scaler.n_features_in_:
            input_array = model2_scaler.transform(input_array)
    except Exception as exc:
        return None, None, None, f"Model2 scaling failed: {exc}"

    try:
        prediction = model2_model.predict(input_array)
        probabilities = [float(x) for x in model2_model.predict_proba(input_array)[0]]
    except Exception as exc:
        return None, None, None, f"Model2 prediction failed: {exc}"

    try:
        label = model2_encoder.inverse_transform(prediction)[0]
    except Exception:
        label = prediction[0]

    return prediction, probabilities, label, None


def _water_needed_text(label):
    label_norm = str(label).strip().lower()
    if label_norm in {"yes", "y", "true", "1", "water", "water needed", "needed", "need water", "need"}:
        return "Water needed"
    if label_norm in {"no", "n", "false", "0", "not needed", "no water", "no water needed", "not required"}:
        return "No water needed"
    if label_norm in {"good", "bad", "medium"}:
        return "Water recommendation unavailable"
    return str(label)


def _validate_features(features):
    if not isinstance(features, list):
        return None, "Request JSON must include 'features' as a list."
    if len(features) != len(FEATURE_NAMES):
        return None, f"Expected {len(FEATURE_NAMES)} features, got {len(features)}."
    try:
        input_array = np.array([features], dtype=float)
    except ValueError:
        return None, "All feature values must be numeric."
    return input_array, None


def _run_soil_prediction(input_array):
    scaled = scaler.transform(input_array)
    pred = model.predict(scaled)
    label = encoder.inverse_transform(pred)[0]
    label = _to_python_scalar(label)
    if isinstance(label, int) and label in CLASS_LABELS:
        label = CLASS_LABELS[label]

    probabilities = None
    try:
        probabilities = [float(x) for x in model.predict_proba(scaled)[0]]
    except Exception:
        probabilities = None

    if pred[0] == 2:
        validation_report = _validate_medium_soil(input_array)
    else:
        validation_report = {
            "status": "ok",
            "message": CLASS_FEEDBACK[pred[0]],
            "issues": [],
        }

    shap_values = explainer(scaled)
    values = _format_shap_values(shap_values, pred[0])
    explanation = {FEATURE_NAMES[i]: float(values[i]) for i in range(len(FEATURE_NAMES))}

    return {
        "prediction": label,
        "probabilities": probabilities,
        "explanation": explanation,
        "validation": validation_report,
        "advice": CLASS_FEEDBACK[pred[0]],
    }


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    features = payload.get("features")

    if not isinstance(features, list):
        return jsonify({"error": "Request JSON must include 'features' as a list."}), 400

    if len(features) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}."
        }), 400

    try:
        input_array = np.array([features], dtype=float)
    except ValueError:
        return jsonify({"error": "All feature values must be numeric."}), 400

    validation_issues = []
    for i, value in enumerate(input_array[0]):
        feature_name = FEATURE_NAMES[i]
        low, high = FEATURE_RANGES[feature_name]
        if not (low <= value <= high):
            validation_issues.append(
                f"{feature_name}={value} is outside realistic range {low}-{high}."
            )

    if validation_issues:
        return jsonify({
            "error": "One or more feature values are outside realistic ranges.",
            "details": validation_issues,
        }), 400

    scaled = scaler.transform(input_array)
    pred = model.predict(scaled)
    label = encoder.inverse_transform(pred)[0]
    label = _to_python_scalar(label)
    if isinstance(label, int) and label in CLASS_LABELS:
        label = CLASS_LABELS[label]

    probabilities = None
    try:
        probabilities = [float(x) for x in model.predict_proba(scaled)[0]]
    except Exception:
        probabilities = None

    validation_report = None
    if pred[0] == 2:
        validation_report = _validate_medium_soil(input_array)
    else:
        validation_report = {
            "status": "ok",
            "message": CLASS_FEEDBACK[pred[0]],
            "issues": [],
        }

    shap_values = explainer(scaled)
    values = _format_shap_values(shap_values, pred[0])

    explanation = {
        FEATURE_NAMES[i]: float(values[i]) for i in range(len(FEATURE_NAMES))
    }

    response = {
        "prediction": label,
        "probabilities": probabilities,
        "explanation": explanation,
        "features": features,
        "validation": validation_report,
        "advice": CLASS_FEEDBACK[pred[0]],
    }

    return jsonify(response)


@app.route("/predict/combined", methods=["POST"])
@app.route("/predict-with-explain", methods=["POST"])
def predict_combined():
    payload = request.get_json(force=True)
    features = payload.get("features")

    input_array, error = _validate_features(features)
    if error:
        return jsonify({"error": error}), 400

    soil_prediction = _run_soil_prediction(input_array)

    prediction, probabilities, label, model2_error = _format_model2_response(features)
    if model2_error:
        model2_response = {"error": model2_error}
    else:
        if isinstance(label, int) and label in CLASS_LABELS:
            label = CLASS_LABELS[label]
        validation_report = None
        if prediction[0] == 2:
            validation_report = _validate_medium_soil(input_array)
        else:
            validation_report = {
                "status": "ok",
                "message": CLASS_FEEDBACK[prediction[0]],
                "issues": [],
            }
        model2_response = {
            "prediction": label,
            "water_needed": _water_needed_text(label),
            "probabilities": probabilities,
            "validation": validation_report,
            "model": "model2",
            "advice": CLASS_FEEDBACK[prediction[0]],
        }

    return jsonify({
        "soil": soil_prediction,
        "irrigation": model2_response,
    })


@app.route("/predict/model2", methods=["POST"])
def predict_model2():
    payload = request.get_json(force=True)
    features = payload.get("features")

    if not isinstance(features, list):
        return jsonify({"error": "Request JSON must include 'features' as a list."}), 400

    if len(features) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}."
        }), 400

    prediction, probabilities, label, error = _format_model2_response(features)
    if error:
        return jsonify({"error": error}), 400

    if isinstance(label, int) and label in CLASS_LABELS:
        label = CLASS_LABELS[label]

    validation_report = None
    if prediction[0] == 2:
        validation_report = _validate_medium_soil(np.array([features], dtype=float))
    else:
        validation_report = {
            "status": "ok",
            "message": CLASS_FEEDBACK[prediction[0]],
            "issues": [],
        }

    response = {
        "prediction": label,
        "probabilities": probabilities,
        "validation": validation_report,
        "model": "model2",
        "advice": CLASS_FEEDBACK[prediction[0]],
    }

    return jsonify(response)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(ROOT / 'frontend'), 'dashboard.html')


@app.route("/predict", methods=["GET"])
def predict_info():
    html = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Soil XAI Predict Endpoint</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
            pre { background: #f9f9f9; padding: 16px; border-radius: 6px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Soil XAI Predict Endpoint</h1>
        <p>This endpoint accepts <strong>POST</strong> requests only.</p>
        <p>Send a JSON body with <code>features</code> as a list of 12 numeric values in this order:</p>
        <ol>
            <li>N</li>
            <li>P</li>
            <li>K</li>
            <li>pH</li>
            <li>EC</li>
            <li>OC</li>
            <li>S</li>
            <li>Zn</li>
            <li>Fe</li>
            <li>Cu</li>
            <li>Mn</li>
            <li>B</li>
        </ol>
        <p>Example request body:</p>
        <pre>{
  "features": [270, 9.9, 444, 7.63, 0.40, 0.86, 11.8, 0.25, 0.76, 1.69, 2.43, 2.26]
}</pre>
        <p>Use this route from the dashboard or any HTTP client.</p>
    </body>
    </html>
    """
    return html, 200, {'Content-Type': 'text/html'}


if __name__ == "__main__":
    app.run(debug=True)


