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
from model2.src.predict import explain_crop, predict_crop
from model3.predict_crop import predict_crop as model3_predict_crop
from model3.predict_crop import predict_crop as model3_predict_crop

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

MODEL2_REQUIRED_CATEGORICALS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

MODEL2_OPTIONAL_NUMERIC_FIELDS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

MODEL2_DEFAULT_NUMERIC_VALUES = {
    "Soil_pH": 6.487857,
    "Soil_Moisture": 36.969207,
    "Organic_Carbon": 0.944731,
    "Electrical_Conductivity": 1.791963,
    "Temperature_C": 26.991423,
    "Humidity": 60.080339,
    "Rainfall_mm": 1252.49942,
    "Sunlight_Hours": 7.518538,
    "Wind_Speed_kmh": 10.163545,
    "Field_Area_hectare": 7.598024,
    "Previous_Irrigation_mm": 59.864122,
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


def _build_model2_feature_array(features, model2_encoder):
    if isinstance(features, list):
        if len(features) != 26:
            return None, f"Model2 prediction failed: a raw feature list must contain 26 values, got {len(features)}."
        try:
            return np.array([features], dtype=float), None
        except ValueError:
            return None, "All feature values must be numeric."

    if not isinstance(features, dict):
        return None, "Model2 prediction requires either a 26-value list or a feature object with the required categorical fields."

    missing = [
        k for k in MODEL2_REQUIRED_CATEGORICALS if k not in features or features[k] is None
    ]
    if missing:
        return None, f"Missing required model2 categorical fields: {', '.join(missing)}."

    try:
        soil_type = features["Soil_Type"]
        soil_classes = list(model2_encoder["Soil_Type"].classes_)
        if soil_type not in soil_classes:
            raise ValueError(f"Unknown Soil_Type: {soil_type}")
        soil_vector = [1.0 if soil_type == cls else 0.0 for cls in soil_classes]

        region = features["Region"]
        region_classes = list(model2_encoder["Region"].classes_)
        if region not in region_classes:
            raise ValueError(f"Unknown Region: {region}")
        region_vector = [1.0 if region == cls else 0.0 for cls in region_classes]

        crop_type = float(model2_encoder["Crop_Type"].transform([features["Crop_Type"]])[0])
        growth_stage = float(model2_encoder["Crop_Growth_Stage"].transform([features["Crop_Growth_Stage"]])[0])
        season = float(model2_encoder["Season"].transform([features["Season"]])[0])
        irrigation_type = float(model2_encoder["Irrigation_Type"].transform([features["Irrigation_Type"]])[0])
        water_source = float(model2_encoder["Water_Source"].transform([features["Water_Source"]])[0])
        mulching_used = float(model2_encoder["Mulching_Used"].transform([features["Mulching_Used"]])[0])

        numeric_values = []
        for field in MODEL2_OPTIONAL_NUMERIC_FIELDS:
            value = features.get(field, MODEL2_DEFAULT_NUMERIC_VALUES[field])
            numeric_values.append(float(value))

        feature_vector = (
            soil_vector
            + numeric_values
            + [
                crop_type,
                growth_stage,
                season,
                irrigation_type,
                water_source,
                mulching_used,
            ]
            + region_vector
        )

        if len(feature_vector) != 26:
            return None, f"Built model2 feature vector has wrong length {len(feature_vector)}; expected 26."

        return np.array([feature_vector], dtype=float), None
    except KeyError as exc:
        return None, f"Model2 prediction failed: unknown categorical field {exc.args[0]}."
    except ValueError as exc:
        return None, f"Model2 prediction failed: {exc}"


def _format_model2_response(features):
    try:
        model2_model, model2_scaler, model2_encoder = _load_model2()
    except Exception as exc:
        return None, None, None, str(exc)

    input_array, error = _build_model2_feature_array(features, model2_encoder)
    if error:
        return None, None, None, error

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


@app.route("/predict/model2/simple", methods=["POST"])
def predict_model2_simple():
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request JSON must be an object with crop, soil, stage, moi, temp, humidity."}), 400

    required_fields = ["crop", "soil", "stage", "moi", "temp", "humidity"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}."}), 400

    try:
        explanation = predict_crop(
            crop=payload["crop"],
            soil=payload["soil"],
            stage=payload["stage"],
            moi=payload["moi"],
            temp=payload["temp"],
            humidity=payload["humidity"],
            explain=True,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(explanation)


@app.route("/predict/model2", methods=["POST"])
def predict_model2():
    payload = request.get_json(force=True)
    features = payload.get("features")

    if not isinstance(features, (list, dict)):
        return jsonify({"error": "Request JSON must include 'features' as a list or object."}), 400

    prediction, probabilities, label, error = _format_model2_response(features)
    if error:
        return jsonify({"error": error}), 400

    if isinstance(label, int) and label in CLASS_LABELS:
        label = CLASS_LABELS[label]

    validation_report = {
        "status": "ok",
        "message": "Irrigation model2 processed the request successfully.",
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
@app.route("/dashboard", methods=["GET"])
def index():
    return send_from_directory(str(ROOT / 'frontend'), 'dashboard.html')


@app.route("/model2-demo", methods=["GET"])
def model2_demo():
    return send_from_directory(str(ROOT / 'frontend'), 'model2_demo.html')


@app.route("/model3", methods=["GET"])
@app.route("/crop-recommendation", methods=["GET"])
def model3_demo():
    return send_from_directory(str(ROOT / 'frontend'), 'web_interface.html')


@app.route("/model3/predict", methods=["POST"])
def model3_predict():
    payload = request.get_json(force=True)
    required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({'error': f"Missing fields: {', '.join(missing_fields)}."}), 400

    try:
        crop, confidence = model3_predict_crop(
            nitrogen=float(payload['nitrogen']),
            phosphorus=float(payload['phosphorus']),
            potassium=float(payload['potassium']),
            temperature=float(payload['temperature']),
            humidity=float(payload['humidity']),
            ph=float(payload['ph']),
            rainfall=float(payload['rainfall']),
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({'prediction': {'crop': crop, 'confidence': round(confidence, 2)}})


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
        <p>Send a JSON body with <code>features</code> as a list of numeric values.</p>
        <p>There are two supported prediction formats:</p>
        <ul>
            <li><code>/predict</code> and <code>/predict/combined</code>: 12 soil values in order <code>N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B</code>.</li>
            <li><code>/predict/model2</code>: 26 numeric features for the irrigation production model.</li>
        </ul>
        <p>Example request body for the irrigation model:</p>
        <pre>{
  "features": [0.0, 1.0, 0.5, 2.17, 21.9, 31.19, 1167.7, 4.01, 1.97, 1.0, 0.0, 4.0, 2.0, 0.0, 2.0, 4.73, 1.0, 1.98, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}</pre>
        <p>Use this route from any HTTP client when calling the irrigation model directly.</p>
    </body>
    </html>
    """
    return html, 200, {'Content-Type': 'text/html'}


if __name__ == "__main__":
    app.run(debug=True)


