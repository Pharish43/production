from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from predict_crop import predict_crop

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(str(FRONTEND_DIR), 'web_interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({'error': f"Missing fields: {', '.join(missing_fields)}."}), 400

    try:
        crop, confidence = predict_crop(
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
