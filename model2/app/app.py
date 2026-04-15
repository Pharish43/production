import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

from flask import Flask, jsonify, request
from predict import predict_crop

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(force=True)
    required_fields = ["crop", "soil", "stage", "moi", "temp", "humidity"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    prediction = predict_crop(
        crop=payload["crop"],
        soil=payload["soil"],
        stage=payload["stage"],
        moi=payload["moi"],
        temp=payload["temp"],
        humidity=payload["humidity"],
    )

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
