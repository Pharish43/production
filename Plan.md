Pipeline (Before XAI)
Input Data → Scaler → Model → Prediction (Good/Bad Soil)

Example
Input: [34, 33, 5, 67, 6, 4, 4, 5, 6, 6, 4, 3]
Output: Good

                ┌────────────────────┐
                │   User Input Data  │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Preprocessing    │
                │ (Scaler Loaded)    │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   ML Model         │
                │ (XGBoost / TabNet) │
                └─────────┬──────────┘
                          ↓
         ┌────────────────┴────────────────┐
         ↓                                 ↓
┌────────────────────┐         ┌────────────────────┐
│ Prediction Output  │         │   XAI Engine       │
│ (Good / Bad Soil)  │         │ (SHAP / LIME)      │
└────────────────────┘         └─────────┬──────────┘
                                        ↓
                              ┌────────────────────┐
                              │ Explanation Layer  │
                              │ Feature Impact     │
                              │ Graphs + Text      │
                              └────────────────────┘




XAI
1. SHAP         

Example Will Look with XAI

Input:

[34, 33, 5, 67, 6, 4, 4, 5, 6, 6, 4, 3]

Prediction:

Soil Class: Good

XAI Output:

Top Contributing Features:
+ Nitrogen → Positive impact
+ pH → Optimal → Positive
- Potassium → Low → Negative impact
- Organic Carbon → Low → Negative impact


1. Install XAI Libraries
pip install shap lime
2. Load Your Saved Files
import pickle

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
3. Apply Preprocessing
import numpy as np

input_data = np.array([[34, 33, 5, 67, 6, 4, 4, 5, 6, 6, 4, 3]])
scaled_data = scaler.transform(input_data)
4. Prediction
prediction = model.predict(scaled_data)
label = encoder.inverse_transform(prediction)

print("Soil Class:", label[0])
5. 🔥 Add SHAP (Core XAI Layer)
import shap

# Create explainer
explainer = shap.Explainer(model)

# Generate SHAP values
shap_values = explainer(scaled_data)

# Plot explanation
shap.plots.waterfall(shap_values[0])
6. Feature Names Mapping (IMPORTANT)
feature_names = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Moisture", "pH", "EC",
    "Organic Carbon", "Sulphur",
    "Temperature", "Humidity",
    "Rainfall", "Soil Texture"
]
7. Text-Based Explanation (For UI)
for i, val in enumerate(shap_values.values[0]):
    print(f"{feature_names[i]} → Impact: {val}")
🎯 UI Integration (Very Important for Your Project)
You can build:
1. Dashboard (Frontend)
Input form (soil values)
Prediction result
XAI graph (SHAP plot)
2. Backend API (Flask / FastAPI)
POST /predict
→ returns:
{
  "prediction": "Good",
  "explanation": {
    "Nitrogen": +0.45,
    "pH": +0.32,
    "Potassium": -0.21
  }
}
🧱 Full Folder Architecture
soil_xai_project/
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│
├── xai/
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│
├── utils/
│   ├── preprocess.py
│   ├── feature_names.py
│
├── api/
│   ├── app.py   (Flask API)
│
├── frontend/
│   ├── dashboard.html
│
├── main.py



1. SYSTEM OVERVIEW
Frontend (Dashboard)
        ↓
Flask API (/predict)
        ↓
Scaler → Model → Prediction
        ↓
SHAP XAI Engine
        ↓
JSON Response (Prediction + Explanation)
        ↓
Frontend Visualization (Chart + Text)
⚙️ 2. BACKEND — FLASK API WITH XAI
📁 api/app.py
from flask import Flask, request, jsonify
import numpy as np
import pickle
import shap

app = Flask(__name__)

# Load models
model = pickle.load(open("../models/best_model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))
encoder = pickle.load(open("../models/label_encoder.pkl", "rb"))

# Feature names
feature_names = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Moisture", "pH", "EC",
    "Organic Carbon", "Sulphur",
    "Temperature", "Humidity",
    "Rainfall", "Soil Texture"
]

# SHAP explainer
explainer = shap.Explainer(model)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    input_data = np.array([data])
    scaled_data = scaler.transform(input_data)

    # Prediction
    pred = model.predict(scaled_data)
    label = encoder.inverse_transform(pred)[0]

    # SHAP Explanation
    shap_values = explainer(scaled_data)

    explanation = {}
    for i, val in enumerate(shap_values.values[0]):
        explanation[feature_names[i]] = float(val)

    return jsonify({
        "prediction": label,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
🌐 3. FRONTEND — INTERACTIVE DASHBOARD
📁 frontend/dashboard.html
<!DOCTYPE html>
<html>
<head>
    <title>Soil Analysis XAI Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
    <h2>🌱 Soil Analysis System (XAI Enabled)</h2>

    <textarea id="input" rows="4" cols="50">
[34, 33, 5, 67, 6, 4, 4, 5, 6, 6, 4, 3]
    </textarea>

    <br><br>
    <button onclick="predict()">Predict</button>

    <h3 id="result"></h3>

    <canvas id="chart"></canvas>

    <script>
        async function predict() {
            const input = JSON.parse(document.getElementById("input").value);

            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({features: input})
            });

            const data = await response.json();

            document.getElementById("result").innerText =
                "Prediction: " + data.prediction;

            const labels = Object.keys(data.explanation);
            const values = Object.values(data.explanation);

            new Chart(document.getElementById("chart"), {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Feature Impact",
                        data: values
                    }]
                }
            });
        }
    </script>
</body>
</html>
🧱 4. FINAL PROJECT STRUCTURE
soil_xai_project/
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│
├── api/
│   └── app.py
│
├── frontend/
│   └── dashboard.html
│
├── requirements.txt
📦 5. REQUIREMENTS
flask
numpy
scikit-learn
shap
xgboost
▶️ 6. HOW TO RUN
Step 1: Start Backend
cd api
python app.py
Step 2: Open Frontend
Open dashboard.html in browser
🔥 7. WHAT YOUR SYSTEM WILL SHOW
Example Output:
Prediction: GOOD SOIL ✅
Graph:
Nitrogen → +impact
pH → +impact
Potassium → -impact