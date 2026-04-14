# Soil Condition Prediction Model

A soil classification pipeline with XGBoost plus explainable predictions using SHAP.

## Features

- **Input Features**: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B
- **Output**: Soil condition classification (Good, Bad, Medium)
- **Model**: XGBoost Classifier
- **Explainability**: SHAP-based feature impact explanations

## Project Structure

```
Soil_analysis/
├── Data/
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── src/
│   └── model_utils.py
├── main.py
├── predict.py
├── interactive_predict.py
├── debug_predict.py
├── predict_xai.py
├── xai_explainer.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python main.py
```

This will:
1. Load `Data/dataset1.csv`
2. Preprocess and scale features
3. Apply SMOTE to balance training classes
4. Train XGBoost, RandomForest, and GradientBoosting
5. Save the best model and preprocessing artifacts in `models/`

### Make predictions with SHAP explanations

```bash
python predict_xai.py
```

This will:
1. Prompt for 12 soil feature values
2. Scale the input using the saved scaler
3. Predict the soil class with the saved model
4. Produce a SHAP explanation for top feature impacts

### Run the simple sample predictor

```bash
python predict.py
```

### Run the Flask API and dashboard

1. Start the backend API and serve the dashboard from the same server:

```bash
python api/app.py
```

2. Open the dashboard in your browser:

```bash
http://127.0.0.1:5000/
```

If you prefer to open the HTML file directly, use `frontend/dashboard.html` in your browser or serve the folder locally:

```bash
cd frontend
python -m http.server 8000
```

Then open:

```bash
http://127.0.0.1:8000/dashboard.html
```

The dashboard includes both soil classification and irrigation recommendation outputs using the combined prediction endpoint.

## Production Model2

The `model2/` folder contains a separate production-ready model implementation.

Required files:
- `model2/irrigation_model_best.pkl`
- `model2/irrigation_scaler.pkl`
- `model2/irrigation_label_encoders.pkl`

To test it:

```bash
python model2/production_model.py
```

Or import it from the project root:

```python
from model2 import load_production_model, predict

model, scaler, encoders = load_production_model()
features = numpy.array([[...]], dtype=float)
prediction, probabilities = predict(features)
```

> `model2` expects a pre-scaled feature vector of 26 values for its production workflow.

## Input Format

The model expects sensor readings in this order:
```
[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
```

Example:
```python
input_data = [[150, 7, 400, 7.2, 0.5, 0.7, 12, 0.3, 0.5, 1.0, 3, 1.5]]
```

## Explanation Output

`predict_xai.py` prints a ranked list of feature contributions with:
- Positive impact features
- Negative impact features
- SHAP magnitude values

## Notes

- SHAP is the core XAI layer used for feature attribution.
- The saved artifacts are loaded from `models/`.
- If `shap` is not installed, run `pip install shap`.
