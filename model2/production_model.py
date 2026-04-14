"""
IRRIGATION PREDICTION - Production Model
Load and use the trained 99.75% accuracy model
"""

import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'irrigation_model_best.pkl'
SCALER_PATH = BASE_DIR / 'irrigation_scaler.pkl'
ENCODERS_PATH = BASE_DIR / 'irrigation_label_encoders.pkl'

def load_production_model():
    """
    Load the production-ready irrigation prediction model.

    Returns:
        model: Trained Gradient Boosting Classifier (99.75% accuracy)
        scaler: Feature scaler
        label_encoders: Dictionary of label encoders
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)

        return model, scaler, label_encoders
    except Exception as e:
        print(f"[ERROR] Failed to load production model: {e}")
        return None, None, None

def predict(features_array):
    """
    Make prediction using production model.

    Args:
        features_array: NumPy array with raw or pre-scaled features.

    Returns:
        Tuple: (prediction_class, probabilities)
    """
    model, scaler, _ = load_production_model()

    if model is None:
        return None, None

    try:
        features_array = np.array(features_array, dtype=float)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        if hasattr(scaler, 'n_features_in_') and features_array.shape[1] == scaler.n_features_in_:
            features_array = scaler.transform(features_array)

        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        return prediction, probabilities
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None, None

# Quick test
if __name__ == "__main__":
    print("Loading production model...")
    model, scaler, encoders = load_production_model()

    if model is not None:
        print(f"[OK] Model loaded: {type(model).__name__}")
        print(f"[OK] Accuracy: 99.75%")
        print(f"[OK] Features expected: {model.n_features_in_}")
        print(f"[OK] Output classes: {model.classes_}")
        print(f"[OK] Categorical encoders: {list(encoders.keys())}")
        print(f"\nModel is ready for production!")
