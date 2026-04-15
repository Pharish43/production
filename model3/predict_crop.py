"""
Simple interface to predict crop from land parameters
This script loads the trained model and makes predictions
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_model():
    """Load the trained model and preprocessors"""
    with open(ROOT / 'crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(ROOT / 'crop_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(ROOT / 'crop_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Predict the best crop for given land parameters

    Args:
        nitrogen (float): Nitrogen content in soil (0-140)
        phosphorus (float): Phosphorus content in soil (5-145)
        potassium (float): Potassium content in soil
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage (0-100)
        ph (float): pH level of soil (3.5-9.9)
        rainfall (float): Rainfall in mm

    Returns:
        tuple: (crop_name, confidence_percentage)
    """
    model, scaler, encoder = load_model()

    # Create all K-FOCUSED engineered features
    k_squared = potassium ** 2
    k_cubed = potassium ** 3
    k_log = np.log(potassium + 1)
    k_n_ratio = potassium / (nitrogen + 1e-6)
    k_p_ratio = potassium / (phosphorus + 1e-6)
    high_k_indicator = float(potassium > 100)
    low_k_indicator = float(potassium <= 80)
    k_n_interaction = potassium * nitrogen
    k_p_interaction = potassium * phosphorus
    k_temp_interaction = potassium * temperature
    k_humidity_interaction = potassium * humidity
    k_rainfall_interaction = potassium * rainfall
    npk_product = nitrogen * phosphorus * potassium
    npk_max_ratio = potassium / max(nitrogen, phosphorus, potassium, 1e-6)
    temp_humidity = temperature * humidity
    ph_neutral_dist = np.abs(ph - 6.5)

    # Create DataFrame with correct feature names (K-focused)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
                     'K_squared', 'K_cubed', 'K_log', 'K_N_ratio', 'K_P_ratio',
                     'high_K_indicator', 'low_K_indicator', 'K_N_interaction', 'K_P_interaction',
                     'K_temp_interaction', 'K_humidity_interaction', 'K_rainfall_interaction',
                     'NPK_product', 'NPK_max_ratio', 'temp_humidity', 'pH_neutral_dist']

    feature_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall,
                      k_squared, k_cubed, k_log, k_n_ratio, k_p_ratio,
                      high_k_indicator, low_k_indicator, k_n_interaction, k_p_interaction,
                      k_temp_interaction, k_humidity_interaction, k_rainfall_interaction,
                      npk_product, npk_max_ratio, temp_humidity, ph_neutral_dist]

    # Create DataFrame with feature names
    input_data = pd.DataFrame([feature_values], columns=feature_names)

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Get prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Decode and get confidence
    crop_name = encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities)) * 100

    return crop_name, confidence


def interactive_prediction():
    """Interactive mode for crop prediction"""
    print("="*60)
    print("CROP RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nEnter your land parameters to predict suitable crops:\n")

    try:
        nitrogen = float(input("Nitrogen (N) [0-140]: "))
        phosphorus = float(input("Phosphorus (P) [5-145]: "))
        potassium = float(input("Potassium (K) [0-120]: "))
        temperature = float(input("Temperature (Celsius) [10-50]: "))
        humidity = float(input("Humidity (%) [0-100]: "))
        ph = float(input("pH level [3.5-9.9]: "))
        rainfall = float(input("Rainfall (mm) [20-300]: "))

        crop, confidence = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Recommended Crop: {crop.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*60)

    except ValueError:
        print("Error: Please enter valid numbers!")


if __name__ == "__main__":
    interactive_prediction()
