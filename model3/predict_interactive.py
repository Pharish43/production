"""
Interactive Crop Prediction with Sample Values
"""

import pickle
import numpy as np

def load_model():
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('crop_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('crop_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('crop_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, encoder, features

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    model, scaler, encoder, features = load_model()

    # Create input array with feature names
    input_dict = {
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    # Add ALL engineered features (must match training)
    input_dict['N_P_ratio'] = nitrogen / (phosphorus + 1e-6)
    input_dict['K_N_ratio'] = potassium / (nitrogen + 1e-6)
    input_dict['P_K_ratio'] = phosphorus / (potassium + 1e-6)
    input_dict['NPK_sum'] = nitrogen + phosphorus + potassium
    input_dict['NPK_avg'] = input_dict['NPK_sum'] / 3
    input_dict['NPK_ratio'] = (nitrogen + phosphorus + potassium) / (temperature + humidity + 1e-6)
    input_dict['temp_humidity'] = temperature * humidity
    input_dict['rainfall_temp'] = rainfall * temperature
    input_dict['pH_neutral_dist'] = abs(ph - 6.5)
    input_dict['hot_humid'] = 1 if (temperature > 25 and humidity > 75) else 0
    input_dict['moderate_rainfall'] = 1 if (rainfall > 100 and rainfall < 250) else 0

    # Create DataFrame in correct order
    import pandas as pd
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    crop_name = encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities)) * 100

    return crop_name, confidence, probabilities, encoder.classes_

def interactive_mode():
    print("="*70)
    print("CROP PREDICTION SYSTEM - INTERACTIVE MODE")
    print("="*70)

    while True:
        print("\nEnter your land parameters:")
        try:
            n = float(input("Nitrogen (N) [0-140]: "))
            p = float(input("Phosphorus (P) [5-145]: "))
            k = float(input("Potassium (K) [0-120]: "))
            temp = float(input("Temperature (Celsius) [10-50]: "))
            humidity = float(input("Humidity (%) [0-100]: "))
            ph = float(input("pH level [3.5-9.9]: "))
            rainfall = float(input("Rainfall (mm) [20-300]: "))

            crop, confidence, probs, classes = predict_crop(n, p, k, temp, humidity, ph, rainfall)

            print("\n" + "="*70)
            print(f"PREDICTION RESULT")
            print("="*70)
            print(f"Recommended Crop: {crop.upper()}")
            print(f"Confidence: {confidence:.2f}%")

            # Show top 3 predictions
            print(f"\nTop 3 Predictions:")
            top_indices = np.argsort(probs)[-3:][::-1]
            for idx in top_indices:
                print(f"  {classes[idx]:15} : {probs[idx]*100:6.2f}%")

            print("="*70)

            again = input("\nPredict another crop? (yes/no): ").lower()
            if again != 'yes' and again != 'y':
                break

        except ValueError:
            print("Error: Please enter valid numbers!")
        except Exception as e:
            print(f"Error: {e}")

def test_samples():
    print("="*70)
    print("CROP PREDICTION - TEST WITH SAMPLE VALUES")
    print("="*70)

    samples = [
        {"name": "Blackgram", "params": (30, 50, 30, 30, 70, 6.8, 80)},
        {"name": "Rice", "params": (90, 42, 43, 20.9, 82, 6.5, 203)},
        {"name": "Wheat", "params": (80, 50, 50, 20, 60, 6.5, 150)},
        {"name": "Maize", "params": (120, 60, 60, 25, 75, 6.5, 200)},
        {"name": "Cotton", "params": (70, 40, 30, 28, 65, 7.0, 90)},
    ]

    for sample in samples:
        n, p, k, t, h, ph, r = sample['params']
        crop, confidence, _, _ = predict_crop(n, p, k, t, h, ph, r)

        print(f"\n{sample['name']}:")
        print(f"  Input: N={n}, P={p}, K={k}, Temp={t}, Humidity={h}, pH={ph}, Rain={r}")
        print(f"  Prediction: {crop.upper()} ({confidence:.2f}%)")

if __name__ == "__main__":
    print("\nChoose mode:")
    print("1. Interactive (enter values manually)")
    print("2. Test with sample values")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        interactive_mode()
    elif choice == '2':
        test_samples()
    else:
        print("Invalid choice!")
