import pickle
import numpy as np

# Load saved model, scaler, and label encoder
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Class labels mapping
class_labels = {
    0: "Good",
    1: "Bad",
    2: "Medium"
}

print("=" * 70)
print("SOIL CONDITION PREDICTION - INTERACTIVE MODE")
print("=" * 70)
print("\nEnter soil nutrient values one by one")
print("\nFeatures required (in order):")
print("  1. N (Nitrogen): 0-300")
print("  2. P (Phosphorous): 0-50")
print("  3. K (Potassium): 0-800")
print("  4. pH: 3-10")
print("  5. EC (Electrical Conductivity): 0-2")
print("  6. OC (Organic Carbon): 0-2")
print("  7. S (Sulphur): 0-50")
print("  8. Zn (Zinc): 0-2")
print("  9. Fe (Iron): 0-2")
print("  10. Cu (Copper): 0-2")
print("  11. Mn (Manganese): 0-10")
print("  12. B (Boron): 0-3")
print("\n" + "=" * 70)

while True:
    try:
        print("\n[Enter values or type 'quit' to exit]")
        input_values = []

        features = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

        for i, feature in enumerate(features, 1):
            while True:
                try:
                    value = input(f"  {i}. {feature}: ").strip()

                    if value.lower() == 'quit':
                        print("\nExiting... Goodbye!")
                        exit()

                    value = float(value)
                    input_values.append(value)
                    break
                except ValueError:
                    print(f"  Invalid input! Please enter a number for {feature}")

        # Create input array and predict
        input_data = [input_values]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prediction_value = prediction[0]

        # Get the label
        result_label = class_labels.get(prediction_value, f"Unknown ({prediction_value})")

        print("\n" + "-" * 70)
        print("PREDICTION RESULT:")
        print("-" * 70)
        print(f"Input values: {input_values}")
        print(f"Predicted Soil Class: {result_label}")
        print("-" * 70)

    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again!")

