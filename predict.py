import pickle
import numpy as np

# Load saved model, scaler, and label encoder
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("=" * 70)
print("SOIL CONDITION PREDICTION - USING BEST TRAINED MODEL")
print("=" * 70)
print("\nFeatures (in order): N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B")

# Example 1: Sample from real data
print("\n--- Example 1 ---")
input_data_1 = [[200, 8, 500, 7.5, 0.6, 0.8, 15, 0.4, 0.7, 1.2, 3.5, 1.8]]
input_scaled_1 = scaler.transform(input_data_1)
pred_1 = model.predict(input_scaled_1)
result_1 = le.inverse_transform(pred_1)[0]
print(f"Input: {input_data_1[0]}")
print(f"Predicted Class: {result_1}")

# Example 2: Different input
print("\n--- Example 2 ---")
input_data_2 = [[150, 7, 400, 7.2, 0.5, 0.7, 12, 0.3, 0.5, 1.0, 3, 1.5]]
input_scaled_2 = scaler.transform(input_data_2)
pred_2 = model.predict(input_scaled_2)
result_2 = le.inverse_transform(pred_2)[0]
print(f"Input: {input_data_2[0]}")
print(f"Predicted Class: {result_2}")

# Example 3: Another input
print("\n--- Example 3 ---")
input_data_3 = [[250, 9, 600, 7.8, 0.75, 0.9, 20, 0.5, 0.9, 1.5, 4, 2.0]]
input_scaled_3 = scaler.transform(input_data_3)
pred_3 = model.predict(input_scaled_3)
result_3 = le.inverse_transform(pred_3)[0]
print(f"Input: {input_data_3[0]}")
print(f"Predicted Class: {result_3}")

print("\n" + "=" * 70)
print("Classes: 0, 1, 2")
print("=" * 70)
