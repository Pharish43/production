import pickle
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model on soil condition classification."""
    model = XGBClassifier(n_estimators=50, random_state=42, verbose=0)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return model

def save_model(model, model_path):
    """Save trained model."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def save_scaler(scaler, scaler_path):
    """Save MinMaxScaler for input preprocessing."""
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

def load_model(model_path):
    """Load trained model."""
    return joblib.load(model_path)

def load_scaler(scaler_path):
    """Load scaler for input preprocessing."""
    return joblib.load(scaler_path)

def predict_soil_condition(model, scaler, le, input_data):
    """Make prediction for new sensor data."""
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = le.inverse_transform(prediction)
    return result[0]
