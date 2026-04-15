import os
import joblib
import shap

from preprocess import load_dataset, preprocess_dataset


def explain_model():
    df = load_dataset()
    X, _, _, _ = preprocess_dataset(df)

    model_path = os.path.join("models", "crop_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run src/train.py first."
        )

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)
    print("SHAP explanation generated for the crop irrigation model.")


if __name__ == "__main__":
    explain_model()
