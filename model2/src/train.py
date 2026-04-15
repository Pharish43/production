import json
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from preprocess import load_dataset, preprocess_dataset, save_encoder

MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


def build_model() -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )
    return search


def save_metadata(df, target_values, encoders):
    metadata = {
        "columns": df.drop(columns=["result"]).columns.tolist(),
        "target_values": sorted(int(x) for x in target_values),
        "crop_values": sorted(df["crop ID"].unique().tolist()),
        "soil_values": sorted(df["soil_type"].unique().tolist()),
        "stage_values": sorted(df["Seedling Stage"].unique().tolist()),
        "encoder_features": list(encoders.keys()),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    df = load_dataset()
    X, y, encoders, target_encoder = preprocess_dataset(df)

    print("Training on crops:", sorted(df["crop ID"].unique().tolist()))
    print("Target categories:", sorted(y.unique().tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    model_search = build_model()
    model_search.fit(X_train, y_train)

    best_model = model_search.best_estimator_
    print("Best hyperparameters:", model_search.best_params_)

    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    for feature_name, encoder in encoders.items():
        encoder_path = os.path.join(MODEL_DIR, f"le_{feature_name.replace(' ', '_')}.pkl")
        save_encoder(encoder, encoder_path)

    if target_encoder is not None:
        save_encoder(target_encoder, os.path.join(MODEL_DIR, "le_target.pkl"))

    save_metadata(df, y.unique().tolist(), encoders)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved {len(encoders)} encoders to {MODEL_DIR}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
