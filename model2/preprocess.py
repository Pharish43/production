import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(csv_path: str = "data/cropdata_updated.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        alternate_path = os.path.join(os.path.dirname(csv_path), os.pardir, "cropdata_updated.csv")
        alternate_path = os.path.normpath(alternate_path)
        if os.path.exists(alternate_path):
            csv_path = alternate_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. Please place cropdata_updated.csv in the data/ folder or the project root."
            )

    df = pd.read_csv(csv_path)
    return df


def build_label_encoders(df: pd.DataFrame, categorical_columns: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}

    for column in categorical_columns:
        if column not in df.columns:
            continue

        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder

    return df, encoders


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder], LabelEncoder | None]:
    if "result" not in df.columns:
        raise ValueError("The dataset must include a 'result' target column.")

    categorical_columns = [col for col in ["crop ID", "soil_type", "Seedling Stage"] if col in df.columns]
    df, encoders = build_label_encoders(df, categorical_columns)

    X = df.drop(columns=["result"])
    y = df["result"]

    target_encoder: LabelEncoder | None = None
    if y.dtype == object or y.dtype.name == "category":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    return X, y, encoders, target_encoder


def save_encoder(encoder: LabelEncoder, path: str) -> None:
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
