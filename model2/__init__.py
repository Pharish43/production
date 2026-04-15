from .src.predict import predict_crop


def load_production_model():
    raise RuntimeError(
        "Production model2 artifacts are missing. "
        "Add model2/irrigation_model_best.pkl, model2/irrigation_scaler.pkl, "
        "and model2/irrigation_label_encoders.pkl."
    )


def predict(*args, **kwargs):
    raise RuntimeError(
        "model2.predict is unavailable because production artifacts are missing. "
        "Add the production model2 artifact files before using this function."
    )

__all__ = ["load_production_model", "predict", "predict_crop"]
