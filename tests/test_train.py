import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.train import train_model


def _dummy_preprocessor() -> ColumnTransformer:
    # Simple numeric preprocessor for tests
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), ["x1", "x2"])],
        remainder="drop",
    )


def test_train_model_returns_pipeline_and_can_predict(tmp_path):
    X_train = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [10, 20, 30, 40]})
    y_train = pd.Series([0, 1, 0, 1])

    config = {
        "model_name": "logistic_regression",
        "random_state": 42,
        "model_output_path": str(tmp_path / "model.joblib"),
    }

    model = train_model(X_train, y_train, _dummy_preprocessor(), config)

    assert isinstance(model, Pipeline)
    preds = model.predict(X_train)
    assert len(preds) == len(X_train)


def test_train_model_raises_on_empty_data():
    X_train = pd.DataFrame(columns=["x1", "x2"])
    y_train = pd.Series([], dtype=int)

    config = {"model_name": "logistic_regression", "random_state": 42}

    with pytest.raises(ValueError):
        train_model(X_train, y_train, _dummy_preprocessor(), config)


def test_train_model_raises_on_shape_mismatch():
    X_train = pd.DataFrame({"x1": [1, 2], "x2": [10, 20]})
    y_train = pd.Series([0])  # mismatch

    config = {"model_name": "logistic_regression", "random_state": 42}

    with pytest.raises(ValueError):
        train_model(X_train, y_train, _dummy_preprocessor(), config)