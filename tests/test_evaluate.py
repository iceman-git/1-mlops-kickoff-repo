import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_model


def _dummy_model() -> Pipeline:
    # A very small model that can fit and predict
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])


def test_evaluate_model_returns_metrics_dict():
    X = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])

    model = _dummy_model()
    model.fit(X, y)

    config = {
        "problem_type": "classification",
        "save_reports": False,
    }

    metrics = evaluate_model(model, X, y, config)

    assert isinstance(metrics, dict)
    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_model_raises_if_no_predict():
    class BadModel:
        pass

    X = pd.DataFrame({"x1": [1], "x2": [2]})
    y = pd.Series([0])

    with pytest.raises(TypeError):
        evaluate_model(BadModel(), X, y, {"problem_type": "classification"})


def test_evaluate_model_raises_on_shape_mismatch():
    model = _dummy_model()
    X = pd.DataFrame({"x1": [1, 2], "x2": [10, 20]})
    y = pd.Series([0])  # mismatch
    model.fit(X, pd.Series([0, 1]))

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, {"problem_type": "classification"})