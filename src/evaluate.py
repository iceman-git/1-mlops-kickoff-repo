"""
Module: Evaluation

Role:
Evaluate trained model performance on validation or test data.

Input:
- Trained sklearn Pipeline
- X (features)
- y (target)

Output:
- Dictionary of metrics
"""

from typing import Any, Dict
import os
import json
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
)


def _validate_inputs(model: Any, X: pd.DataFrame, y: pd.Series) -> None:
    """Fail-fast validation checks."""
    if model is None:
        raise ValueError("Model cannot be None.")

    if not hasattr(model, "predict"):
        raise TypeError("Model must implement a .predict() method.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict,
) -> Dict[str, float]:
    """
    Evaluate a trained model and return performance metrics.

    Expected config keys:
    - problem_type: "classification" or "regression"
    - primary_metric: "f1" or "rmse"
    - save_reports: True/False
    - report_path: "reports/metrics.json"
    """

    _validate_inputs(model, X, y)

    problem_type = config.get("problem_type", "classification")
    primary_metric = config.get("primary_metric", "f1")
    save_reports = config.get("save_reports", False)
    report_path = config.get("report_path", "reports/metrics.json")

    y_pred = model.predict(X)

    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"] = accuracy_score(y, y_pred)
        metrics["precision"] = precision_score(y, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y, y_pred, zero_division=0)

    elif problem_type == "regression":
        rmse = mean_squared_error(y, y_pred, squared=False)
        metrics["rmse"] = rmse

    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    # Optionally save metrics
    if save_reports:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics