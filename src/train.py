"""
Module: Train

Role:
Train machine learning model using training split only.

Input:
- X_train (pd.DataFrame)
- y_train (pd.Series)
- preprocessor (sklearn transformer)
- config dictionary

Output:
- Fitted sklearn Pipeline
- Saved model artifact (.joblib)
"""

from typing import Any, Dict
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier


def _validate_training_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Fail-fast validation before training."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X_train and y_train must have same number of rows.")


def _select_model(config: Dict) -> Any:
    """Select model based on configuration."""
    model_name = config.get("model_name", "logistic_regression")
    random_state = config.get("random_state", 42)

    if model_name == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        )
        
    if model_name == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    raise ValueError(f"Unsupported model_name: {model_name}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    config: Dict,
) -> Pipeline:
    """
    Train model using sklearn Pipeline.
    """

    _validate_training_data(X_train, y_train)

    if preprocessor is None:
        raise ValueError("Preprocessor cannot be None.")

    model = _select_model(config)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    # Fit ONLY on training data
    pipeline.fit(X_train, y_train)

    # Save model artifact
    model_path = config.get("model_output_path", "models/model.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline