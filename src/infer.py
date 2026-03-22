"""
Module: infer.py
----------------
Role: Generate predictions from a fitted model on new data.
Responsibility: Inference only — no retraining, no evaluation, no model loading.
Pipeline contract:
    Input:  fitted sklearn Pipeline + pd.DataFrame of new feature rows
    Output: pd.DataFrame with prediction, probability, outcome, and confidence columns
"""

import logging

import pandas as pd

logger = logging.getLogger("mlops")


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model:   fitted sklearn Pipeline from train_model() or loaded via utils.load_model()
    - X_infer: pd.DataFrame of new records with the same feature columns used during training.

    Outputs:
    - pd.DataFrame with columns (all sharing the input index):
        "prediction"           int   — predicted class label
        "survival_probability" float — model confidence for class 1 (0.0–1.0)
        "outcome"              str   — human-readable label
        "high_confidence"      bool  — True if probability >= 0.7 or <= 0.3
    """
    logger.info("[infer] Running inference on %d records.", len(X_infer))

    raw_predictions = model.predict(X_infer)

    predictions_df = pd.DataFrame(
        {"prediction": raw_predictions},
        index=X_infer.index,
    )

    logger.info("[infer] Inference complete. Predictions shape: %s", predictions_df.shape)
    logger.info(
        "[infer] Predicted class 1 rate: %.2f%%",
        predictions_df["prediction"].mean() * 100,
    )

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_infer)[:, 1]
        predictions_df["survival_probability"] = proba

        logger.info("[infer] Mean survival probability: %.2f%%", proba.mean() * 100)

        predictions_df["outcome"] = predictions_df["prediction"].map(
            {0: "Did not survive", 1: "Survived"}
        )

        predictions_df["high_confidence"] = (
            (predictions_df["survival_probability"] >= 0.7) |
            (predictions_df["survival_probability"] <= 0.3)
        )

        logger.info(
            "[infer] High-confidence predictions: %.2f%%",
            predictions_df["high_confidence"].mean() * 100,
        )

    return predictions_df