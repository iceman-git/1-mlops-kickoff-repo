"""
Educational Goal:
- Why this module exists in an MLOps system:
  Inference is a completely separate concern from training and evaluation.
  In production, you will load a saved model and call this module — training
  code is never run again. Keeping inference isolated here makes that handoff clean.
- Responsibility (separation of concerns):
  infer.py answers: "Given a fitted model and new data, what are the predictions?"
  It does NOT retrain, does NOT evaluate against ground truth, and does NOT load
  the model from disk (that happens in main.py via utils.load_model).
- Pipeline contract (inputs and outputs):
  Input:  fitted sklearn Pipeline + pd.DataFrame of new feature rows
  Output: pd.DataFrame with four columns — prediction, survival_probability,
          outcome, high_confidence — all sharing the input index

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model:    fitted sklearn Pipeline from train_model() or loaded via utils.load_model()
    - X_infer:  pd.DataFrame of new passenger records with the same feature columns
                used during training:
                ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]

    Outputs:
    - pd.DataFrame with four columns, all sharing the input index:
        "prediction"           int   — 0 (did not survive) or 1 (survived)
        "survival_probability" float — model confidence the passenger survived (0.0–1.0)
        "outcome"              str   — human-readable label ("Survived" / "Did not survive")
        "high_confidence"      bool  — True if probability >= 0.7 or <= 0.3

    Why this contract matters for reliable ML delivery:
    - Returning a structured DataFrame makes downstream code robust: you can join
      predictions back to passenger records by index, log them, or write them to
      a database with a consistent schema.
    - Preserving the input index is critical when inference runs on a subset of a
      larger DataFrame — row alignment must be guaranteed.
    - Probability and confidence columns let consumers (dashboards, human reviewers,
      business rules) act on model uncertainty, not just the binary outcome.

    Titanic baseline:
    - Returns binary survival predictions (0 or 1) for each input passenger row
    - Mirrors the new_passengers example from your notebook's inference section
    """
    print(f"[infer] Running inference on {len(X_infer)} records...")  # TODO: replace with logging later

    raw_predictions = model.predict(X_infer)

    predictions_df = pd.DataFrame(
        {"prediction": raw_predictions},
        index=X_infer.index,
    )

    print(f"[infer] Inference complete. Predictions shape: {predictions_df.shape}")  # TODO: replace with logging later
    print(f"[infer] Survival rate in predictions: "
          f"{predictions_df['prediction'].mean():.2%}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):
        # 1. Survival probability — how confident is the model that the passenger survived?
        #    predict_proba returns [[prob_class_0, prob_class_1], ...]; we take column 1.
        proba = model.predict_proba(X_infer)[:, 1]
        predictions_df["survival_probability"] = proba
        print(f"[infer] Mean survival probability: {proba.mean():.2%}")  # TODO: replace with logging later

        # 2. Human-readable outcome label — makes the predictions CSV self-explanatory
        #    to non-technical stakeholders without needing a data dictionary.
        predictions_df["outcome"] = predictions_df["prediction"].map(
            {0: "Did not survive", 1: "Survived"}
        )

        # 3. High-confidence flag — True when the model is sure either way (>= 0.7 or <= 0.3).
        #    Low-confidence predictions (0.3–0.7) could be flagged for human review
        #    in a production system.
        predictions_df["high_confidence"] = (
            (predictions_df["survival_probability"] >= 0.7) |
            (predictions_df["survival_probability"] <= 0.3)
        )
        high_conf_pct = predictions_df["high_confidence"].mean()
        print(f"[infer] High-confidence predictions: {high_conf_pct:.2%}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return predictions_df
