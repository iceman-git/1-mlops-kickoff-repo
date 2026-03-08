
import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model:    fitted sklearn Pipeline from train_model() or loaded via utils.load_model()
    - X_infer:  pd.DataFrame of new passenger records with the same feature columns
                used during training:
                ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]

    Outputs:
    - pd.DataFrame with exactly one column: "prediction"
      - Values are 0 (did not survive) or 1 (survived)
      - Index matches X_infer's index exactly
      - Does NOT include the original feature columns (clean separation)

    Why this contract matters for reliable ML delivery:
    - Returning a single-column DataFrame (not a raw array) makes downstream
      code robust: you can join predictions back to passenger records by index,
      log them, or write them to a database with a consistent schema.
    - Preserving the input index is critical when inference is run on a subset
      of a larger DataFrame — row alignment must be guaranteed.

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

    

    return predictions_df
