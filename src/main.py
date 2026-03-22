"""
Module: main.py
---------------
Role: Single orchestration entry point for the full ML pipeline.
Responsibility: Glue modules together in order with fail-fast checks.
Pipeline contract: Reads config.yaml, produces clean data, a saved model,
                   and a predictions report as artifacts.

All non-secret runtime settings come from config.yaml.
Secrets (WANDB_API_KEY, WANDB_ENTITY, etc.) come from .env only.
"""

import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from src.clean_data import clean_data
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_data
from src.train import train_model
from src.utils import read_config, save_csv, save_model, setup_logger
from src.validate import validate_dataframe

# ── Load config once at module level ─────────────────────────────────────────
CONFIG = read_config("config.yaml")


def main():
    """
    Orchestrates the full end-to-end pipeline:
      load → clean → validate → split → feature build → train → evaluate
      → save model → batch inference → save predictions
    """
    logger = setup_logger(CONFIG["logging"]["log_file"], CONFIG["logging"]["level"])
    logger.info("[main] Starting end-to-end pipeline.")

    # ── 0) Ensure artifact directories exist ──────────────────────────────────
    logger.info("[main] Ensuring artifact directories exist.")
    for directory in ["data/raw", "data/processed", "models", "reports", "logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # ── 1) Resolve raw data path (scaffolding fallback for tests) ─────────────
    raw_path = Path(CONFIG["data"]["raw_path"])
    example_path = Path("data/raw/example.csv")
    example_mode = False

    if (not raw_path.exists()) and example_path.exists():
        raw_path = example_path
        example_mode = True
        logger.info(
            "[main] NOTE: Configured raw data not found. "
            "Falling back to example.csv scaffold (test/scaffolding mode)."
        )

    # ── 2) Load raw data ──────────────────────────────────────────────────────
    data_config = {
        "data": {"raw_path": str(raw_path)},
        "logging": CONFIG["logging"],
    }
    df_raw = load_data(data_config)

    # ── 3) Resolve active schema and paths ────────────────────────────────────
    # In example/scaffold mode the column names and paths differ from production.
    if example_mode or raw_path.name == "example.csv":
        example_mode = True
        target_column = "target"
        features_cfg = {
            "quantile_bin": ["num_feature"],
            "binary_sum_cols": [],
            "categorical_onehot": ["cat_feature"],
            "numeric_passthrough": [],
            "n_bins": CONFIG["features"]["n_bins"],
        }
        processed_path = Path("data/processed/clean.csv")
        model_path = Path("models/model.joblib")
        predictions_path = Path("reports/predictions.csv")
        problem_type = "classification"
    else:
        target_column = CONFIG["schema"]["target"]
        features_cfg = CONFIG["features"]
        processed_path = Path(CONFIG["data"]["processed_path"])
        model_path = Path(CONFIG["artifacts"]["model_path"])
        predictions_path = Path(CONFIG["inference"]["predictions_output"])
        problem_type = CONFIG["training"]["problem_type"]

    # ── 4) Clean ──────────────────────────────────────────────────────────────
    required_cols = (
        features_cfg["quantile_bin"]
        + features_cfg.get("categorical_onehot", [])
        + features_cfg["numeric_passthrough"]
        + [target_column]
    )
    clean_config = {
        "logging": CONFIG["logging"],
        "schema": {
            "required_columns": required_cols,
            "target": target_column,
        },
    }
    df_clean = clean_data(df_raw, clean_config)

    # ── 5) Save processed clean CSV ───────────────────────────────────────────
    save_csv(df_clean, processed_path)

    # ── 6) Validate ───────────────────────────────────────────────────────────
    validate_dataframe(df_clean, required_columns=required_cols)

    # ── 7) Train-test split (BEFORE feature fitting to prevent leakage) ───────
    logger.info("[main] Creating train-test split (before feature fitting).")
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    stratify = y if problem_type == "classification" else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG["split"]["test_size"],
            random_state=CONFIG["split"]["random_state"],
            stratify=stratify,
        )
    except ValueError as e:
        logger.warning(
            "[main] Stratified split failed: %s. Falling back to non-stratified split.", e
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG["split"]["test_size"],
            random_state=CONFIG["split"]["random_state"],
            stratify=None,
        )

    # ── 8) Fail-fast feature checks ───────────────────────────────────────────
    logger.info("[main] Running fail-fast feature configuration checks.")
    quantile_bin_cols = features_cfg["quantile_bin"]
    categorical_onehot_cols = features_cfg.get("categorical_onehot", [])
    numeric_passthrough_cols = features_cfg["numeric_passthrough"]

    configured_cols = set(quantile_bin_cols + categorical_onehot_cols + numeric_passthrough_cols)
    missing = [c for c in configured_cols if c not in X_train.columns]
    if missing:
        raise ValueError(f"Configured feature columns missing from training data: {missing}")

    non_numeric_bin = [c for c in quantile_bin_cols if not is_numeric_dtype(X_train[c])]
    if non_numeric_bin:
        raise ValueError(
            f"Quantile-binning requires numeric dtypes. "
            f"Non-numeric columns configured for quantile_bin: {non_numeric_bin}"
        )

    # ── 9) Build feature preprocessor ────────────────────────────────────────
    logger.info("[main] Building preprocessing recipe.")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=quantile_bin_cols,
        categorical_onehot_cols=categorical_onehot_cols,
        numeric_passthrough_cols=numeric_passthrough_cols,
        n_bins=int(features_cfg["n_bins"]),
    )

    # ── 10) Train model pipeline ──────────────────────────────────────────────
    train_config = {
        "model_name": CONFIG["training"]["model_name"],
        "random_state": CONFIG["training"]["random_state"],
        "model_output_path": str(model_path),
    }
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        config=train_config,
    )

    # ── 11) Save model artifact ───────────────────────────────────────────────
    save_model(model, model_path)

    # ── 12) Evaluate on held-out test set ─────────────────────────────────────
    eval_config = {
        "problem_type": problem_type,
        "primary_metric": CONFIG["evaluation"]["primary_metric"],
        "save_reports": CONFIG["evaluation"]["save_reports"],
        "report_path": CONFIG["evaluation"]["report_path"],
    }
    metrics = evaluate_model(model=model, X=X_test, y=y_test, config=eval_config)
    metric_value = metrics.get(eval_config["primary_metric"])

    if problem_type == "regression":
        logger.info("[main] Test RMSE: %.4f", metric_value)
    else:
        logger.info("[main] Test weighted F1: %.4f", metric_value)

    # ── 13) Batch inference and save predictions report ───────────────────────
    logger.info("[main] Running inference on example rows and saving report.")
    top_n = CONFIG["inference"].get("top_n", 5)
    X_infer = X_test.head(top_n).copy()
    df_preds = run_inference(model, X_infer=X_infer)
    save_csv(df_preds[["prediction"]], predictions_path)

    logger.info("[main] Pipeline completed successfully.")
    print(">>> TEST MARKER: main() executed")


if __name__ == "__main__":
    main()
