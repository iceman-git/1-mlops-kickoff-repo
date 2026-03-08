"""
Module: Main Pipeline

Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Split -> Features -> Train -> Evaluate -> Infer)
Usage: python -m src.main --config config.yaml

Design goals (per assignment guidance):
- Centralize orchestration and delegate execution to single-purpose modules.
- Drive execution via config.
- Enforce Train/Validation/Test split early (prevent leakage).
- Pass data explicitly between steps (prevent hidden state).
- Fail fast on config/data issues; log every step; standardize artifacts.

Refs: group guidelines + module summary. (See docs in repo.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# -----------------------------
# Errors
# -----------------------------
class ConfigError(ValueError):
    """Raised when the configuration is missing required keys or has invalid values."""


class PipelineStepError(RuntimeError):
    """Raised when a pipeline step fails with a contextualized error message."""


# -----------------------------
# Config + Logging
# -----------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config into a dict."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ConfigError(f"Config must be a YAML file (.yml/.yaml). Got: {path.name}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a dictionary (YAML mapping).")
    return cfg


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Dict[str, Any], run_id: str) -> logging.Logger:
    """
    Setup logging with both console + optional file output.
    Config keys (optional):
      logging:
        level: INFO
        dir: logs
        file: main.log
    """
    log_cfg = cfg.get("logging", {}) if isinstance(cfg.get("logging", {}), dict) else {}

    level_str = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    log_dir = Path(log_cfg.get("dir", "logs"))
    _ensure_dir(log_dir)

    log_file = log_cfg.get("file", "main.log")
    log_path = log_dir / f"{run_id}__{log_file}"

    logger = logging.getLogger("mlops-pipeline")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging initialized. level=%s file=%s", level_str, log_path.as_posix())
    return logger


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Fail-fast configuration validation.
    Required keys (minimal contract):
      data:
        target: <str>
      split:
        test_size: <float 0-1>
        val_size: <float 0-1>  (fraction of remaining after test split)
        random_state: <int>
        stratify: <bool> (optional)
      artifacts:
        dir: <str> (where metrics/preds/etc go)
    """
    def req(path: str) -> Any:
        cur: Any = cfg
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                raise ConfigError(f"Missing required config key: '{path}'")
            cur = cur[key]
        return cur

    target = req("data.target")
    if not isinstance(target, str) or not target.strip():
        raise ConfigError("config.data.target must be a non-empty string.")

    split = cfg.get("split", {})
    if not isinstance(split, dict):
        raise ConfigError("config.split must be a dictionary.")

    test_size = split.get("test_size", None)
    val_size = split.get("val_size", None)
    rs = split.get("random_state", 42)

    for k, v in [("test_size", test_size), ("val_size", val_size)]:
        if not isinstance(v, (int, float)) or not (0 < float(v) < 1):
            raise ConfigError(f"config.split.{k} must be a float in (0, 1). Got: {v}")

    if not isinstance(rs, int):
        raise ConfigError("config.split.random_state must be an int.")

    art = cfg.get("artifacts", {})
    if not isinstance(art, dict):
        raise ConfigError("config.artifacts must be a dictionary.")
    art_dir = art.get("dir", "artifacts")
    if not isinstance(art_dir, str) or not art_dir.strip():
        raise ConfigError("config.artifacts.dir must be a non-empty string.")


# -----------------------------
# Utility: call teammates' modules safely
# -----------------------------
def resolve_callable(module: Any, candidates: Tuple[str, ...]) -> Callable[..., Any]:
    """
    Find the first callable attribute in `module` from candidates.
    Raise a clear error if none exist.
    """
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"Expected one of {candidates} in module '{module.__name__}', but none were found."
    )


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_three_way(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
    logger: logging.Logger,
) -> SplitData:
    """
    Enforce Train/Validation/Test split early.
    - test_size: fraction of full dataset allocated to test
    - val_size: fraction of the *remaining* allocated to validation
    """
    if target not in df.columns:
        raise PipelineStepError(f"Target column '{target}' not found in dataframe columns.")

    if df.empty:
        raise PipelineStepError("Input dataframe is empty; cannot split.")

    y = df[target]
    X = df.drop(columns=[target])

    strat = y if stratify else None

    logger.info("Splitting data: test_size=%.3f val_size=%.3f stratify=%s",
                float(test_size), float(val_size), stratify)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=random_state,
        stratify=strat,
    )

    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=float(val_size),
        random_state=random_state,
        stratify=strat_tmp,
    )

    logger.info(
        "Split shapes: X_train=%s X_val=%s X_test=%s",
        tuple(X_train.shape),
        tuple(X_val.shape),
        tuple(X_test.shape),
    )
    return SplitData(X_train=X_train, X_val=X_val, X_test=X_test,
                     y_train=y_train, y_val=y_val, y_test=y_test)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# -----------------------------
# Pipeline Orchestration
# -----------------------------
def run_pipeline(cfg: Dict[str, Any], logger: logging.Logger, run_id: str) -> Dict[str, Any]:
    """
    Orchestrate end-to-end pipeline by delegating to the specialized modules.
    Returns a summary dict (useful for tests or CI).
    """
    # Import modules (local package imports)
    try:
        from src import load_data, clean_data, validate, features, train, evaluate, infer
    except Exception as e:
        raise PipelineStepError(
            "Failed to import pipeline modules from src/. "
            "Expected: load_data.py, clean_data.py, validate.py, features.py, train.py, evaluate.py, infer.py"
        ) from e

    # Resolve functions with flexible naming
    load_fn = resolve_callable(load_data, ("load_data", "load", "run", "main"))
    clean_fn = resolve_callable(clean_data, ("clean_data", "clean", "run", "main"))
    validate_fn = resolve_callable(validate, ("validate_dataframe", "validate", "run", "main"))
    feat_fn = resolve_callable(features, ("build_preprocessor", "build_features", "make_preprocessor", "run", "main"))
    train_fn = resolve_callable(train, ("train_model", "train", "run", "main"))
    eval_fn = resolve_callable(evaluate, ("evaluate_model", "evaluate", "run", "main"))
    infer_fn = resolve_callable(infer, ("predict", "infer", "run_inference", "run", "main"))

    # Artifact paths
    art_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts"))
    _ensure_dir(art_dir)

    # 1) Load
    logger.info("STEP 1/7: load_data")
    try:
        df_raw = load_fn(cfg, logger=logger)
    except TypeError:
        # fallback if teammate didn't include logger kwarg
        df_raw = load_fn(cfg)
    except Exception as e:
        raise PipelineStepError("load_data step failed.") from e

    if not isinstance(df_raw, pd.DataFrame):
        raise PipelineStepError("load_data must return a pandas.DataFrame.")
    logger.info("Loaded data shape=%s", tuple(df_raw.shape))

    # 2) Clean
    logger.info("STEP 2/7: clean_data")
    try:
        df_clean = clean_fn(df_raw, cfg, logger=logger)
    except TypeError:
        df_clean = clean_fn(df_raw, cfg)
    except Exception as e:
        raise PipelineStepError("clean_data step failed.") from e

    if not isinstance(df_clean, pd.DataFrame):
        raise PipelineStepError("clean_data must return a pandas.DataFrame.")
    if df_clean.empty:
        raise PipelineStepError("clean_data produced an empty dataframe. Stop early.")
    logger.info("Cleaned data shape=%s", tuple(df_clean.shape))

    # 3) Validate (security gate)
    logger.info("STEP 3/7: validate")
    try:
        validation_report = validate_fn(df_clean, cfg, logger=logger)
    except TypeError:
        validation_report = validate_fn(df_clean, cfg)
    except Exception as e:
        raise PipelineStepError("validate step failed.") from e

    # Save validation report if dict-like
    if isinstance(validation_report, dict):
        write_json(art_dir / f"{run_id}__validation_report.json", validation_report)
        logger.info("Wrote validation report JSON.")

    # 4) Split early (prevents leakage downstream) :contentReference[oaicite:2]{index=2}
    logger.info("STEP 4/7: split (train/val/test)")
    split_cfg = cfg.get("split", {})
    target = cfg["data"]["target"]
    split_data = split_three_way(
        df=df_clean,
        target=target,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        random_state=int(split_cfg.get("random_state", 42)),
        stratify=bool(split_cfg.get("stratify", False)),
        logger=logger,
    )

    # 5) Build feature recipe (unfitted) on X_train only, to avoid leakage by design :contentReference[oaicite:3]{index=3}
    logger.info("STEP 5/7: features (build preprocessing recipe)")
    try:
        preprocessor = feat_fn(split_data.X_train, cfg, logger=logger)
    except TypeError:
        preprocessor = feat_fn(split_data.X_train, cfg)
    except Exception as e:
        raise PipelineStepError("features step failed.") from e

    # 6) Train (fit only on train split; bundle recipe + model) :contentReference[oaicite:4]{index=4}
    logger.info("STEP 6/7: train")
    try:
        train_out = train_fn(
            split_data.X_train,
            split_data.y_train,
            preprocessor,
            cfg,
            logger=logger,
        )
    except TypeError:
        train_out = train_fn(split_data.X_train, split_data.y_train, preprocessor, cfg)
    except Exception as e:
        raise PipelineStepError("train step failed.") from e

    # We accept either:
    # - a model object
    # - or a dict with {"model": ..., "model_path": ...}
    model = train_out.get("model") if isinstance(train_out, dict) else train_out
    model_path = train_out.get("model_path") if isinstance(train_out, dict) else None

    if not hasattr(model, "predict"):
        raise PipelineStepError("Trained model artifact must implement .predict().")

    # 7) Evaluate on validation (or test) strictly on untouched split :contentReference[oaicite:5]{index=5}
    logger.info("STEP 7/7: evaluate + infer")
    try:
        metrics = eval_fn(model, split_data.X_val, split_data.y_val, cfg, logger=logger)
    except TypeError:
        metrics = eval_fn(model, split_data.X_val, split_data.y_val, cfg)
    except Exception as e:
        raise PipelineStepError("evaluate step failed.") from e

    if isinstance(metrics, dict):
        write_json(art_dir / f"{run_id}__metrics.json", metrics)
        logger.info("Wrote metrics JSON.")
    else:
        metrics = {"metric": metrics}
        write_json(art_dir / f"{run_id}__metrics.json", metrics)
        logger.info("Wrote metrics JSON (scalar wrapped).")

    # Inference (commonly on test split)
    try:
        preds = infer_fn(model, split_data.X_test, cfg, logger=logger)
    except TypeError:
        preds = infer_fn(model, split_data.X_test, cfg)
    except Exception as e:
        raise PipelineStepError("infer step failed.") from e

    # Standardize prediction artifact
    pred_path = art_dir / f"{run_id}__predictions.csv"
    if isinstance(preds, pd.DataFrame):
        preds.to_csv(pred_path, index=True)
    else:
        # if teammate returns array-like, wrap into DataFrame
        pd.DataFrame({"prediction": preds}, index=split_data.X_test.index).to_csv(pred_path, index=True)
    logger.info("Wrote predictions: %s", pred_path.as_posix())

    summary = {
        "run_id": run_id,
        "data_shape_raw": tuple(df_raw.shape),
        "data_shape_clean": tuple(df_clean.shape),
        "split_shapes": {
            "train": tuple(split_data.X_train.shape),
            "val": tuple(split_data.X_val.shape),
            "test": tuple(split_data.X_test.shape),
        },
        "model_path": model_path,
        "artifacts": {
            "validation_report": (art_dir / f"{run_id}__validation_report.json").as_posix(),
            "metrics": (art_dir / f"{run_id}__metrics.json").as_posix(),
            "predictions": pred_path.as_posix(),
        },
        "metrics": metrics,
    }
    return summary


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the end-to-end MLOps pipeline.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_config(args.config)
    validate_config(cfg)

    logger = setup_logging(cfg, run_id)
    logger.info("Starting pipeline run_id=%s config=%s", run_id, Path(args.config).resolve().as_posix())

    try:
        summary = run_pipeline(cfg, logger, run_id)
    except Exception as e:
        logger.exception("PIPELINE FAILED: %s", str(e))
        raise

    # Print a compact end-of-run summary for terminal/CI readability
    logger.info("PIPELINE SUCCESS run_id=%s metrics=%s", run_id, summary.get("metrics"))


if __name__ == "__main__":
    main()