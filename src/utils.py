"""
Module: utils.py
----------------
Role: Centralize config reading and I/O plumbing so pipeline steps stay focused.
Responsibility: Read configs, handle CSV/model persistence.
Pipeline contract: Stable file I/O functions used across the pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yaml

from src.logger import setup_logger  # noqa: F401 — re-exported for backward compatibility

logger = logging.getLogger("mlops")


def read_config(path: str = "config.yaml") -> dict:
    """
    Inputs:
    - path: Path to a YAML config file (default: config.yaml).
    Outputs:
    - A dictionary containing configuration settings.
    """
    p = Path(path)

    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    repo_root = Path(__file__).resolve().parents[1]
    p2 = repo_root / path

    if p2.exists():
        with p2.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise FileNotFoundError(f"Missing config file: {p} (also tried {p2})")


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file.
    Outputs:
    - A pandas DataFrame loaded from the CSV.
    """
    logger.info("Loading CSV from: %s", filepath)
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Output CSV path.
    Outputs:
    - None (writes file to disk).
    """
    logger.info("Saving CSV to: %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: A fitted model object (often an sklearn Pipeline).
    - filepath: Output model artifact path.
    Outputs:
    - None (writes file to disk).
    """
    logger.info("Saving model to: %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a saved model artifact.
    Outputs:
    - The loaded model object.
    """
    logger.info("Loading model from: %s", filepath)
    return joblib.load(filepath)