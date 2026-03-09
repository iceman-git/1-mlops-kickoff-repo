"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize config, logging, and I/O plumbing so pipeline steps stay focused.
- Responsibility (separation of concerns): Read configs, configure logging, and handle CSV/model persistence.
- Pipeline contract (inputs and outputs): Stable file I/O functions used across the pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# YAML is used by your existing tests; keep it.
import yaml


def read_config(path: str = "config.yaml") -> dict:
    """
    Inputs:
    - path: Path to a YAML config file (default: config.yaml).
    Outputs:
    - A dictionary containing configuration settings.
    """
    print(f"[utils.read_config] Reading config from: {path}")

    p = Path(path)

    # 1) Try relative to current working directory (normal CLI usage)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # 2) Fallback: try relative to repo root (works when pytest runs in tmp dirs)
    repo_root = Path(__file__).resolve().parents[1]  # src/utils.py → repo root
    p2 = repo_root / path

    if p2.exists():
        with p2.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise FileNotFoundError(f"Missing config file: {p} (also tried {p2})")


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    """
    Inputs:
    - log_file: Path to the log file to write.
    - level: Logging level string (default: INFO).
    Outputs:
    - A configured logging.Logger instance.
    Why this contract matters for reliable ML delivery:
    - Consistent logs are essential for debugging runs in CI/production without relying on prints.
    """
    print(f"[utils.setup_logger] Setting up logger: file={log_file} level={level}")  # TODO: replace with logging later

    # TODO_STUDENT:
    # - Expand handlers/formatters later if needed; keep minimal now.
    logger = logging.getLogger("mlops")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file.
    Outputs:
    - A pandas DataFrame loaded from the CSV.
    Why this contract matters for reliable ML delivery:
    - Standardized data loading reduces “works on my notebook” differences across teammates.
    """
    print(f"[utils.load_csv] Loading CSV from: {filepath}")  # TODO: replace with logging later

    # TODO_STUDENT:
    # - Add pd.read_csv kwargs here (parse_dates, dtype, na_values) once you know your schema.
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Output CSV path.
    Outputs:
    - None (writes file to disk).
    Why this contract matters for reliable ML delivery:
    - Deterministic outputs make downstream steps predictable and easier to test.
    """
    print(f"[utils.save_csv] Saving CSV to: {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # TODO_STUDENT:
    # - Adjust to_csv parameters if needed (sep, encoding). Keep index=False by default.
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: A fitted model object (often an sklearn Pipeline).
    - filepath: Output model artifact path.
    Outputs:
    - None (writes file to disk).
    Why this contract matters for reliable ML delivery:
    - Persisting the full pipeline reduces training/serving skew and enables reproducible inference.
    """
    print(f"[utils.save_model] Saving model to: {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # TODO_STUDENT:
    # - If your org later uses a registry, this function becomes the bridge to that system.
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a saved model artifact.
    Outputs:
    - The loaded model object.
    Why this contract matters for reliable ML delivery:
    - Enables reusing the exact same trained pipeline for evaluation and inference.
    """
    print(f"[utils.load_model] Loading model from: {filepath}")  # TODO: replace with logging later

    # TODO_STUDENT:
    # - Keep model loading simple; versioning can be layered later.
    return joblib.load(filepath)