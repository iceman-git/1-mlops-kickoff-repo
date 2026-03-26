"""
Module: utils.py
----------------
Role: Centralize config reading and I/O plumbing so pipeline steps stay focused.
Responsibility: Read configs, handle CSV/model persistence.
Pipeline contract: Stable file I/O functions used across the pipeline.
"""

# Standard library 
import logging
import os
from pathlib import Path

# Third-party 
import joblib
import pandas as pd
import yaml

# Local 
from src.logger import setup_logger  

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
    logger.info("Loading model from local path: %s", filepath)
    return joblib.load(filepath)


def load_model_for_serving(config: dict):
    """
    Load the model for serving based on the MODEL_SOURCE environment variable.

    - MODEL_SOURCE=wandb  → download the artifact aliased 'prod' from W&B registry
    - MODEL_SOURCE=local  → load from the local path defined in config.yaml

    This function is the single handoff point between training and serving,
    ensuring inference always uses a managed, traceable model artifact.

    Inputs:
    - config: Full config dict loaded from config.yaml.
    Outputs:
    - Loaded model object (sklearn Pipeline).
    """
    import wandb

    model_source = os.environ.get("MODEL_SOURCE", "local")

    if model_source == "wandb":
        entity = os.environ.get("WANDB_ENTITY")
        project = config["wandb"]["project"]
        artifact_name = config["wandb"]["artifact_name"]
        alias = os.environ.get("WANDB_MODEL_ALIAS", config["wandb"]["artifact_alias"])

        logger.info(
            "Loading model from W&B registry: %s/%s/%s:%s",
            entity, project, artifact_name, alias,
        )

        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
        artifact_dir = artifact.download()

        model_filename = Path(config["artifacts"]["model_path"]).name
        model_path = Path(artifact_dir) / model_filename

        return joblib.load(model_path)

    else:
        local_path = Path(config["artifacts"]["model_path"])
        logger.info("Loading model from local path: %s", local_path)
        return joblib.load(local_path)