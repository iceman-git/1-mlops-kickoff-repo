import logging
from pathlib import Path
import yaml

def read_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
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