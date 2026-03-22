"""
Module: logger.py
-----------------
Role: Centralize logging configuration for the entire pipeline.
Responsibility: One place to configure handlers, formatters, and log levels.
All other modules obtain the logger via logging.getLogger("mlops").
"""

import logging
from pathlib import Path


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    """
    Configure and return the shared pipeline logger.
    Writes to both the console and a local log file simultaneously.

    Args:
        log_file: Path to the log file.
        level:    Logging level string (INFO, DEBUG, WARNING, etc.)

    Returns:
        Configured logging.Logger instance named "mlops".
    """
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