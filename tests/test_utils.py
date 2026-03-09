import pytest
from pathlib import Path
import logging

from src.utils import read_config, setup_logger


def test_read_config_returns_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
data:
  raw_path: data/raw/test.csv
logging:
  log_file: test.log
  level: INFO
""")

    config = read_config(str(config_file))

    assert isinstance(config, dict)
    assert "data" in config
    assert "logging" in config


def test_setup_logger_creates_logger(tmp_path):
    log_file = tmp_path / "test.log"

    logger = setup_logger(str(log_file), "INFO")

    assert isinstance(logger, logging.Logger)

    logger.info("test message")

    assert log_file.exists()