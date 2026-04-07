"""
logger.py
Lightweight logger that writes to stdout AND a timestamped log file simultaneously.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: Path, filename: str) -> logging.Logger:
    """
    Create (or retrieve) a logger that tees output to:
      - stdout (with the same formatting)
      - log_dir / filename

    Appends to the file if it already exists (safe for resumed runs).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — append mode so resumed runs accumulate in one file
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"=== Log opened: {log_path} ===")
    return logger
