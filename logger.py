"""
logger.py
Lightweight logger that writes to stdout AND a timestamped log file simultaneously.
"""

import logging
import sys
from pathlib import Path

# Prevent Python's logging framework from printing "--- Logging error ---"
# tracebacks when a handler's stream is disconnected (common in Colab when
# the frontend briefly loses its ZMQ socket connection).
logging.raiseExceptions = False


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

    # Use the full log path as the logger name so that different runs/files
    # (train.txt vs eval.txt, or different run_name dirs) each get their own
    # logger instance.  A plain name like "evaluate" would be reused across
    # calls and return stale handlers pointing to the wrong file.
    logger = logging.getLogger(str(log_path))
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    # Stop messages bubbling up to the root logger (prevents Colab's duplicate INFO: lines)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — append mode so resumed runs accumulate in one file
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler — wrapped to silently ignore Colab transport disconnects.
    # Must override handleError (not emit) because the base StreamHandler.emit
    # already catches all exceptions internally and routes them to handleError.
    class _SafeStreamHandler(logging.StreamHandler):
        def handleError(self, record):
            import sys
            exc_type = sys.exc_info()[0]
            if exc_type is OSError:
                return          # silently drop transport-disconnect errors
            super().handleError(record)

    sh = _SafeStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"=== Log opened: {log_path} ===")
    return logger
