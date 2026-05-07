"""
Centralised logger — all pipeline modules import from here.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d}.log"

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with console + rotating file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (INFO+) ────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # ── File handler (DEBUG+) ──────────────────────────────────────────────────
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
