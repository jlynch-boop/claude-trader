"""
utils.py — Shared utilities: logging setup, date helpers.

Usage:
    from src.utils import get_logger, ts_to_datetime, datetime_to_ts

    logger = get_logger(__name__)
    logger.info("Starting data fetch...")
"""

import logging
import os
from datetime import datetime, timezone


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both the console and logs/trader.log.

    Args:
        name: Logger name (use __name__ in each module).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they already exist (avoid duplicate log lines)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — writes to logs/trader.log
    log_dir = _find_logs_dir()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "trader.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _find_logs_dir() -> str | None:
    """Find the logs/ directory relative to this file's project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    logs_dir = os.path.join(root, "logs")
    return logs_dir


def ts_to_datetime(timestamp_ms: int) -> datetime:
    """
    Convert a millisecond UNIX timestamp (as returned by CCXT) to a UTC datetime.

    Args:
        timestamp_ms: Milliseconds since epoch.

    Returns:
        UTC-aware datetime object.

    Example:
        >>> ts_to_datetime(1704067200000)
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def datetime_to_ts(dt: datetime) -> int:
    """
    Convert a datetime to a millisecond UNIX timestamp.

    Args:
        dt: datetime object (naive datetimes assumed to be UTC).

    Returns:
        Milliseconds since epoch as an integer.

    Example:
        >>> datetime_to_ts(datetime(2024, 1, 1, tzinfo=timezone.utc))
        1704067200000
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def date_str_to_ts(date_str: str) -> int:
    """
    Convert a date string like "2024-01-01" to a millisecond timestamp.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Milliseconds since epoch as an integer.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return datetime_to_ts(dt)
