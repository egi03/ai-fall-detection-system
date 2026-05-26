"""
Logging infrastructure for the fall detection system.

Provides centralized, configurable logging using Python's logging module
with JSON formatting for structured telemetry output.

Reference: research/8.1 - Python logging module with python-json-logger
for SIEM-ready, thread-safe, structured JSON logs.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pythonjsonlogger.json import JsonFormatter as _JsonFormatter


def setup_logger(
    name: str = "fall_detection",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure and return a logger instance with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name, used to identify the source module.
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        Path to the log file. If None, logs only to console.
    log_format : str
        Output format: 'json' for structured JSON, 'text' for human-readable.
    max_bytes : int
        Maximum size of a single log file before rotation.
    backup_count : int
        Number of rotated log files to retain.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if log_format == "json":
        formatter = _JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve an existing logger by name.

    Parameters
    ----------
    name : str
        Logger name, typically the module's __name__.

    Returns
    -------
    logging.Logger
        The logger instance.
    """
    return logging.getLogger(name)
