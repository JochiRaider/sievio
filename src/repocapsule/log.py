"""repocapsule.log

Library-friendly logging helpers.

- Installs a `NullHandler` by default to avoid "No handler found" warnings.
- Provides `configure_logging(...)` for apps/CLIs to attach a StreamHandler.
- `get_logger(name=None)` returns the package logger (or any named logger).
- Includes a small `temp_level(...)` context manager.

Usage (library):
    from .log import get_logger
    log = get_logger(__name__)
    log.debug("message")

Usage (CLI/script):
    from .log import configure_logging
    configure_logging(level="INFO")
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional
import logging
import sys

__all__ = [
    "PACKAGE_LOGGER_NAME",
    "get_logger",
    "configure_logging",
    "temp_level",
]

PACKAGE_LOGGER_NAME = "repocapsule"

# Install a NullHandler on the package logger so importing libraries
# don't emit warnings if the application hasn't configured logging.
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger. If `name` is None, return the package logger."""
    return logging.getLogger(name or PACKAGE_LOGGER_NAME)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    stream = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Attach a StreamHandler to the package logger (idempotent-ish).

    - `level` can be an int or a logging level name (e.g., "INFO").
    - `stream` defaults to `sys.stderr`.
    - If a StreamHandler is already present, we keep it and only adjust level.
    - `propagate=False` means the package logger won't bubble up to root.
    """
    logger = get_logger(PACKAGE_LOGGER_NAME)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = bool(propagate)

    if stream is None:
        stream = sys.stderr
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Add a single StreamHandler if none present
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    return logger


@contextmanager
def temp_level(level: int | str, name: Optional[str] = None):
    """Temporarily set a logger's level within a context manager."""
    logger = get_logger(name or PACKAGE_LOGGER_NAME)
    old = logger.level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old)
