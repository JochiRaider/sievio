# log.py
# SPDX-License-Identifier: MIT
"""Utilities for package-wide logging configuration.

Installs a NullHandler on the package logger to avoid noisy warnings from
importing clients and exposes helpers for runtime configuration and temporary
level overrides.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

__all__ = [
    "PACKAGE_LOGGER_NAME",
    "get_logger",
    "configure_logging",
    "temp_level",
]

PACKAGE_LOGGER_NAME = "sievio"

# Install a NullHandler on the package logger so importing libraries
# don't emit warnings if the application hasn't configured logging.
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger scoped to sievio.

    Args:
        name (str | None): Fully qualified logger name. Defaults to the
            package logger when omitted.

    Returns:
        logging.Logger: Logger instance for the requested name.
    """
    return logging.getLogger(name or PACKAGE_LOGGER_NAME)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    stream = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    propagate: bool | None = None,
    logger_name: str = PACKAGE_LOGGER_NAME,
) -> logging.Logger:
    """Configure a stream handler for a sievio logger.

    Args:
        level (int | str): Logging level or level name. Defaults to
            logging.INFO.
        stream (IO[str] | None): Target stream; defaults to sys.stderr.
        fmt (str | None): Log format string. Defaults to a basic format when
            omitted.
        datefmt (str | None): Date format string for the handler.
        propagate (bool | None): Whether log records bubble up to ancestor loggers.
            When None, defaults to True to allow root handlers (e.g., pytest caplog).
        logger_name (str): Logger name to configure. Defaults to the package
            logger.

    Returns:
        logging.Logger: Logger configured with a single StreamHandler.
    """
    logger = get_logger(logger_name or PACKAGE_LOGGER_NAME)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    if propagate is None:
        logger.propagate = True
    else:
        logger.propagate = bool(propagate)

    if stream is None:
        stream = sys.stderr
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Add a single StreamHandler if none present; refresh closed streams.
    has_stream = False
    for handler in logger.handlers:
        if not isinstance(handler, logging.StreamHandler):
            continue
        has_stream = True
        if getattr(getattr(handler, "stream", None), "closed", False):
            handler.stream = stream
    if not has_stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    return logger


@contextmanager
def temp_level(level: int | str, name: str | None = None):
    """Temporarily set a logger level inside a context manager.

    Args:
        level (int | str): Logging level or level name to apply.
        name (str | None): Logger name. Defaults to the package logger.

    Yields:
        logging.Logger: Logger with the temporary level applied.
    """
    logger = get_logger(name or PACKAGE_LOGGER_NAME)
    old = logger.level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old)
