"""
Logging helpers for QTen internals.

This module keeps logger creation consistent across the package by installing a
small stream formatter the first time a named logger is requested.
"""

import logging


def get_logger(name: str, show_datetime: bool = False) -> logging.Logger:
    """
    Return a configured logger for this package.

    Creates or reuses a named logger and, on first use for that logger,
    attaches a StreamHandler with a fixed format of:
    `[{name}|{level}] {message}` or, when requested,
    `[{datetime}|{name}|{level}] {message}`.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` from the caller module.
    show_datetime : bool, default=False
        Whether to include `%(asctime)s` in the formatter.

    Returns
    -------
    logging.Logger
        A logger instance configured with the package formatter.

    Examples
    --------
    ```python
    logger = get_logger(__name__, show_datetime=True)
    logger.debug("message")
    ```
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        if show_datetime:
            fmt = "[%(asctime)s|%(name)s|%(levelname)s] %(message)s"
        else:
            fmt = "[%(name)s|%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
