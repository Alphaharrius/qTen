import logging


def get_logger(name: str, show_datetime: bool = False) -> logging.Logger:
    """Return a configured logger for this package.

    Creates or reuses a named logger and, on first use for that logger,
    attaches a StreamHandler with a fixed format of:
    `[{name}|{level}] {message}` or, when requested,
    `[{datetime}|{name}|{level}] {message}`.

    Parameters
    ----------
    `name`
        Logger name, typically `__name__` from the caller module.
    `show_datetime`
        When True, include `%(asctime)s` in the format.

    Returns
    -------
    `logging.Logger`
        A logger instance configured with the package formatter.
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
