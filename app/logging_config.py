import logging
import logging.config
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging for the service.

    Parameters
    ----------
    level: str
        Minimum log level for the root logger (e.g. "INFO", "DEBUG").
    """

    log_level = level.upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": log_level,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience wrapper to get a configured logger."""

    return logging.getLogger(name)
