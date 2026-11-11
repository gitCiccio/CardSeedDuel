"""Central logging configuration for CardSuitDuel.

Configure logging once in an entry point, then obtain named loggers in modules::

    from card_suit_duel.logging_config import LoggingManager

    LoggingManager.configure(level="INFO", log_file="results/logs/inspection.log")
    logger = LoggingManager.get_logger(__name__)
    logger.info("Dataset inspection started")

Library modules should only call ``get_logger(__name__)``. They must not configure
logging during import. No handlers are installed automatically by this module.
"""

import logging
from pathlib import Path
from threading import RLock
from typing import Optional, Union


class LoggingManager:
    """Configure and access the application's standard Python loggers.

    This class is a configuration facade, not a replacement for ``logging.Logger``.
    Use the returned logger's ``debug``, ``info``, ``warning``, ``error`` and
    ``exception`` methods. All application loggers share the ``card_suit_duel``
    namespace, format and destinations while retaining their module names.

    Call ``configure`` at script startup before starting worker threads. Repeated
    calls replace only handlers owned by this manager. The Python root logger and
    third-party logging configuration are left untouched. Application records do
    not propagate to the root logger, preventing duplicate root output.
    """

    LOGGER_NAME = "card_suit_duel"
    FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _HANDLER_NAMES = ("card_suit_duel.console", "card_suit_duel.file")
    _lock = RLock()

    @classmethod
    def configure(
        cls,
        level: Union[int, str] = "INFO",
        log_file: Optional[Union[str, Path]] = None,
    ) -> logging.Logger:
        """Enable console logging and, optionally, append to a UTF-8 log file.

        Args:
            level: A standard level name (case-insensitive), or a nonnegative
                integer logging level. The threshold applies to both outputs.
            log_file: Optional output path. Missing parent directories are
                created. Relative paths are resolved against the working directory.

        Returns:
            The application namespace logger. Console output goes to stderr;
            file output is appended, preserving earlier runs.

        Raises:
            ValueError: If the logging level is invalid.
            OSError: If the log directory or file cannot be opened. Existing
                managed handlers remain in place if file setup fails.

        Calling this method again with ``log_file=None`` disables previously
        configured file output. Timestamps use the local system time.
        """
        resolved_level = level
        if isinstance(level, str):
            resolved_level = logging.getLevelName(level.upper())
        if (
            isinstance(resolved_level, bool)
            or not isinstance(resolved_level, int)
            or resolved_level < 0
        ):
            raise ValueError("level must be a standard logging level name or a nonnegative integer")

        with cls._lock:
            formatter = logging.Formatter(cls.FORMAT, datefmt=cls.DATE_FORMAT)
            handlers = []
            try:
                console = logging.StreamHandler()
                console.set_name(cls._HANDLER_NAMES[0])
                handlers.append(console)
                if log_file is not None:
                    path = Path(log_file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
                    file_handler.set_name(cls._HANDLER_NAMES[1])
                    handlers.append(file_handler)
                for handler in handlers:
                    handler.setLevel(resolved_level)
                    handler.setFormatter(formatter)
            except Exception:
                for handler in handlers:
                    handler.close()
                raise

            logger = logging.getLogger(cls.LOGGER_NAME)
            for handler in list(logger.handlers):
                if handler.get_name() in cls._HANDLER_NAMES:
                    logger.removeHandler(handler)
                    handler.close()
            logger.setLevel(resolved_level)
            logger.propagate = False
            for handler in handlers:
                logger.addHandler(handler)
            return logger

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Return a standard logger inside the application namespace.

        Pass ``__name__`` in each module. Existing application-qualified names
        are preserved; other names (including script ``__main__``) are prefixed
        with ``card_suit_duel.``. With no name, return the namespace logger.

        This method does not configure handlers or create files. Call
        ``configure`` in the entry point before emitting application messages.
        """
        if not name:
            return logging.getLogger(cls.LOGGER_NAME)
        if name == cls.LOGGER_NAME or name.startswith(cls.LOGGER_NAME + "."):
            return logging.getLogger(name)
        return logging.getLogger(cls.LOGGER_NAME + "." + name)
