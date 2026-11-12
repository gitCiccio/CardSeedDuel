"""Central logging configuration for CardSuitDuel.

Configure logging once in an entry point, then obtain named loggers in modules::

    from card_suit_duel.logging_config import LoggingManager

    LoggingManager.configure(level="INFO", log_file="results/logs/inspection.log")
    logger = LoggingManager.get_logger(__name__)
    logger.info("Dataset inspection started")
    logger.success("Inspection completed")

To select exact severities instead of a minimum threshold::

    LoggingManager.configure(
        levels=["DEBUG", "INFO", "SUCCESS", "ERROR", "CRITICAL"],
        log_file="results/logs/inspection.log",
    )

Library modules should only call ``get_logger(__name__)``. They must not configure
logging during import. No handlers are installed automatically by this module.
"""

import logging
import sys
from pathlib import Path
from threading import RLock
from typing import Iterable, Optional, Union


SUCCESS = 25


class ProjectLogger(logging.LoggerAdapter):
    """Standard logger adapter adding ``success(message, *args)`` at level 25.

    Use info/error/debug as usual, and success for completed operations. The
    adapter avoids replacing Python's global logger class.
    """

    def success(self, message, *args, **kwargs) -> None:
        """Log a successfully completed operation between INFO and WARNING."""
        self.log(SUCCESS, message, *args, **kwargs)


def _resolve_level(level: Union[int, str]) -> int:
    """Validate a standard/custom integer level or a recognized level name."""
    resolved = SUCCESS if isinstance(level, str) and level.upper() == "SUCCESS" else level
    if isinstance(resolved, str):
        resolved = logging.getLevelName(resolved.upper())
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
        raise ValueError("Invalid logging level: {!r}".format(level))
    return resolved


class _LevelFilter(logging.Filter):
    """Accept only explicitly selected severity levels."""

    def __init__(self, levels: Iterable[int]):
        super().__init__()
        self.levels = frozenset(levels)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.levels


class _ColorFormatter(logging.Formatter):
    """Color console records without changing the underlying shared LogRecord."""

    COLORS = {logging.DEBUG: "\033[33m", logging.INFO: "\033[34m",
              SUCCESS: "\033[32m", logging.WARNING: "\033[33m",
              logging.ERROR: "\033[31m", logging.CRITICAL: "\033[31m"}

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return color + text + "\033[0m" if color else text


class LoggingManager:
    """Configure and access the application's standard Python loggers.

    This class is a configuration facade, not a replacement for ``logging.Logger``.
    Use the returned adapter's ``debug``, ``info``, ``success``, ``warning``,
    ``error`` and ``exception`` methods. All application loggers share the ``card_suit_duel``
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
        *,
        levels: Optional[Iterable[Union[int, str]]] = None,
        colors: bool = True,
    ) -> logging.Logger:
        """Enable console logging and, optionally, append to a UTF-8 log file.

        Args:
            level: Minimum severity for both outputs when levels is None.
                DEBUG includes INFO, SUCCESS, WARNING, ERROR and CRITICAL.
            log_file: Optional output path. Missing parent directories are
                created. Relative paths are resolved against the working directory.
            levels: Optional exact selection, e.g. ["DEBUG", "INFO", "ERROR"].
                Overrides the minimum threshold; other severities are excluded.
                Add "SUCCESS", "WARNING" or "CRITICAL" explicitly when wanted.
                An empty selection is rejected to avoid accidental silence.
            colors: Enable ANSI console colors (including PyCharm terminals).
                Disable for consoles without ANSI support or redirected output.

        Returns:
            The application namespace logger. Console output goes to stdout to
            avoid IDE-wide red stderr coloring. File output is appended as plain
            UTF-8 text without ANSI colors. Exact level selection applies to both.

        Raises:
            ValueError: If the logging level is invalid.
            OSError: If the log directory or file cannot be opened. Existing
                managed handlers remain in place if file setup fails.

        Calling this method again with ``log_file=None`` disables previously
        configured file output. Timestamps use the local system time.
        """
        if isinstance(levels, (str, bytes)):
            raise ValueError('Pass levels as a list, e.g. ["INFO", "ERROR"]')
        selected = None if levels is None else frozenset(_resolve_level(item) for item in levels)
        if selected == frozenset():
            raise ValueError("levels must contain at least one level")
        resolved_level = _resolve_level(level) if selected is None else min(selected)

        with cls._lock:
            formatter = logging.Formatter(cls.FORMAT, datefmt=cls.DATE_FORMAT)
            logging.addLevelName(SUCCESS, "SUCCESS")
            handlers = []
            try:
                console = logging.StreamHandler(sys.stdout)
                console.set_name(cls._HANDLER_NAMES[0])
                handlers.append(console)
                console.setFormatter(_ColorFormatter(cls.FORMAT, datefmt=cls.DATE_FORMAT) if colors else formatter)
                if log_file is not None:
                    path = Path(log_file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
                    file_handler.set_name(cls._HANDLER_NAMES[1])
                    handlers.append(file_handler)
                    file_handler.setFormatter(formatter)
                for handler in handlers:
                    handler.setLevel(resolved_level)
                    if selected is not None:
                        handler.addFilter(_LevelFilter(selected))
            except Exception:
                for handler in handlers:
                    handler.close()
                raise

            logger = logging.getLogger(cls.LOGGER_NAME)
            for handler in list(logger.handlers):
                if handler.get_name() in cls._HANDLER_NAMES:
                    logger.removeHandler(handler)
                    handler.close()
            # NOTSET on the namespace would inherit the root threshold. Level 1
            # keeps numeric level 0 configurations independent of root logging.
            logger.setLevel(max(1, resolved_level))
            logger.propagate = False
            for handler in handlers:
                logger.addHandler(handler)
            return logger

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> ProjectLogger:
        """Return a logger adapter supporting standard methods plus success().

        Pass ``__name__`` in each module. Existing application-qualified names
        are preserved; other names (including script ``__main__``) are prefixed
        with ``card_suit_duel.``. With no name, return the namespace logger.

        This method does not configure handlers or create files. Call
        ``configure`` in the entry point before emitting application messages.
        """
        if not name:
            return ProjectLogger(logging.getLogger(cls.LOGGER_NAME), {})
        if name == cls.LOGGER_NAME or name.startswith(cls.LOGGER_NAME + "."):
            return ProjectLogger(logging.getLogger(name), {})
        return ProjectLogger(logging.getLogger(cls.LOGGER_NAME + "." + name), {})
