"""Path grouping, metadata conflicts and portable path presentation."""
from collections import defaultdict
from pathlib import PurePosixPath, PureWindowsPath
from card_suit_duel.data.mapping.definitions import REQUIRED_COLUMNS

from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def path_key(value: str | None) -> str | None:
    """Normalize separators for grouping references without accessing files."""
    if not isinstance(value, str) or not value.strip():
        logger.warning("Invalid or empty path provided.")
        return None

    return value.strip().replace("\\", "/")


def public_path(value: str | None) -> str | None:
    """Avoid exporting absolute local paths in shared reports."""
    path = path_key(value)

    if path is None:
        logger.warning("Invalid or empty path provided for publication.")
        return None

    if (
            PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).drive
    ):
        logger.warning("Absolute path detected; replacing with placeholder.")
        return "<non-relative path omitted>"

    return path


def find_path_groups(rows: list[dict]) -> dict[str, list[int]]:
    """Group all CSV references, including rows rejected during parsing."""
    groups = defaultdict(list)

    for row_number, row in enumerate(rows, start=2):
        key = path_key(row.get("filepaths"))

        if key is not None:
            groups[key].append(row_number)

    return dict(groups)


def find_path_conflicts(
    rows: list[dict],
    groups: dict[str, list[int]],
) -> set[str]:
    """Find repeated paths whose required metadata values disagree.

    A malformed row remains rejected. Other parseable rows referring to
    the same conflicting path are marked for review.
    """
    conflicts = set()

    for path, row_numbers in groups.items():
        signatures = {
            tuple(
                rows[number - 2].get(field)
                for field in sorted(REQUIRED_COLUMNS)
                if field != "filepaths"
            )
            for number in row_numbers
        }

        if len(signatures) > 1:
            conflicts.add(path)

    return conflicts
