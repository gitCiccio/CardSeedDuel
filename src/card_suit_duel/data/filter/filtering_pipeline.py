"""Coordinate validated-index filtering and run-specific output generation."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.filter.filtering import (
    filter_game_cards,
    load_validate_index,
)
from card_suit_duel.data.serialization import (
    build_filtering_summary,
    export_excluded_images,
    export_filtered_images,
    write_json,
)
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager


logger = LoggingManager.get_logger(__name__)


def _create_run_id() -> str:
    """Build a unique run identifier using UTC time and a random suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return f"{timestamp}_{uuid4().hex[:8]}"


def _calculate_source_hash(index_path: Path) -> str:
    """Fingerprint the exact input bytes without modifying the file."""
    return hashlib.sha256(index_path.read_bytes()).hexdigest()


def run_filtering(
    index_path: Path,
    output_root: Path,
) -> Path:
    """Load, filter and export a validated image index.

    Each execution creates a new output directory. Source images and the
    input CSV remain unchanged. Original card order and splits are preserved.

    Args:
        index_path: Explicit validated_images.csv produced by F2.3.
        output_root: Parent directory containing filtering executions.

    Returns:
        The directory containing this execution's outputs.

    Raises:
        MetadataError: If the input is invalid or changes during loading.
        OSError: If an input or output file cannot be accessed.

    Example:
        output_directory = run_filtering(
            index_path=validated_index,
            output_root=filtering_results,
        )
        assert (output_directory / "summary.json").is_file()
    """
    logger.info("Loading the validated image index")

    source_hash = _calculate_source_hash(index_path)
    cards = load_validate_index(index_path)

    # Detect an input file modified while the pipeline was loading it.
    if _calculate_source_hash(index_path) != source_hash:
        raise MetadataError(
            "The validated image index changed during loading."
        )

    logger.info("Loaded %d validated cards", len(cards))

    included, excluded = filter_game_cards(cards)

    if len(cards) != len(included) + len(excluded):
        raise MetadataError(
            "Filtering counts do not match the input card count."
        )

    summary = build_filtering_summary(
        included,
        excluded,
        run_id=_create_run_id(),
        source_sha256=source_hash,
    )

    output_directory = output_root / summary["run_id"]
    output_directory.mkdir(parents=True, exist_ok=False)

    export_filtered_images(
        included,
        output_directory / "filtered_images.csv",
    )
    export_excluded_images(
        excluded,
        output_directory / "excluded_images.csv",
    )
    write_json(
        report=summary,
        output_path=output_directory / "summary.json",
    )

    logger.success(
        "Filtering completed: %d included, %d excluded",
        len(included),
        len(excluded),
    )

    return output_directory