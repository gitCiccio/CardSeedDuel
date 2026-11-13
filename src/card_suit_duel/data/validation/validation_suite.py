"""Public orchestration for validation; checks live in sub_validation.

validate_dataset returns structured results. export_validation explicitly writes
artifacts. Metadata helpers log per-row progress through LoggingManager;
expected invalid rows become findings without full exception tracebacks.
"""
from pathlib import Path
from card_suit_duel.data.mapping.definitions import DATASET_SPLITS
from card_suit_duel.data.validation.validation_result import DatasetValidationResult
from card_suit_duel.data.validation.sub_validation.validation_metadata import read_input, parse_rows
from card_suit_duel.data.validation.sub_validation.validation_paths import find_path_groups, find_path_conflicts
from card_suit_duel.data.validation.sub_validation.validation_classes import find_class_conflicts, check_result_integrity
from card_suit_duel.data.validation.sub_validation.validation_images import classify_candidates, find_unreferenced_images
from card_suit_duel.exceptions import DatasetStructureError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _is_valid_dir(root: Path) -> None:
    """Require an available dataset directory before processing rows."""
    if not root.is_dir():
        raise DatasetStructureError(
            "Dataset root is missing or is not a directory."
        )


def validate_dataset(
    dataset_root: Path,
    metadata_path: Path,
) -> DatasetValidationResult:
    """Validate original metadata and files without modifying source data.

    Parseable conflicts are classified as review. Invalid rows and files
    are rejected. Identical repeated references retain one accepted entry.

    Diamonds and Joker are not filtered here. Image-content duplicate
    detection belongs to a later pipeline.

    Returns:
        One final decision per CSV row, unreferenced image paths and
        the fingerprint of the exact CSV content parsed.

    Raises:
        DatasetStructureError: Required dataset structure is unavailable.
        MetadataError: The CSV cannot be read or its header is invalid.
        RuntimeError: An internal output invariant is violated.
    """
    root = dataset_root.resolve()

    _is_valid_dir(root)

    # Also verifies that all required split directories exist.
    # The resulting scan is performed again for the final reconciliation.
    for split in DATASET_SPLITS:
        _is_valid_dir(root / split)

    rows, fingerprint = read_input(metadata_path)
    logger.info("Metadata loaded: %d rows", len(rows))
    rejected, candidates = parse_rows(rows)

    groups = find_path_groups(rows)
    path_conflicts = find_path_conflicts(rows, groups)
    class_conflicts = find_class_conflicts(candidates)

    classified = classify_candidates(
        candidates,
        path_conflicts,
        class_conflicts,
        root,
    )

    all_results = {**rejected, **classified}
    ordered_results = tuple(
        all_results[number]
        for number in sorted(all_results)
    )

    check_result_integrity(ordered_results, len(rows))

    logger.info("Validation classified %d rows", len(ordered_results))
    return DatasetValidationResult(
        rows=ordered_results,
        unreferenced_images=find_unreferenced_images(rows, root),
        source_sha256=fingerprint,
    )


# Re-export after definitions to avoid circular initialization with serialization.
from card_suit_duel.data.serialization import export_validation as export_validation
