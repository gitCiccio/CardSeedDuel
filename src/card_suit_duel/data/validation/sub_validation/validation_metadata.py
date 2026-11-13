"""CSV snapshot reading and per-row metadata conversion.

Expected invalid rows become structured findings, not terminal tracebacks.
"""
import csv
import hashlib
import io
from pathlib import Path
from card_suit_duel.data.metadata_checks import validate_required_columns, validate_record_shape
from card_suit_duel.data.mapping.definitions import REQUIRED_COLUMNS, ValidationStatus
from card_suit_duel.data.validation.validation import parse_card_row
from card_suit_duel.data.validation.validation_result import RowValidationResult, ValidationIssue
from card_suit_duel.exceptions import MetadataError

from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def read_input(
    metadata_path: Path
) -> tuple[list[dict], str]:
    """Read one CSV snapshot and calculate its SHA-256 fingerprint.

        Missing or duplicate column names stop the pipeline.
        Malformed individual records are handled during row validation.
    """
    logger.info("Reading metadata CSV: %s", metadata_path.name)
    try:
        content = metadata_path.read_bytes()
        text = content.decode("utf-8-sig")

        reader = csv.DictReader(io.StringIO(text, newline=""),
                                strict=True
                                )

        columns = reader.fieldnames

        logger.info("Checking the columns: %s", columns)
        missing = REQUIRED_COLUMNS - set(columns or ())
        logger.info("Checking for missing required columns: %s", sorted(missing))
        validate_required_columns(columns, REQUIRED_COLUMNS, source_name="Metadata")

        rows = list(reader)
        logger.info("Read %d rows from metadata CSV: %s", len(rows), metadata_path.name)

    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(
            "Cannot read or parse the metadata CSV."
        ) from exc

    fingerprint = hashlib.sha256(content).hexdigest()

    return rows, fingerprint


def parse_rows(
        rows: list[dict]
) -> tuple[dict[int, RowValidationResult], dict[int, tuple]]:
    """Parse all rows while preserving failures and original record numbers.

        Candidate tuples contain CardImage and original numeric class index.
        Error messages are deliberately generic in shareable artifacts.
    """

    candidates = {}
    rejected = {}

    for row_number, row in enumerate(rows, start=2):
        original_path = row.get("filepaths")

        try:
            # DictReader uses a None key for excess fields and None values
            # when a record has fewer fields than the header.
            validate_record_shape(row, source_name="Metadata")

            card = parse_card_row(row)
            class_index = int(row["class index"])
            logger.info("Row %d parsed successfully: %s, with class index %d", row_number, card, class_index)

        except MetadataError:
            logger.warning("Invalid metadata in row %d", row_number)
            rejected[row_number] = RowValidationResult(
                row_number=row_number,
                original_path=original_path,
                status=ValidationStatus.REJECTED,
                issues=(
                    ValidationIssue(
                        code="invalid_metadata",
                        message=(
                            "Required metadata is missing, invalid "
                            "or inconsistent."
                        ),
                    ),
                ),
            )
        else:
            candidates[row_number] = (card, class_index)
            logger.info("Candidate %s added in row: %d", (card, class_index), row_number)

    return rejected, candidates
