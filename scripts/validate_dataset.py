"""Run the original dataset validation pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.validation.validation_suite import (
    export_validation,
    validate_dataset,
)
from card_suit_duel.exceptions import (
    DatasetStructureError,
    MetadataError,
)
from card_suit_duel.logging_config import LoggingManager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = DATASET_ROOT / "cards.csv"

logger = LoggingManager.get_logger(__name__)


def main() -> int:
    """Validate the dataset and publish portable run artifacts."""
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    output_directory = (
        PROJECT_ROOT
        / "results"
        / "pipeline_output"
        / "validation"
        / run_id
    )

    # Export creates the directory and refuses an existing destination.
    # During validation, messages go only to the console.
    LoggingManager.configure(
        levels=[
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
    )

    logger.info("Dataset validation started")

    try:
        result = validate_dataset(
            DATASET_ROOT,
            METADATA_PATH,
        )
        summary = export_validation(
            result,
            output_directory,
        )

    except (DatasetStructureError, MetadataError, OSError) as exc:
        # Do not publish raw exception messages or tracebacks: filesystem
        # errors may contain absolute personal paths.
        logger.error(
            "Validation could not complete. Error category: %s",
            type(exc).__name__,
        )
        return 1

    # The shared log contains controlled messages and relative references.
    LoggingManager.configure(
        levels=[
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        log_file=output_directory / "execution.log",
    )

    logger.info("Pipeline version: %d", summary["pipeline_version"])
    logger.info("Run ID: %s", run_id)
    logger.info("Input: data/raw/cards.csv")
    logger.info("Input SHA-256: %s", summary["source_sha256"])
    logger.info("CSV rows: %d", summary["row_count"])
    logger.info("Accepted: %d", summary["accepted_count"])
    logger.info("Rejected: %d", summary["rejected_count"])
    logger.info("Review: %d", summary["review_count"])

    logger.info(
        "Output: %s",
        output_directory.relative_to(PROJECT_ROOT).as_posix(),
    )

    if not summary["ready_for_next_phase"]:
        logger.warning(
            "Validation artifacts were generated, but unresolved "
            "findings or an empty index prevent progression."
        )
        return 2

    logger.success("Dataset validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())