"""Run game-suit filtering from an explicitly selected validated index."""

import argparse
from pathlib import Path

from card_suit_duel.data.filter.filtering_pipeline import run_filtering
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager


logger = LoggingManager.get_logger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Read the input index and output parent directory from the CLI."""
    parser = argparse.ArgumentParser(
        description="Filter validated cards to the game suits."
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to the validated_images.csv produced by F2.3.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for filtering execution outputs.",
    )

    return parser.parse_args()


def main() -> int:
    """Configure logging, execute filtering and return a process exit code."""
    arguments = _parse_arguments()

    LoggingManager.configure(level="INFO")

    try:
        output_directory = run_filtering(
            index_path=arguments.index,
            output_root=arguments.output_root,
        )
    except (MetadataError, OSError, ValueError) as exc:
        logger.error(
            "Filtering failed [%s]: %s",
            type(exc).__name__,
            exc,
        )
        return 1

    # Persist a portable completion record after the output directory exists.
    LoggingManager.configure(
        level="INFO",
        log_file=output_directory / "execution.log",
    )
    logger.success("Game-suit filtering completed successfully")
    logger.info("Run ID: %s", output_directory.name)
    logger.info("Counts and input fingerprint are available in summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())