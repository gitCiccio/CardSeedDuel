"""Generate nested 20%, 50% and 100% training-pair subsets."""

import argparse
from pathlib import Path

from card_suit_duel.data.pairs.subset_pipeline import (
    run_subset_generation,
)
from card_suit_duel.exceptions import DatasetError
from card_suit_duel.logging_config import LoggingManager


logger = LoggingManager.get_logger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Read source locations, destination and the subset selection seed."""
    parser = argparse.ArgumentParser(
        description="Generate nested training subsets."
    )

    parser.add_argument(
        "--pairs-directory",
        type=Path,
        required=True,
        help="F2.7 run directory containing pair CSVs and summary.json.",
    )
    parser.add_argument(
        "--image-index",
        type=Path,
        required=True,
        help="Deduplicated image index used to generate the source pairs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for subset generation runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for nested training-pair selection.",
    )

    return parser.parse_args()


def main() -> int:
    """Execute subset generation and return zero on success."""
    arguments = _parse_arguments()

    LoggingManager.configure(level="INFO")

    try:
        output_directory = run_subset_generation(
            pairs_directory=arguments.pairs_directory,
            image_index_path=arguments.image_index,
            output_root=arguments.output_root,
            seed=arguments.seed,
        )

        LoggingManager.configure(
            level="INFO",
            log_file=output_directory / "execution.log",
        )

        logger.info("Run ID: %s", output_directory.name)
        logger.info("Subset selection seed: %d", arguments.seed)
        logger.info("Training percentages: 20, 50, 100")
        logger.info("Validation and test remain fixed at the F2.7 source")
        logger.success("Nested training subsets completed successfully")

        return 0

    except (DatasetError, OSError, ValueError) as exc:
        logger.error(
            "Subset generation failed [%s]: %s",
            type(exc).__name__,
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())