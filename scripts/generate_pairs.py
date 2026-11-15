"""Generate balanced card-duel CSVs from an approved deduplicated index.

Place this file in scripts/generate_pairs.py and run with the project installed.
Counts are per outcome: a request of 10000 generates 30000 pairs in that split.
"""

import argparse
from pathlib import Path

from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pairs_pipeline import run_pair_generation
from card_suit_duel.exceptions import DatasetError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Read explicit source selection and independently configurable splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    for split, count, seed in (("train", 10000, 42), ("valid", 1000, 43), ("test", 1000, 44)):
        parser.add_argument(f"--{split}-per-outcome", type=int, default=count)
        parser.add_argument(f"--{split}-seed", type=int, default=seed)
    return parser.parse_args()


def main() -> int:
    """Run generation, persist a portable completion log and return a status."""
    args = _parse_arguments()
    LoggingManager.configure(level="INFO")
    counts = {
        split: getattr(args, f"{split.value}_per_outcome")
        for split in DatasetSplit
    }
    seeds = {
        split: getattr(args, f"{split.value}_seed")
        for split in DatasetSplit
    }

    try:
        output = run_pair_generation(
            index_path=args.index,
            output_root=args.output_root,
            pairs_per_outcome=counts,
            seeds=seeds,
        )
        LoggingManager.configure(level="INFO", log_file=output / "execution.log")
        logger.info("Run ID: %s", output.name)
        for split in DatasetSplit:
            logger.info(
                "%s: %d pairs, %d per outcome, seed %d",
                split.value, counts[split] * 3, counts[split], seeds[split],
            )
        logger.success("Card-pair generation completed successfully")
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        logger.error("Pair generation failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
