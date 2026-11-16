"""Run exact-image deduplication and export portable, run-specific reports.

Copy this entry point into scripts/ in the project checkout. Run from the
repository root after installing the project in the selected environment.
Original images and the input CSV are never modified or deleted.
"""

import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.deduplication.exporting import (
    serialize_duplicate_groups as _serialize_groups,
    export_removed_images as _export_removed,
    build_deduplication_summary as _build_summary,
)
from card_suit_duel.data.deduplication.deduplication import deduplicate_cards
from card_suit_duel.data.deduplication.fingerprints import fingerprint_card
from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DATASET_SPLITS, INDEX_FIELDS, SUIT_CLASSES
from card_suit_duel.data.serialization import export_filtered_images, serialize_card, write_json
from card_suit_duel.exceptions import DatasetError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Require explicit input selection, dataset root and output parent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()








def main() -> int:
    """Return 0 for success, 2 for pending review, or 1 for execution errors."""
    args = _parse_arguments()
    LoggingManager.configure(level="INFO")
    try:
        source_hash = hashlib.sha256(args.index.read_bytes()).hexdigest()
        cards = load_validate_index(args.index)
        if hashlib.sha256(args.index.read_bytes()).hexdigest() != source_hash:
            raise ValueError("The input index changed during loading.")
        if any(card.suit not in SUIT_CLASSES for card in cards):
            raise ValueError("The input index contains cards outside the game suits.")

        fingerprints = []
        for number, card in enumerate(cards, start=1):
            fingerprints.append(fingerprint_card(card, args.dataset_root))
            if number % 500 == 0 or number == len(cards):
                logger.info("Fingerprint progress: %d/%d", number, len(cards))

        result = deduplicate_cards(cards, fingerprints)
        if len(cards) != len(result.retained) + len(result.removed) + len(result.review):
            raise ValueError("Deduplication partition counts do not match the input.")
        groups = _serialize_groups(result, fingerprints)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        output = args.output_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        summary = _build_summary(cards, result, groups, run_id, source_hash)

        export_filtered_images(result.retained, output / "deduplicated_images.csv")
        _export_removed(result, output / "removed_images.csv")
        write_json(report=groups, output_path=output / "duplicate_groups.json")
        write_json(report=[g for g in groups if g["requires_review"]], output_path=output / "review_groups.json")
        # Write the summary last: failed exports must not look like complete runs.
        write_json(report=summary, output_path=output / "summary.json")
        LoggingManager.configure(level="INFO", log_file=output / "execution.log")
        logger.info("Run ID: %s", run_id)
        logger.info("Decisions: %d retained, %d removed, %d review", len(result.retained), len(result.removed), len(result.review))
        if not summary["ready_for_next_phase"]:
            logger.warning("Deduplication outputs are provisional; review is required")
            return 2
        logger.success("Exact-image deduplication completed successfully")
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        logger.error("Deduplication failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
