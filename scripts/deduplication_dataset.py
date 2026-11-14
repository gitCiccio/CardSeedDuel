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


def _serialize_groups(result, fingerprints) -> list[dict]:
    """Describe exact groups, member file hashes and retention decisions."""
    indexed = {fp.image_id: fp for fp in fingerprints}
    return [
        {
            "pixels_sha256": group.pixels_sha256,
            "requires_review": group.requires_review,
            "cross_split": len({card.split for card in group.members}) > 1,
            "representative_id": (
                group.representative.image_id if group.representative else None
            ),
            "members": [
                {
                    **serialize_card(card),
                    "file_sha256": indexed[card.image_id].file_sha256,
                    "decision": (
                        "review" if group.requires_review else
                        "retained" if card == group.representative else "removed"
                    ),
                }
                for card in group.members
            ],
        }
        for group in result.groups
    ]


def _export_removed(result, output_path: Path) -> None:
    """Record confirmed exclusions and the representative retained for each."""
    representatives = {
        card.image_id: group.representative.image_id
        for group in result.groups
        if not group.requires_review
        for card in group.members
    }
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(*INDEX_FIELDS, "removal_reason", "representative_id"),
        )
        writer.writeheader()
        for card in result.removed:
            writer.writerow({
                **serialize_card(card),
                "removal_reason": "exact_pixel_duplicate",
                "representative_id": representatives[card.image_id],
            })


def _build_summary(cards, result, groups, run_id: str, source_hash: str) -> dict:
    """Build counts and explicit policy; review blocks downstream use."""
    partitions = {
        "input": cards, "retained": result.retained,
        "removed": result.removed, "review": result.review,
    }
    summary = {
        "pipeline": "exact_image_deduplication",
        "pipeline_version": 1,
        "run_id": run_id,
        "source_sha256": source_hash,
        "policy": {
            "group_by": "pixels_sha256",
            "pixel_hash": "sha256(json_mode_width_height + null_byte + raw_pixels)",
            "color_mode": "RGB",
            "split_priority": ["test", "valid", "train"],
            "tie_break": "lexical_image_id",
            "metadata_fields": ["original_label", "rank", "suit"],
            "conflicting_metadata": "review_entire_group",
            "delete_original_files": False,
            "perceptual_similarity_checked": False,
        },
        "duplicate_group_count": len(groups),
        "cross_split_group_count": sum(g["cross_split"] for g in groups),
        "review_group_count": sum(g["requires_review"] for g in groups),
        "retained_index": "deduplicated_images.csv",
        "ready_for_next_phase": not result.review and bool(result.retained),
    }
    for name, members in partitions.items():
        counts = Counter(card.split.value for card in members)
        summary[f"{name}_count"] = len(members)
        summary[f"{name}_by_split"] = {split: counts[split] for split in DATASET_SPLITS}
        summary[f"{name}_by_suit"] = dict(sorted(Counter(
            card.suit.value for card in members
        ).items()))
    return summary


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
