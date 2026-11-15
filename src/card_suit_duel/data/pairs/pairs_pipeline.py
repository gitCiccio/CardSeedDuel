"""Coordinate reproducible duel generation and run-specific exports."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pairs_generator import generate_pairs_for_split
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.data.serialization import (
    build_pair_split_summary,
    export_pairs,
    write_json,
)
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager


logger = LoggingManager.get_logger(__name__)

def _read_source_summary(index_path: Path) -> dict:
    """Require an approved deduplication output matching the selected index."""
    summary_path = index_path.parent / "summary.json"

    try:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError(
            "Cannot read the source deduplication summary."
        ) from exc

    if not isinstance(summary, dict):
        raise MetadataError(
            "The source summary must be a JSON object."
        )

    if summary.get("pipeline") != "exact_image_deduplication":
        raise MetadataError(
            "Pair generation requires a deduplication output."
        )

    if summary.get("ready_for_next_phase") is not True:
        raise MetadataError(
            "The source dataset is not approved for the next phase."
        )

    if summary.get("retained_index") != index_path.name:
        raise MetadataError(
            "The selected index does not match the source summary."
        )

    return summary

def _require_disjoint_split_images(
    pairs_by_split: Mapping[
        DatasetSplit,
        Sequence[CardPair],
    ],
) -> None:
    """Reject image identifiers appearing in more than one generated split."""
    previously_used: set[str] = set()

    for split in DatasetSplit:
        current_images = {
            card.image_id
            for pair in pairs_by_split[split]
            for card in (pair.first_card, pair.second_card)
        }

        if previously_used & current_images:
            raise MetadataError(
                "Generated splits share image identifiers."
            )

        previously_used.update(current_images)

def run_pair_generation(
    index_path: Path,
    output_root: Path,
    *,
    pairs_per_outcome: Mapping[DatasetSplit, int],
    seeds: Mapping[DatasetSplit, int],
) -> Path:
    """Generate and export balanced duels from an approved image index.

    Each split uses an independent seed and requested outcome count.
    All pairs are generated before creating the output directory.

    Source images and the input index remain unchanged.
    Existing run directories are never overwritten.

    Args:
        index_path: Approved deduplicated_images.csv.
        output_root: Parent directory for generation outputs.
        pairs_per_outcome: Requested count for each outcome in every split.
        seeds: Independent integer seed for every split.

    Returns:
        The directory containing the generated CSVs and summary.

    Raises:
        MetadataError: Source metadata, configuration or generated
            collections violate the expected contract.
        OSError: Input or output files cannot be accessed.
    """
    expected_splits = set(DatasetSplit)

    if set(pairs_per_outcome) != expected_splits:
        raise MetadataError(
            "Pair counts must be provided for train, valid and test."
        )

    if set(seeds) != expected_splits:
        raise MetadataError(
            "Seeds must be provided for train, valid and test."
        )

    source_summary = _read_source_summary(index_path)
    source_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()

    cards = load_validate_index(index_path)

    if hashlib.sha256(index_path.read_bytes()).hexdigest() != source_hash:
        raise MetadataError(
            "The source image index changed during loading."
        )

    if len(cards) != source_summary.get("retained_count"):
        raise MetadataError(
            "The image count does not match the source summary."
        )

    pairs_by_split: dict[
        DatasetSplit,
        tuple[CardPair, ...],
    ] = {}

    split_summaries: dict[str, dict] = {}

    for split in DatasetSplit:
        logger.info("Generating pairs for split: %s", split.value)

        pairs = generate_pairs_for_split(
            cards,
            split=split,
            pairs_per_outcome=pairs_per_outcome[split],
            seed=seeds[split],
        )

        split_summary = build_pair_split_summary(pairs)

        if (
            split_summary["pair_count"]
            != split_summary["unique_pair_count"]
        ):
            raise MetadataError(
                f"Repeated ordered pairs in split {split.value!r}."
            )

        pairs_by_split[split] = pairs
        split_summaries[split.value] = {
            **split_summary,
            "seed": seeds[split],
            "requested_pairs_per_outcome": pairs_per_outcome[split],
            "index": f"{split.value}_pairs.csv",
        }

    _require_disjoint_split_images(pairs_by_split)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    output_directory = output_root / run_id
    output_directory.mkdir(parents=True, exist_ok=False)

    for split in DatasetSplit:
        export_pairs(
            pairs_by_split[split],
            output_directory / f"{split.value}_pairs.csv",
        )

    summary = {
        "pipeline": "card_pair_generation",
        "pipeline_version": 1,
        "run_id": run_id,
        "source_run_id": source_summary.get("run_id"),
        "source_sha256": source_hash,
        "source_image_count": len(cards),
        "policy": {
            "ordered_pairs": True,
            "allow_self_pairs": False,
            "allow_repeated_ordered_pairs": False,
            "preserve_original_splits": True,
            "balance_outcomes": True,
            "balance_suit_combinations": True,
            "insufficient_capacity": "error",
            "oracle": "determine_outcome",
        },
        "splits": split_summaries,
    }

    # Write the summary after exporting all three datasets.
    write_json(
        report=summary,
        output_path=output_directory / "summary.json",
    )

    logger.success("Pair generation completed")

    return output_directory