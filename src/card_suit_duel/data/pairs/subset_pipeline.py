"""Export nested training subsets while referencing fixed validation/test files.

Place this module in src/card_suit_duel/data/pairs/subset_pipeline.py.
The existing implementation is named subset.py (singular).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pairs_reader import load_pairs
from card_suit_duel.data.pairs.subset import build_nested_subsets
from card_suit_duel.data.serialization import build_pair_split_summary, export_pairs, write_json
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _read_summary(path: Path) -> dict:
    """Read a JSON object without treating malformed metadata as an empty run."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError("Cannot read the source pipeline summary.") from exc
    if not isinstance(value, dict):
        raise MetadataError("The source pipeline summary must be a JSON object.")
    return value


def _hash_file(path: Path) -> str:
    """Identify the exact bytes used as a pipeline input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pair_snapshot(path, cards, split):
    """Validate pair metadata and detect an input changed during loading."""
    fingerprint = _hash_file(path)
    pairs = load_pairs(path, cards, expected_split=split)
    if _hash_file(path) != fingerprint:
        raise MetadataError("A source pair index changed during loading.")
    return pairs, fingerprint


def _require_nested_subsets(source, subsets) -> None:
    """Verify exact sizes, uniqueness, balance and inclusion before exporting."""
    previous_ids: set[str] = set()
    for percentage in (20, 50, 100):
        pairs = subsets[percentage]
        ids = {pair.pair_id for pair in pairs}
        summary = build_pair_split_summary(pairs)
        expected = len(source) * percentage // 100
        if len(pairs) != expected or len(ids) != len(pairs):
            raise MetadataError("Subset counts or pair uniqueness are invalid.")
        if not previous_ids <= ids:
            raise MetadataError("Training subsets are not nested.")
        if any(count != expected // 3 for count in summary["pairs_by_outcome"].values()):
            raise MetadataError("Subset outcomes are not balanced.")
        previous_ids = ids
    if subsets[100] != tuple(source):
        raise MetadataError("The 100% subset must preserve the complete input.")


def run_subset_generation(
    pairs_directory: Path,
    image_index_path: Path,
    output_root: Path,
    *,
    seed: int,
) -> Path:
    """Export 20/50/100 training indices from an existing pair-generation run.

    Validation and test indices are validated, fingerprinted and referenced;
    they are never resampled or copied into percentage directories.
    Manifest paths are relative to the returned run directory.

    Source cards must match the image-index fingerprint recorded by F2.7.
    Output directories are new, and the summary is written after all CSVs.
    A later CLI wrapper may configure execution.log using the returned path.

    Example:
        output = run_subset_generation(
            pairs_directory=pair_run,
            image_index_path=approved_image_index,
            output_root=subset_results,
            seed=123,
        )
        assert (output / "20" / "train_pairs.csv").is_file()
    """
    source_summary = _read_summary(pairs_directory / "summary.json")
    if source_summary.get("pipeline") != "card_pair_generation":
        raise MetadataError("Subset generation requires an F2.7 pair run.")

    image_hash = _hash_file(image_index_path)
    if image_hash != source_summary.get("source_sha256"):
        raise MetadataError("The image index does not match the F2.7 source fingerprint.")
    cards = load_validate_index(image_index_path)
    if _hash_file(image_index_path) != image_hash:
        raise MetadataError("The image index changed during loading.")
    if len(cards) != source_summary.get("source_image_count"):
        raise MetadataError("The image count does not match the F2.7 source.")

    loaded = {}
    fingerprints = {}
    source_paths = {}
    used_images = set()
    for split in DatasetSplit:
        path = pairs_directory / f"{split.value}_pairs.csv"
        pairs, fingerprint = _load_pair_snapshot(path, cards, split)
        metadata = source_summary.get("splits", {}).get(split.value, {})
        if metadata.get("index") != path.name or metadata.get("pair_count") != len(pairs):
            raise MetadataError("A source pair index does not match the F2.7 summary.")
        current_images = {card.image_id for pair in pairs for card in (pair.first_card, pair.second_card)}
        if current_images & used_images:
            raise MetadataError("Source pair splits share image identifiers.")
        used_images.update(current_images)
        loaded[split] = pairs
        fingerprints[split] = fingerprint
        source_paths[split] = path

    train = loaded[DatasetSplit.TRAIN]
    subsets = build_nested_subsets(train, seed=seed)
    _require_nested_subsets(train, subsets)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for percentage in (20, 50, 100):
        directory = output / str(percentage)
        directory.mkdir()
        export_pairs(subsets[percentage], directory / "train_pairs.csv")
        summaries[str(percentage)] = {
            **build_pair_split_summary(subsets[percentage]),
            "index": f"{percentage}/train_pairs.csv",
        }

    # Check fixed evaluation files still match the versions validated above.
    for split, path in source_paths.items():
        if _hash_file(path) != fingerprints[split]:
            raise MetadataError("A source pair file changed during subset generation.")

    summary = {
        "pipeline": "nested_training_subsets",
        "pipeline_version": 1,
        "run_id": run_id,
        "source_run_id": source_summary.get("run_id"),
        "seed": seed,
        "path_base": "this_run_directory",
        "image_index_sha256": image_hash,
        "policy": {
            "percentages": [20, 50, 100],
            "percentage_unit": "training_pairs",
            "nested": True,
            "balanced_outcomes": True,
            "selection": "seeded_prefixes_per_ordered_suit_combination",
            "remainder_assignment": "fixed_suit_combination_order",
            "output_order": "source_train_order",
            "resample_validation_test": False,
        },
        "sources": {
            split.value: {
                "index": Path(os.path.relpath(path.resolve(), output.resolve())).as_posix(),
                "sha256": fingerprints[split],
                "pair_count": len(loaded[split]),
            }
            for split, path in source_paths.items()
        },
        "subsets": summaries,
    }
    write_json(report=summary, output_path=output / "summary.json")
    logger.success("Nested subsets completed: %d / %d / %d pairs", len(subsets[20]), len(subsets[50]), len(subsets[100]))
    return output
