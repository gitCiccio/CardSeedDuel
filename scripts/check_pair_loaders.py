"""Check neural pair batches using an F2.8 manifest, without training models.

Copy into scripts/check_pair_loaders.py. By default one batch per split is
checked; report.json explicitly records this limited image-inspection scope.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from uuid import uuid4

import torch

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DatasetSplit, OUTCOME_CLASSES
from card_suit_duel.data.pairs.pair_dataset import CardPairDataset
from card_suit_duel.data.pairs.pair_loader import build_pair_loader
from card_suit_duel.data.pairs.pairs_reader import load_pairs
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.data.serialization import write_json
from card_suit_duel.exceptions import DatasetError, MetadataError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _parse_arguments():
    """Select a training percentage and the batch inspection scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--percentage", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=1)
    return parser.parse_args()


def _hash_file(path):
    """Fingerprint exact source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path):
    """Require the explicit F2.8 manifest schema used by this checker."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError("Cannot read the subset manifest.") from exc
    if not isinstance(data, dict) or data.get("pipeline") != "nested_training_subsets":
        raise MetadataError("Expected an F2.8 nested-training-subsets manifest.")
    if data.get("path_base") != "this_run_directory":
        raise MetadataError("Unsupported manifest path base.")
    for field in ("sources", "subsets"):
        if not isinstance(data.get(field), dict):
            raise MetadataError(f"The manifest is missing {field!r}.")
    return data


def _resolve_manifest_index(manifest_path, entry):
    """Resolve portable file references relative to the manifest directory."""
    if not isinstance(entry, dict):
        raise MetadataError("The manifest index entry must be an object.")
    relative = entry.get("index")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise MetadataError("The manifest must provide a relative index path.")
    return (manifest_path.parent / relative).resolve(strict=True)


def _load_pair_snapshot(path, cards, split, entry):
    """Check source identity, parse all pair metadata and verify row count."""
    fingerprint = _hash_file(path)
    if "sha256" in entry and fingerprint != entry["sha256"]:
        raise MetadataError(f"Source fingerprint mismatch for {split.value}.")
    pairs = load_pairs(path, cards, expected_split=split)
    if _hash_file(path) != fingerprint:
        raise MetadataError("A pair CSV changed while being read.")
    if len(pairs) != entry.get("pair_count"):
        raise MetadataError(f"Manifest count mismatch for {split.value}.")
    return pairs, fingerprint


def _load_experiment(args, manifest):
    """Load the selected train and fixed evaluation data with provenance checks."""
    image_hash = _hash_file(args.image_index)
    if image_hash != manifest.get("image_index_sha256"):
        raise MetadataError("The image index does not match the subset manifest.")
    cards = load_validate_index(args.image_index)
    if _hash_file(args.image_index) != image_hash:
        raise MetadataError("The image index changed while being read.")

    originals = {}
    hashes = {}
    used_images = set()
    for split in DatasetSplit:
        entry = manifest["sources"].get(split.value)
        path = _resolve_manifest_index(args.subset_manifest, entry)
        pairs, fingerprint = _load_pair_snapshot(path, cards, split, entry)
        current = {c.image_id for p in pairs for c in (p.first_card, p.second_card)}
        if current & used_images:
            raise MetadataError("Source splits share image identifiers.")
        used_images.update(current)
        originals[split] = pairs
        hashes[f"source_{split.value}"] = fingerprint

    selected_entry = manifest["subsets"].get(str(args.percentage))
    selected_path = _resolve_manifest_index(args.subset_manifest, selected_entry)
    selected, fingerprint = _load_pair_snapshot(selected_path, cards, DatasetSplit.TRAIN, selected_entry)
    source_ids = {p.pair_id for p in originals[DatasetSplit.TRAIN]}
    if not {p.pair_id for p in selected} <= source_ids:
        raise MetadataError("Selected training pairs are not a subset of the source.")
    counts = Counter(p.outcome.value for p in selected)
    expected = len(originals[DatasetSplit.TRAIN]) * args.percentage // 100
    if len(selected) != expected or any(counts[o.value] != expected // 3 for o in OUTCOME_CLASSES):
        raise MetadataError("Selected training counts or outcome balance are invalid.")
    originals[DatasetSplit.TRAIN] = selected
    hashes["selected_train"] = fingerprint
    return originals, hashes


def _check_batch(batch):
    """Require compatible finite image tensors and valid integer targets."""
    first, second, targets = batch["first_image"], batch["second_image"], batch["target"]
    if first.ndim != 4 or first.shape[1] != 3 or first.shape != second.shape:
        raise MetadataError("Expected equally shaped image batches [B, 3, H, W].")
    for tensor in (first, second):
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise MetadataError("Image tensors must be finite float32 values.")
        if tensor.min().item() < 0 or tensor.max().item() > 1:
            raise MetadataError("The current preprocessing must produce values in [0, 1].")
    if targets.dtype != torch.long or tuple(targets.shape) != (first.shape[0],):
        raise MetadataError("Expected a target batch with dtype long and shape [B].")
    if targets.min().item() < 0 or targets.max().item() >= len(OUTCOME_CLASSES):
        raise MetadataError("A target is outside the shared outcome mapping.")


def _inspect_split(pairs, args, seed):
    """Inspect up to max_batches; do not claim all images were decoded."""
    dataset = CardPairDataset(pairs, args.dataset_root, transform=build_image_transform())
    loader = build_pair_loader(dataset, batch_size=args.batch_size, seed=seed)
    sizes = []
    checked_targets = Counter()
    first_ids = []
    first_shape = None
    image_min, image_max = 1.0, 0.0
    for batch in islice(loader, args.max_batches):
        _check_batch(batch)
        sizes.append(len(batch["pair_id"]))
        checked_targets.update(batch["target"].tolist())
        if first_shape is None:
            first_shape = list(batch["first_image"].shape)
            first_ids = list(batch["pair_id"][:3])
        for key in ("first_image", "second_image"):
            image_min = min(image_min, batch[key].min().item())
            image_max = max(image_max, batch[key].max().item())
    return {
        "dataset_pair_count": len(dataset),
        "available_batch_count": len(loader),
        "checked_batch_count": len(sizes),
        "checked_pair_count": sum(sizes),
        "all_pairs_decoded": sum(sizes) == len(dataset),
        "observed_batch_sizes": sizes,
        "first_batch_image_shape": first_shape,
        "image_dtype": "float32",
        "target_dtype": "int64",
        "observed_image_range": [image_min, image_max],
        "checked_pairs_by_outcome": {o.value: checked_targets[i] for i, o in enumerate(OUTCOME_CLASSES)},
        "first_pair_ids": first_ids,
        "shuffle": dataset.split is DatasetSplit.TRAIN,
        "drop_last": False,
        "num_workers": 0,
        "seed": seed,
    }


def main():
    """Write a run report only after all requested batch checks pass."""
    args = _parse_arguments()
    LoggingManager.configure(level="INFO")
    try:
        if args.max_batches <= 0:
            raise MetadataError("max-batches must be positive.")
        manifest = _read_manifest(args.subset_manifest)
        pairs_by_split, hashes = _load_experiment(args, manifest)
        results = {}
        for i, split in enumerate(DatasetSplit):
            logger.info("Checking %s batches", split.value)
            results[split.value] = _inspect_split(pairs_by_split[split], args, args.seed + i)
            logger.info("%s: %d pairs checked", split.value, results[split.value]["checked_pair_count"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{uuid4().hex[:8]}"
        output = args.output_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "pipeline": "pair_loader_check", "pipeline_version": 1, "run_id": run_id,
            "subset_run_id": manifest.get("run_id"), "training_percentage": args.percentage,
            "scope": "limited_batches_per_split", "max_batches_per_split": args.max_batches,
            "batch_size": args.batch_size, "input_hashes": hashes,
            "versions": {name: version(name) for name in ("torch", "torchvision", "Pillow")},
            "preprocessing": {"resize": None, "dtype": "float32", "scale": "uint8_to_zero_one", "augmentation": False},
            "splits": results,
        }
        write_json(report=report, output_path=output / "report.json")
        LoggingManager.configure(level="INFO", log_file=output / "execution.log")
        logger.info("Run ID: %s", run_id)
        logger.info("Training percentage: %d", args.percentage)
        for split, result in results.items():
            logger.info("%s: %d/%d pairs decoded", split, result["checked_pair_count"], result["dataset_pair_count"])
        logger.success("Requested batch checks completed; no model was trained")
        return 0
    except (DatasetError, OSError, ValueError, RuntimeError) as exc:
        logger.error("Loader check failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
