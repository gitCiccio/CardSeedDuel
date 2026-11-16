"""Run the existing data-preparation stages from one portable configuration."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from card_suit_duel.data.inspection import run_inspection
from card_suit_duel.data.validation.validation_suite import validate_dataset, export_validation
from card_suit_duel.data.filter.filtering_pipeline import run_filtering
from card_suit_duel.data.deduplication.pipeline import run_deduplication
from card_suit_duel.data.pairs.pairs_pipeline import run_pair_generation
from card_suit_duel.data.pairs.subset_pipeline import run_subset_generation
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.serialization import write_json
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def _read_object(path):
    """Require a JSON object for stage configuration or reports."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError("Cannot read a preparation configuration/report.") from exc
    if not isinstance(value, dict):
        raise MetadataError("Preparation configuration/report must be a JSON object.")
    return value


def _project_path(project_root, relative):
    """Keep configuration paths portable and contained in the checkout."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise MetadataError("Preparation paths must be relative to the project.")
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root):
        raise MetadataError("A preparation path resolves outside the project.")
    return path


def _hash_file(path):
    """Identify exact index bytes for semantic reproducibility checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value):
    """Hash deterministic JSON data, excluding operational run metadata."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def run_preparation(project_root: Path, config_path: Path) -> Path:
    """Rebuild data indices without modifying raw data or previous runs.

    Operational reports retain unique run IDs. reproducibility.json contains
    stable input/content hashes and seed configuration without timestamps.
    A companion's clean-clone check and model integration remain separate.
    """
    project_root = project_root.resolve(strict=True)
    config = _read_object(config_path)
    if config.get("schema_version") != 1:
        raise MetadataError("Unsupported preparation configuration version.")
    dataset_root = _project_path(project_root, config.get("dataset_root"))
    metadata = _project_path(project_root, config.get("metadata"))
    output_root = _project_path(project_root, config.get("output_root"))
    decision_path = _project_path(project_root, config.get("manual_decisions"))
    if output_root.is_relative_to(dataset_root):
        raise MetadataError("Preparation outputs must remain outside the raw dataset.")
    try:
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))
        counts = {s: config["pairs_per_outcome"][s.value] for s in DatasetSplit}
        seeds = {s: config["pair_seeds"][s.value] for s in DatasetSplit}
        subset_seed = config["subset_seed"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise MetadataError("Invalid preparation seeds, quantities or review input.") from exc
    if not isinstance(decisions, list):
        raise MetadataError("Manual decisions must be a JSON list.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(report=config, output_path=output / "configuration.json")
    logger.success("Preparation: inspection")
    report = run_inspection(dataset_root, metadata, project_root)
    write_json(report=report, output_path=output / "00_inspection.json")
    logger.success("Preparation: validation")
    validation = validate_dataset(dataset_root, metadata)
    validation_dir = output / "01_validation"
    validation_summary = export_validation(validation, validation_dir)
    if not validation_summary["ready_for_next_phase"]:
        raise MetadataError("Validation is not approved for downstream preparation.")
    logger.success("Preparation: suit filtering")
    filtered = run_filtering(validation_dir / "validated_images.csv", output / "02_filtering")
    logger.success("Preparation: exact deduplication and review")
    deduplicated = run_deduplication(filtered / "filtered_images.csv", dataset_root, output / "03_deduplication", manual_decisions=decisions)
    image_index = deduplicated / "deduplicated_images.csv"
    logger.success("Preparation: pairs")
    pairs = run_pair_generation(image_index, output / "04_pairs", pairs_per_outcome=counts, seeds=seeds)
    logger.success("Preparation: nested training subsets")
    subsets = run_subset_generation(pairs, image_index, output / "05_subsets", seed=subset_seed)

    indices = {
        "validated": validation_dir / "validated_images.csv",
        "filtered": filtered / "filtered_images.csv",
        "deduplicated": image_index,
        **{f"pairs_{s.value}": pairs / f"{s.value}_pairs.csv" for s in DatasetSplit},
        **{f"train_{p}": subsets / str(p) / "train_pairs.csv" for p in (20, 50, 100)},
    }
    dedup_summary = _read_object(deduplicated / "summary.json")
    reproducibility = {
        "schema_version": 1,
        "source_csv_sha256": validation.source_sha256,
        "source_game_images_sha256": dedup_summary["source_images_sha256"],
        "manual_decisions": decisions,
        "pairs_per_outcome": config["pairs_per_outcome"],
        "pair_seeds": config["pair_seeds"],
        "subset_seed": subset_seed,
        "index_sha256": {name: _hash_file(path) for name, path in indices.items()},
    }
    write_json(report=reproducibility, output_path=output / "reproducibility.json")
    write_json(report={
        "pipeline": "dataset_preparation", "pipeline_version": 1, "run_id": run_id,
        "semantic_sha256": _canonical_hash(reproducibility),
        "indices": {name: path.relative_to(output).as_posix() for name, path in indices.items()},
        "subset_manifest": (subsets / "summary.json").relative_to(output).as_posix(),
        "data_preparation_complete": True,
        "companion_clone_verified": False,
        "model_integration_verified": False,
    }, output_path=output / "summary.json")
    return output
