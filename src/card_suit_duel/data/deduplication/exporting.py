"""Shared export of exact deduplication decisions and provenance."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from card_suit_duel.data.mapping.definitions import DATASET_SPLITS, INDEX_FIELDS
from card_suit_duel.data.serialization import serialize_card, export_filtered_images, write_json

def serialize_duplicate_groups(result, fingerprints) -> list[dict]:
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


def export_removed_images(result, output_path: Path, reasons=None) -> None:
    """Record confirmed exclusions and the representative retained for each."""
    reasons = reasons or {}
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
                "removal_reason": reasons.get(card.image_id, "exact_pixel_duplicate"),
                "representative_id": representatives[card.image_id],
            })


def build_deduplication_summary(cards, result, groups, run_id: str, source_hash: str) -> dict:
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


def export_deduplication(cards, result, fingerprints, output_directory, source_hash, manual_decisions):
    """Write a fresh deduplication run, including applied review evidence."""
    output_directory.mkdir(parents=True, exist_ok=False)
    groups = serialize_duplicate_groups(result, fingerprints)
    reviewed = {d["pixels_sha256"]: d for d in manual_decisions}
    reasons = {}
    for group in groups:
        if group["pixels_sha256"] in reviewed:
            decision = reviewed[group["pixels_sha256"]]
            group["resolution"] = "manual_visual_review"
            group["resolution_reason"] = decision["reason"]
            for member in group["members"]:
                if member["decision"] == "removed":
                    reasons[member["image_id"]] = decision["reason"]
    export_filtered_images(result.retained, output_directory / "deduplicated_images.csv")
    export_removed_images(result, output_directory / "removed_images.csv", reasons)
    write_json(report=groups, output_path=output_directory / "duplicate_groups.json")
    write_json(report=[g for g in groups if g["requires_review"]], output_path=output_directory / "review_groups.json")
    write_json(report=manual_decisions, output_path=output_directory / "manual_decisions.json")
    summary = build_deduplication_summary(cards, result, groups, output_directory.name, source_hash)
    summary["manual_decisions_applied"] = len(manual_decisions)
    summary["policy"]["manual_conflict_resolution"] = "versioned_fingerprint_bound_decisions"
    payload = json.dumps(sorted((f.image_id, f.file_sha256) for f in fingerprints), separators=(",", ":"))
    summary["source_images_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    write_json(report=summary, output_path=output_directory / "summary.json")
    return summary
