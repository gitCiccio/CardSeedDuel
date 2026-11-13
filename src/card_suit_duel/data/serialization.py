"""Portable row serialization, JSON writing and report construction.

Builders return dictionaries; writers perform filesystem output explicitly.
Inspection imports the same JSON writer instead of maintaining a second copy.
"""
import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from card_suit_duel.data.mapping.definitions import (
    ValidationStatus, INDEX_FIELDS, SUIT_CLASSES, DATASET_SPLITS,
)
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.validation.validation_result import (
    DatasetValidationResult, RowValidationResult,
)
from card_suit_duel.data.validation.sub_validation.validation_paths import public_path

def _count_cards_by_split(
    cards: Sequence[CardImage],
) -> dict[str, int]:
    """Count cards in each original split, including empty splits."""
    counts = Counter(card.split.value for card in cards)

    return {
        split: counts[split]
        for split in DATASET_SPLITS
    }


def _count_cards_by_suit(
    cards: Sequence[CardImage],
) -> dict[str, int]:
    """Count observed suits, representing suitless Joker cards as 'joker'."""
    counts = Counter(
        card.suit.value if card.suit is not None else "joker"
        for card in cards
    )

    return dict(sorted(counts.items()))

def serialize_row(result: RowValidationResult) -> dict:
    """Convert a row result into a portable rejection/review record."""
    return {
        "row_number": result.row_number,
        "original_path": public_path(result.original_path),
        "status": result.status.value,
        "class_index": result.class_index,
        "issues": [
            asdict(issue)
            for issue in result.issues
        ],
    }


def serialize_card(card: CardImage) -> dict[str, str]:
    """Convert card metadata into the shared image-index representation.

    Enum members are exported as their string values.
    Joker's missing suit is represented by an empty CSV field.

    The returned keys match INDEX_FIELDS. The card is not modified.

    Example:
        row = serialize_card(card)
        assert row["image_id"] == card.image_id
        assert row["split"] == card.split.value
    """
    return {
        "image_id": card.image_id,
        "path": card.path,
        "split": card.split.value,
        "original_label": card.original_label,
        "rank": card.rank,
        "suit": card.suit.value if card.suit is not None else "",
    }


def write_json(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the report as readable UTF-8 JSON.

    Parent directories are created when missing.
    An existing report at the same path is overwritten.

    Raises:
        OSError: If the directory or output file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        report_file.write("\n")

def _write_csv(
        output_path: Path,
        fieldnames: Sequence[str],
        rows: Sequence[dict[str, str]]
) -> None:
    """Write CSV records without overwriting an existing file.

        The parent directory must already exist. Headers are written even when
        there are no records, so empty outputs retain a readable schema.
    """
    with output_path.open(
            mode="x",
            encoding="utf-8",
            newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

def build_report(
        rows: list[dict[str, str]],
        summary: dict[str, dict[str, int]],
        files_by_split: dict[str, list[Path]],
        path_comparison: dict[str, Any],
        image_summary: dict[str, Any],
        *,
        dataset_root: Path,
        metadata_path: Path,
        project_root: Path,
) -> dict[str, Any]:
    """Build a report from computed results; paths must be inside project_root."""
    return {
        "schema_version": 1,
        "inspection_scope": "csv_metadata_filesystem_and_image_properties",
        "dataset_root": dataset_root.relative_to(
            project_root
        ).as_posix(),
        "metadata": {
            "source": metadata_path.relative_to(
                project_root
            ).as_posix(),
            "row_count": len(rows),
            "distinct_class_count": sum(
                label != "<missing>"
                for label in summary["rows_by_class"]
            ),
            "rows_by_split": summary["rows_by_split"],
            "rows_by_class": summary["rows_by_class"],
        },
        "anomalies": path_comparison,
        "filesystem": {
            "image_file_count": sum(
                len(paths)
                for paths in files_by_split.values()
            ),
            "image_files_by_split": {
                split: len(paths)
                for split, paths in files_by_split.items()
            },
        },
        "images": image_summary,
    }


def build_validation_summary(result: DatasetValidationResult, run_id: str) -> dict:
    """Build portable run counts and policy metadata without writing files."""
    counts = Counter(row.status.value for row in result.rows)
    issue_counts = Counter(
        issue.code
        for row in result.rows
        for issue in row.issues
    )

    accepted_rows = [
        row
        for row in result.rows
        if row.status is ValidationStatus.ACCEPTED
    ]

    summary = {
        "pipeline": "dataset_validation",
        "pipeline_version": 1,
        "run_id": run_id,
        "source_sha256": result.source_sha256,
        "policy": {
            "metadata_normalization": "strict",
            "identical_references": "keep_first_valid",
            "conflicting_metadata": "review",
            "filter_game_suits": False,
            "decode_images": True,
        },
        "row_count": len(result.rows),
        "accepted_count": counts["accepted"],
        "rejected_count": counts["rejected"],
        "review_count": counts["review"],
        "issue_counts": dict(sorted(issue_counts.items())),
        "unreferenced_image_count": len(
            result.unreferenced_images
        ),
        "accepted_by_split": dict(sorted(Counter(
            row.card.split.value
            for row in accepted_rows
        ).items())),
        "accepted_index": "validated_images.csv",
        "ready_for_next_phase": (
            counts["review"] == 0
            and not result.unreferenced_images
            and bool(accepted_rows)
        ),
    }

    return summary

def build_filtering_summary(
    included: Sequence[CardImage],
    excluded: Sequence[CardImage],
    *,
    run_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build a portable summary of the game-domain filtering phase.

    Counts describe the partition produced by filter_game_cards.
    Original split membership is preserved.

    Args:
        included: Cards belonging to the allowed game suits.
        excluded: Valid dataset cards excluded from the game.
        run_id: Identifier of this filtering execution.
        source_sha256: SHA-256 fingerprint of the input validated CSV.

    Returns:
        JSON-compatible counts, policy and output filenames.

    Example:
        summary = build_filtering_summary(
            included,
            excluded,
            run_id="example_run",
            source_sha256=source_hash,
        )
        assert summary["input_count"] == (
            summary["included_count"] + summary["excluded_count"]
        )
    """
    return {
        "pipeline": "game_suit_filtering",
        "pipeline_version": 1,
        "run_id": run_id,
        "source_sha256": source_sha256,
        "policy": {
            "allowed_suits": [
                suit.value
                for suit in SUIT_CLASSES
            ],
            "excluded_suits": ["diamonds"],
            "exclude_joker": True,
            "preserve_original_splits": True,
        },
        "input_count": len(included) + len(excluded),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "included_by_split": _count_cards_by_split(included),
        "excluded_by_split": _count_cards_by_split(excluded),
        "included_by_suit": _count_cards_by_suit(included),
        "excluded_by_suit": _count_cards_by_suit(excluded),
        "included_index": "filtered_images.csv",
        "excluded_index": "excluded_images.csv",
    }

def export_validation(
    result: DatasetValidationResult,
    output_directory: Path,
) -> dict:
    """Save a run-specific accepted index and validation reports.

    The destination must not already exist, preventing accidental
    replacement of an earlier run.

    The accepted index is provisional when review rows or unreferenced
    images exist. Callers must inspect ready_for_next_phase.
    """
    output_directory.mkdir(parents=True, exist_ok=False)

    accepted_rows = [
        row
        for row in result.rows
        if row.status is ValidationStatus.ACCEPTED
    ]

    index_path = output_directory / "validated_images.csv"

    with index_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=INDEX_FIELDS,
        )
        writer.writeheader()

        for row in accepted_rows:
            card = row.card

            writer.writerow(serialize_card(card))

    for status, filename in (
        (ValidationStatus.REJECTED, "rejected_rows.json"),
        (ValidationStatus.REVIEW, "review_rows.json"),
    ):
        write_json(
            output_path=output_directory / filename,
            report=[
                serialize_row(row)
                for row in result.rows
                if row.status is status
            ],
        )

    write_json(
        output_path=output_directory / "unreferenced_images.json",
        report=list(result.unreferenced_images),
    )

    summary = build_validation_summary(result, output_directory.name)

    write_json(report=summary, output_path=output_directory / "summary.json")

    return summary

def export_filtered_images(
    cards: Sequence[CardImage],
    output_path: Path,
) -> None:
    """Export included cards using the shared image-index schema.

    Card order and original split membership are preserved.
    The parent directory must exist; existing files are not overwritten.

    Example:
        export_filtered_images(
            included,
            output_directory / "filtered_images.csv",
        )
    """
    rows = [serialize_card(card) for card in cards]

    _write_csv(output_path, INDEX_FIELDS, rows)


def _require_excluded_card(card: CardImage) -> None:
    """Reject allowed game suits before creating an exclusion CSV."""
    if card.suit in SUIT_CLASSES:
        raise ValueError(
            "Cannot export an allowed card as excluded: "
            f"{card.image_id!r}."
        )


def export_excluded_images(
    cards: Sequence[CardImage],
    output_path: Path,
) -> None:
    """Export excluded cards with an explicit game-domain exclusion reason.

    Joker uses 'joker'; suits outside SUIT_CLASSES use 'excluded_suit'.
    An allowed card raises ValueError because it indicates an incorrect
    partition. All records are checked before the output file is created.

    Example:
        export_excluded_images(
            excluded,
            output_directory / "excluded_images.csv",
        )
    """
    rows: list[dict[str, str]] = []

    for card in cards:
        _require_excluded_card(card)

        row = serialize_card(card)
        row["exclusion_reason"] = (
            "joker" if card.suit is None else "excluded_suit"
        )
        rows.append(row)

    fieldnames = (*INDEX_FIELDS, "exclusion_reason")

    _write_csv(output_path, fieldnames, rows)