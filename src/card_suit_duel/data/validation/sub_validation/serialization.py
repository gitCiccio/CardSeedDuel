"""Portable row serialization, JSON writing and report construction.

Builders return dictionaries; writers perform filesystem output explicitly.
Inspection imports the same JSON writer instead of maintaining a second copy.
"""
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from card_suit_duel.data.mapping.definitions import ValidationStatus, INDEX_FIELDS
from card_suit_duel.data.validation.validation_result import (
    DatasetValidationResult, RowValidationResult,
)
from card_suit_duel.data.validation.sub_validation.validation_paths import public_path


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

            writer.writerow({
                "image_id": card.image_id,
                "path": card.path,
                "split": card.split.value,
                "original_label": card.original_label,
                "rank": card.rank,
                "suit": (
                    card.suit.value
                    if card.suit is not None
                    else ""
                ),
            })

    for status, filename in (
        (ValidationStatus.REJECTED, "rejected_rows.json"),
        (ValidationStatus.REVIEW, "review_rows.json"),
    ):
        write_json(
            output_directory / filename,
            [
                serialize_row(row)
                for row in result.rows
                if row.status is status
            ],
        )

    write_json(
        output_directory / "unreferenced_images.json",
        list(result.unreferenced_images),
    )

    summary = build_validation_summary(result, output_directory.name)

    write_json(output_directory / "summary.json", summary)

    return summary
