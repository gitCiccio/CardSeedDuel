"""Reusable, read-only dataset inspection with explicit input paths.

Functions do not configure logging or assume a checkout location. Call
run_inspection() to build a report, then save_report() to persist it.
Only the CLI configures logs and writes output files.
"""

from card_suit_duel.data.mapping.definitions import DATASET_SPLITS, IMAGE_EXTENSIONS, REQUIRED_COLUMNS
from card_suit_duel.data.validation.sub_validation.serialization import (
    build_report, write_json as save_report,
)

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from card_suit_duel.exceptions import DatasetStructureError, MetadataError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)






def validate_dataset_paths(dataset_root: Path, metadata_path: Path) -> None:
    """Check that the dataset root and metadata file are available."""
    if not dataset_root.is_dir():
        raise DatasetStructureError(
            f"Dataset directory not found: {dataset_root}"
        )

    if not metadata_path.is_file():
        raise DatasetStructureError(
            f"Dataset metadata file not found: {metadata_path}"
        )


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Read the metadata CSV and validate its required columns.

        Returns:
            Metadata rows as dictionaries whose keys are CSV column names.

        Raises:
            MetadataError: If the file cannot be read, decoded or parsed,
                or required columns are missing.
    """

    # "class index": numeric identifier of the original card class.
    # Useful for checking consistency; it is not the duel outcome.
    #
    # "filepaths": image path relative to the original dataset root.
    # Used to locate images and compare CSV entries with files on disk.
    #
    # "labels": original card label, such as "ace of hearts".
    # Used to identify the card and derive/check its suit.
    #
    # "card type": card rank/type, such as "ace", "two" or "king".
    # Retained for descriptive analysis; rank does not determine the outcome.
    #
    # "data set": original split membership: "train", "valid" or "test".
    # Used to count samples and preserve separation between the splits.


    try:
        with metadata_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file, strict=True)
            columns = reader.fieldnames

            if columns is None:
                raise MetadataError(
                    f"Metadata file is empty: {metadata_path}"
                )

            missing_columns = REQUIRED_COLUMNS - set(columns)
            logger.debug("Missing required columns: %s", sorted(missing_columns))

            if missing_columns:
                raise MetadataError(
                    f"Missing required columns in {metadata_path}: "
                    f"{sorted(missing_columns)}. "
                    f"Found: {columns}"
                )

            rows = list(reader)

    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(
            f"Cannot read metadata file {metadata_path}: {exc}"
        ) from exc

    logger.info("Metadata columns: %s", columns)
    logger.info("Metadata rows: %d", len(rows))

    if not rows:
        logger.warning("Metadata contains column names but no data rows")

    return rows


def summarize_metadata(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Count metadata rows by split and original card class.

        Empty or missing values are reported as "<missing>".
        Source rows are not modified.

        Returns:
            Counts grouped by split and original class label.
        """
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for row in rows:


        split = (row.get("data set") or "").strip()
        label = (row.get("labels") or "").strip()



        split_counts[split or "<missing>"] += 1
        class_counts[label or "<missing>"] += 1

    logger.info("Metadata rows by split:")

    for split, count in sorted(split_counts.items()):
        logger.info("  %s: %d", split, count)

    logger.info(
        "Distinct non-empty class labels: %d",
        sum(label != "<missing>" for label in class_counts),
    )

    for label, count in sorted(class_counts.items()):
        logger.debug("  Class %s: %d rows", label, count)

    if "<missing>" in split_counts:
        logger.warning(
            "Rows with missing split: %d",
            split_counts["<missing>"],
        )

    if "<missing>" in class_counts:
        logger.warning(
            "Rows with missing class label: %d",
            class_counts["<missing>"],
        )

    return {
        "rows_by_split": dict(sorted(split_counts.items())),
        "rows_by_class": dict(sorted(class_counts.items())),
    }


def scan_dataset_files(dataset_root: Path) -> dict[str, list[Path]]:
    """Find image files inside each original dataset split.

    Files are selected by extension, without opening or modifying them.
    Missing split directories stop the inspection.

    Returns:
        Image paths grouped by split, sorted for reproducible processing.

    Raises:
        DatasetStructureError: If a required split directory is missing.
        OSError: If a filesystem operation fails.
    """
    files_by_split: dict[str, list[Path]] = {}

    for split in DATASET_SPLITS:
        split_directory = dataset_root / split

        if not split_directory.is_dir():
            raise DatasetStructureError(
                f"Required split directory not found: {split_directory}"
            )

        image_paths = sorted(
            path
            for path in split_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        files_by_split[split] = image_paths

        logger.info(
            "Image files found in %s: %d",
            split,
            len(image_paths),
        )

    logger.info(
        "Total image files found: %d",
        sum(len(paths) for paths in files_by_split.values()),
    )

    return files_by_split


def compare_metadata_and_files(
    rows: list[dict[str, str]],
    files_by_split: dict[str, list[Path]],
    dataset_root: Path,
) -> dict[str, Any]:
    """Compare CSV image references with scanned image file paths.

    Paths are compared relative to the dataset root, using forward slashes.
    Duplicate CSV references are collapsed for this comparison.
    No source files or metadata rows are modified.
    """
    metadata_paths: set[str] = set()
    rows_without_path = 0

    for row in rows:
        raw_path = (row.get("filepaths") or "").strip()

        if not raw_path:
            rows_without_path += 1
            continue

        normalized_path = raw_path.replace("\\", "/")
        metadata_paths.add(normalized_path)

    disk_paths = {
        path.relative_to(dataset_root).as_posix()
        for paths in files_by_split.values()
        for path in paths
    }

    metadata_only = sorted(metadata_paths - disk_paths)
    disk_only = sorted(disk_paths - metadata_paths)

    logger.info(
        "CSV paths not found among scanned images: %d",
        len(metadata_only),
    )
    logger.info(
        "Scanned images not listed in CSV: %d",
        len(disk_only),
    )

    for path in metadata_only:
        logger.warning("CSV reference without scanned image: %s", path)

    for path in disk_only:
        logger.warning("Scanned image without CSV reference: %s", path)

    if rows_without_path:
        logger.warning(
            "Metadata rows without an image path: %d",
            rows_without_path,
        )

    return {
        "csv_paths_without_scanned_image": metadata_only,
        "scanned_images_without_csv_reference": disk_only,
        "rows_without_path": rows_without_path,
    }


def inspect_images(
    files_by_split: dict[str, list[Path]],
    dataset_root: Path,
) -> dict[str, Any]:
    """Inspect image properties and attempt full pixel decoding.

    Source images are never modified. Unreadable images are recorded
    without stopping the inspection of the remaining files.
    """
    dimensions: Counter[str] = Counter()
    color_modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()

    unreadable_images: list[dict[str, str]] = []
    readable_by_split: Counter[str] = Counter()

    total_files = sum(
        len(paths)
        for paths in files_by_split.values()
    )
    processed = 0

    for split, paths in files_by_split.items():
        readable_by_split[split] = 0

        for path in paths:
            relative_path = path.relative_to(
                dataset_root
            ).as_posix()

            try:
                with Image.open(path) as image:
                    image.load()

                    width, height = image.size
                    mode = image.mode
                    image_format = image.format or "<unknown>"

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                unreadable_images.append({
                    "path": relative_path,
                    "error_type": type(exc).__name__,
                })

                logger.warning(
                    "Cannot decode image %s: %s",
                    relative_path,
                    exc,
                )

            else:
                dimensions[f"{width}x{height}"] += 1
                color_modes[mode] += 1
                formats[image_format] += 1
                readable_by_split[split] += 1

            processed += 1

            if processed % 1000 == 0 or processed == total_files:
                logger.info(
                    "Image inspection progress: %d/%d",
                    processed,
                    total_files,
                )

    readable_count = sum(readable_by_split.values())

    logger.info("Readable images: %d", readable_count)
    logger.info("Unreadable images: %d", len(unreadable_images))
    logger.info("Image dimensions: %s", dict(dimensions))
    logger.info("Image color modes: %s", dict(color_modes))
    logger.info("Image formats: %s", dict(formats))

    return {
        "inspected_file_count": processed,
        "readable_image_count": readable_count,
        "readable_images_by_split": dict(
            sorted(readable_by_split.items())
        ),
        "dimensions": dict(sorted(dimensions.items())),
        "color_modes": dict(sorted(color_modes.items())),
        "formats": dict(sorted(formats.items())),
        "unreadable_image_count": len(unreadable_images),
        "unreadable_images": unreadable_images,
    }








def run_inspection(
    dataset_root: Path,
    metadata_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Inspect original data and return a report without writing any files.

    Paths must be inside project_root for portable relative report references.
    Errors propagate to the caller; individual unreadable images are findings.
    """
    validate_dataset_paths(dataset_root, metadata_path)
    rows = read_metadata(metadata_path)
    summary = summarize_metadata(rows)
    files = scan_dataset_files(dataset_root)
    comparison = compare_metadata_and_files(rows, files, dataset_root)
    images = inspect_images(files, dataset_root)
    return build_report(
        rows, summary, files, comparison, images,
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        project_root=project_root,
    )


def main(argv=None, *, project_root: Path | None = None) -> int:
    """Run the CLI after installation; default root is the working directory.

    The repository script supplies its own location as a convenient default.
    Use --project-root to inspect another checkout explicitly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path,
                        default=project_root if project_root is not None else Path.cwd())
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    dataset_root = root / 'data' / 'raw'
    report_path = root / 'results' / 'data_quality' / 'dataset_inspection.json'
    LoggingManager.configure(
        levels=['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'],
        log_file=root / 'results' / 'logs' / 'inspection.log',
    )
    logger.info('Dataset inspection started')
    logger.info('Dataset directory: %s', dataset_root)
    try:
        report = run_inspection(dataset_root, dataset_root / 'cards.csv', root)
        save_report(report, report_path)
    except (DatasetStructureError, MetadataError, OSError) as exc:
        logger.error("Dataset inspection failed [%s]. Check input paths and CSV schema.", type(exc).__name__)
        return 1
    logger.success('Inspection report saved to %s', report_path.relative_to(root).as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
