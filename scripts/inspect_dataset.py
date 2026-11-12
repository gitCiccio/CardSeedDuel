"""Inspect the original card dataset without modifying its contents."""
from pathlib import Path
from collections import Counter
from typing import Any
from PIL import Image, UnidentifiedImageError

import csv
import json

from card_suit_duel.exceptions import (
    DatasetStructureError,
    MetadataError
)
from card_suit_duel.logging_config import LoggingManager

# __file__ indica questo script; resolve() restituisce il percorso assoluto.
# Due .parent risalgono da scripts/inspect_dataset.py alla radice del progetto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# L'operatore / unisce le parti dei percorsi usando pathlib, senza concatenare stringhe.
# DATASET_ROOT individua i dati originali; METADATA_PATH il loro CSV;
# LOG_PATH indica dove il logger scrive il file di log (non modifica il dataset).
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = DATASET_ROOT / "cards.csv"
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "inspection.log"
REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "data_quality"
    / "dataset_inspection.json"
)
DATASET_SPLITS = ("train", "valid", "test")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

logger = LoggingManager.get_logger(__name__)

def validate_dataset_paths() -> None:
    """Check that the dataset root and metadata file are available."""
    if not DATASET_ROOT.is_dir():
        raise DatasetStructureError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    if not METADATA_PATH.is_file():
        raise DatasetStructureError(
            f"Dataset metadata file not found: {METADATA_PATH}"
        )

def build_report(
        rows: list[dict[str, str]],
        summary: dict[str, dict[str, int]],
        files_by_split: dict[str, list[Path]],
        path_comparison: dict[str, Any],
        image_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build a report of metadata, image file counts and path differences."""
    # schema_version identifica la versione della struttura del report, sarà utile per estenderla in futuro
    # inspect_scope chiarisce la provenienza dei risultati (il csv)
    # relative_to(PROJECT_ROOT) evita di salvare il percorso personale
    # as_posix() rappresenta i percorsi con /
    # il metodo in question costruisce solo il dizionario, ma non salva il report sul disco
    return {
        "schema_version": 1,
        "inspection_scope": "csv_metadata_filesystem_and_image_properties",
        "dataset_root": DATASET_ROOT.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "metadata": {
            "source": METADATA_PATH.relative_to(
                PROJECT_ROOT
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

def save_report(
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

def read_metadata() -> list[dict[str, str]]:
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
    required_columns = {
        "class index",
        "filepaths",
        "labels",
        "card type",
        "data set"
    }

    try:
        with METADATA_PATH.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file, strict=True)
            columns = reader.fieldnames

            if columns is None:
                raise MetadataError(
                    f"Metadata file is empty: {METADATA_PATH}"
                )

            missing_columns = required_columns - set(columns)
            logger.debug("Missing required columns: %s", sorted(missing_columns))

            if missing_columns:
                raise MetadataError(
                    f"Missing required columns in {METADATA_PATH}: "
                    f"{sorted(missing_columns)}. "
                    f"Found: {columns}"
                )

            rows = list(reader)

    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(
            f"Cannot read metadata file {METADATA_PATH}: {exc}"
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

        # Qui vengono recuperati i valori delle colonne "data set" e "label" per ogni riga del CSV.
        # Se un valore è assente o vuoto viene gestito con i doppi apici
        # strip() rimuove gli spazi iniziali e finali

        split = (row.get("data set") or "").strip()
        label = (row.get("labels") or "").strip()


        # Qui si incrementa il valore di split_count e class_counts se sono presenti
        # Se non sono presenti invece viene incrementata la categoria speciale missing

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

def scan_dataset_files() -> dict[str, list[Path]]:
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
        split_directory = DATASET_ROOT / split

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

def inspect_images(
    files_by_split: dict[str, list[Path]],
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
                DATASET_ROOT
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

def compare_metadata_and_files(
    rows: list[dict[str, str]],
    files_by_split: dict[str, list[Path]],
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
        path.relative_to(DATASET_ROOT).as_posix()
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

def inspection() -> None:
    """Inspect the dataset structure and log the findings."""

    logger.info("Dataset inspection started")
    logger.info("Dataset directory: %s", DATASET_ROOT)

    try:
        validate_dataset_paths()

        metadata_rows = read_metadata()
        metadata_summary = summarize_metadata(metadata_rows)

        files_by_split = scan_dataset_files()
        path_comparison = compare_metadata_and_files(
            metadata_rows,
            files_by_split,
        )
        image_summary = inspect_images(files_by_split)

        report = build_report(
            metadata_rows,
            metadata_summary,
            files_by_split,
            path_comparison,
            image_summary,
        )

        save_report(report, REPORT_PATH)

    except (DatasetStructureError, MetadataError, OSError):
        logger.exception("Dataset inspection failed")
        raise

    logger.success(
        "Inspection report saved to %s",
        REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    )

def main() -> None:
    """Configure logging and run the dataset inspection."""
    LoggingManager.configure(
        levels=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        log_file=LOG_PATH,
    )

    logger.info("Starting dataset inspection script")
    inspection()
    logger.success("Dataset inspection completed")

if __name__ == "__main__":
    main()