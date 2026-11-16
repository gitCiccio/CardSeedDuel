"""Shared filesystem checks for dataset image references.

These functions resolve paths and optionally check file existence; they never
decode pixels. CSV field checks belong to metadata_checks, not this module.
"""

from pathlib import Path

from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import DatasetStructureError, ImageValidationError


def resolve_dataset_root(dataset_root: Path) -> Path:
    """Require an existing dataset directory and return its resolved path."""
    try:
        root = dataset_root.resolve(strict=True)

        if not root.is_dir():
            raise DatasetStructureError(
                "Dataset root is not a directory."
            )

    except (OSError, RuntimeError) as exc:
        raise DatasetStructureError(
            "Cannot access the dataset root."
        ) from exc

    return root


def _resolve_image_reference(
    card: CardImage,
    root: Path,
) -> Path:
    """Resolve the image reference, following any symbolic links."""
    try:
        return (root / card.path).resolve()

    except (OSError, RuntimeError) as exc:
        raise ImageValidationError(
            f"Cannot resolve image: {card.path}",
            reason="unresolvable_path",
            image_path=card.path,
        ) from exc


def _require_contained_path(
    image_path: Path,
    root: Path,
    original_path: str,
) -> None:
    """Reject resolved image paths outside the dataset directory."""
    if not image_path.is_relative_to(root):
        raise ImageValidationError(
            f"Image resolves outside the dataset: {original_path}",
            reason="outside_dataset",
            image_path=original_path,
        )


def _require_image_file(
    image_path: Path,
    original_path: str,
) -> None:
    """Require an existing regular file at the resolved location."""
    try:
        exists = image_path.is_file()

    except OSError as exc:
        raise ImageValidationError(
            f"Cannot access image: {original_path}",
            reason="file_access_error",
            image_path=original_path,
        ) from exc

    if not exists:
        raise ImageValidationError(
            f"Image file is missing: {original_path}",
            reason="missing_file",
            image_path=original_path,
        )


def resolve_card_reference(card: CardImage, dataset_root: Path) -> Path:
    """Resolve an image reference and enforce dataset containment.

    The dataset root must exist. The image file need not exist: callers such
    as fingerprinting may report existence/access failures during byte reading.
    Symbolic links are followed before checking containment.
    """
    root = resolve_dataset_root(dataset_root)
    image_path = _resolve_image_reference(card, root)
    _require_contained_path(image_path, root, card.path)
    return image_path


def resolve_card_path(card: CardImage, dataset_root: Path) -> Path:
    """Return a contained, existing image-file path without decoding pixels.

    Missing roots raise DatasetStructureError. Invalid image references raise
    ImageValidationError, preserving the relative path and reason code.
    """
    image_path = resolve_card_reference(card, dataset_root)
    _require_image_file(image_path, card.path)
    return image_path
