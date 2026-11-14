"""Compute exact file and decoded-pixel fingerprints for dataset images."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import (
    DatasetStructureError,
    ImageValidationError,
)


@dataclass(frozen=True)
class ImageFingerprint:
    """Store exact-content fingerprints associated with one image.

    file_sha256 identifies identical file bytes.
    pixels_sha256 identifies identical decoded pixels, mode and dimensions.

    Neither fingerprint measures perceptual similarity.
    """

    image_id: str
    file_sha256: str
    pixels_sha256: str


def _resolve_image_path(
    card: CardImage,
    dataset_root: Path,
) -> Path:
    """Resolve an image path and require containment within the dataset.

    Symbolic links are resolved before containment is checked.
    A missing dataset root stops the operation.
    """
    try:
        root = dataset_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DatasetStructureError(
            "Cannot resolve the dataset root."
        ) from exc

    if not root.is_dir():
        raise DatasetStructureError(
            "Dataset root is not a directory."
        )

    try:
        image_path = (root / card.path).resolve()
    except (OSError, RuntimeError) as exc:
        raise ImageValidationError(
            f"Cannot resolve image: {card.path}",
            reason="unresolvable_path",
            image_path=card.path,
        ) from exc

    if not image_path.is_relative_to(root):
        raise ImageValidationError(
            f"Image resolves outside the dataset: {card.path}",
            reason="outside_dataset",
            image_path=card.path,
        )

    return image_path

def _read_image_bytes(
    image_path: Path,
    original_path: str,
) -> bytes:
    """Read one image file into an immutable byte snapshot.

    The same snapshot is used for file hashing and pixel decoding.
    """
    try:
        return image_path.read_bytes()
    except FileNotFoundError as exc:
        raise ImageValidationError(
            f"Image file is missing: {original_path}",
            reason="missing_file",
            image_path=original_path,
        ) from exc
    except OSError as exc:
        raise ImageValidationError(
            f"Cannot read image: {original_path}",
            reason="file_access_error",
            image_path=original_path,
        ) from exc

def _calculate_pixels_hash(
    content: bytes,
    original_path: str,
) -> str:
    """Hash decoded pixels together with their mode and dimensions.

    No resizing or color conversion is performed. Palette images are rejected
    because their raw pixel indices do not fully describe displayed colors.

    The current validated dataset is expected to contain RGB images.
    """
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()

            if image.mode != "RGB":
                raise ImageValidationError(
                    f"Expected an RGB image: {original_path}",
                    reason="unexpected_color_mode",
                    image_path=original_path,
                )

            descriptor = json.dumps(
                {
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

            digest = hashlib.sha256()
            digest.update(descriptor)
            digest.update(b"\x00")
            digest.update(image.tobytes())

            return digest.hexdigest()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageValidationError(
            f"Cannot decode image: {original_path}",
            reason="unreadable_image",
            image_path=original_path,
        ) from exc

def fingerprint_card(
    card: CardImage,
    dataset_root: Path,
) -> ImageFingerprint:
    """Compute file and pixel fingerprints without modifying an image.

    Args:
        card: Validated metadata containing the relative image path.
        dataset_root: Original dataset directory containing the image.

    Returns:
        Immutable fingerprints associated with the card's image_id.

    Raises:
        DatasetStructureError: The dataset root is unavailable.
        ImageValidationError: The image cannot be accessed or decoded,
            resolves outside the dataset, or is not RGB.

    Example:
        fingerprint = fingerprint_card(card, dataset_root)
        assert fingerprint.image_id == card.image_id
        assert len(fingerprint.file_sha256) == 64
        assert len(fingerprint.pixels_sha256) == 64
    """
    image_path = _resolve_image_path(card, dataset_root)
    content = _read_image_bytes(image_path, card.path)

    return ImageFingerprint(
        image_id=card.image_id,
        file_sha256=hashlib.sha256(content).hexdigest(),
        pixels_sha256=_calculate_pixels_hash(content, card.path),
    )