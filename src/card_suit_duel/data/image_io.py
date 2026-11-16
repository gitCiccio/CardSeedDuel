"""Provide safe, read-only access to original dataset images."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from card_suit_duel.data.mapping.schema import CardImage
# Preserve the public image_io import while sharing its implementation.
from card_suit_duel.data.path_checks import resolve_card_path as resolve_card_path
from card_suit_duel.exceptions import (
    ImageValidationError,
)


def _require_rgb_image(
    image: Image.Image,
    original_path: str,
) -> None:
    """Require the dataset's RGB contract without silent color conversion."""
    if image.mode != "RGB":
        raise ImageValidationError(
            f"Expected an RGB image: {original_path}",
            reason="unexpected_color_mode",
            image_path=original_path,
        )

def load_card_image(
    card: CardImage,
    dataset_root: Path,
) -> Image.Image:
    """Load an RGB image independently of its source file.

    The source file is closed before returning. The returned image owns
    its in-memory pixels and must be closed by the caller.

    No resizing, normalization, augmentation or file modification occurs.

    Args:
        card: Validated metadata referencing the original image.
        dataset_root: Directory containing the original dataset splits.

    Returns:
        A decoded RGB image detached from the closed source file.

    Raises:
        DatasetStructureError: The dataset directory is unavailable.
        ImageValidationError: The image location, contents or mode
            do not satisfy the loading contract.

    Example:
        with load_card_image(card, dataset_root) as image:
            assert image.mode == "RGB"
            width, height = image.size
    """
    image_path = resolve_card_path(card, dataset_root)

    try:
        with Image.open(image_path) as image:
            image.load()
            _require_rgb_image(image, card.path)

            return image.copy()

    except FileNotFoundError as exc:
        raise ImageValidationError(
            f"Image file is missing: {card.path}",
            reason="missing_file",
            image_path=card.path,
        ) from exc

    except PermissionError as exc:
        raise ImageValidationError(
            f"Cannot read image: {card.path}",
            reason="file_access_error",
            image_path=card.path,
        ) from exc

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageValidationError(
            f"Cannot decode image: {card.path}",
            reason="unreadable_image",
            image_path=card.path,
        ) from exc