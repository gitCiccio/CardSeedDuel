"""Build configurable, deterministic image-to-tensor transformations."""

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from card_suit_duel.exceptions import MetadataError


def _is_valide_image_size(image_size):
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise MetadataError(
            "image_size must be None or a (height, width) tuple."
        )


def _is_a_valid_dimension(dimension):
    if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
    ):
        raise MetadataError(
            "Image dimensions must be positive integers."
        )


def _require_image_size(
    image_size: tuple[int, int] | None,
) -> None:
    """Accept native dimensions or an explicit positive (height, width)."""
    if image_size is None:
        return

    _is_valide_image_size(image_size)

    for dimension in image_size:
        _is_a_valid_dimension(dimension)


def build_image_transform(
    *,
    image_size: tuple[int, int] | None = None,
) -> transforms.Compose:
    """Build a deterministic transformation for decoded RGB images.

    Input images must satisfy the RGB contract of load_card_image.

    Native dimensions are preserved unless image_size is supplied.
    Eight-bit pixels are converted into float32 values in [0, 1].
    Output dimensions follow the channels, height, width convention.

    No random augmentation or mean/std normalization is applied.

    Args:
        image_size: Optional target (height, width). An explicit size
            resizes to those dimensions and may change the aspect ratio.

    Returns:
        A callable transformation producing a PyTorch tensor.

    Examples:
        transform = build_image_transform()
        tensor = transform(image)

        smaller_transform = build_image_transform(
            image_size=(128, 128),
        )
    """
    _require_image_size(image_size)

    steps = []

    if image_size is not None:
        steps.append(
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )

    steps.extend([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    return transforms.Compose(steps)