"""LeNet-style convolutional feature extractor faithful to Lab 03."""

import torch
from torch import Tensor, nn

from card_suit_duel.exceptions import CardSuitDuelError


def _require_positive_int(value: int, *, name: str) -> None:
    """Require a strictly positive integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CardSuitDuelError(f"{name} must be a positive integer; received {value!r}.")


def _require_image_batch(images: Tensor, *, name: str = "images") -> None:
    """Require a float32 RGB image batch with shape [B, 3, H, W]."""
    if not isinstance(images, Tensor):
        raise CardSuitDuelError(f"{name} must be a PyTorch Tensor.")

    if images.ndim != 4 or images.shape[1] != 3:
        raise CardSuitDuelError(
            f"Expected {name} with shape [B, 3, H, W]; received {tuple(images.shape)}."
        )

    if images.shape[0] <= 0 or images.shape[2] < 28 or images.shape[3] < 28:
        raise CardSuitDuelError(
            f"Expected non-empty batch with spatial dimensions >= 28; received {tuple(images.shape)}."
        )

    if images.dtype != torch.float32:
        raise CardSuitDuelError(
            f"Expected {name} with dtype float32; received {images.dtype}."
        )


class CardEncoder(nn.Module):
    """LeNet-style convolutional encoder from Lab 03 adapted for 3-channel RGB cards.

    Applies the Lab 03 normalization (pixel - 0.5) / 0.5 to center inputs in [-1, 1],
    followed by two 5x5 convolutional + 2x2 max-pooling stages (3 -> 6 -> 16 channels)
    and adaptive pooling to 4x4 so the flattened vector has size 16 * 4 * 4 = 256.
    """

    def __init__(self, *, feature_dim: int = 16 * 4 * 4) -> None:
        super().__init__()
        _require_positive_int(feature_dim, name="feature_dim")
        self.size = 16 * 4 * 4
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = (
            nn.Identity()
            if feature_dim == self.size
            else nn.Sequential(nn.Linear(self.size, feature_dim), nn.ReLU())
        )

    def forward(self, images: Tensor) -> Tensor:
        """Extract flattened features [B, feature_dim] from [B, 3, H, W] images."""
        _require_image_batch(images, name="images")
        # Normalize [0, 1] -> [-1, 1] as in Lab 03: transforms.Normalize((0.5,), (0.5,))
        x = (images - 0.5) / 0.5
        x = self.encoder(x)
        x = x.view(-1, self.size)
        return self.projection(x)
