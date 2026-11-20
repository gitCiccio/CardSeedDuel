"""LeNet-style convolutional encoder and SuitNet faithful to Lab 03."""

import torch
from torch import Tensor, nn

from card_suit_duel.data.mapping.definitions import SUIT_CLASSES
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
    Both Approach A (SuitNet) and Approach B (BaselineDuelNet) use this exact
    encoder class so visual capacity and normalization are identical.
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


class SuitClassifier(nn.Module):
    """Fully-connected MLP head (256 -> 120 -> 84 -> 3) with Softmax(dim=1), as in Lab 03."""

    def __init__(self, n: int = len(SUIT_CLASSES), size: int = 16 * 4 * 4) -> None:
        super().__init__()
        _require_positive_int(n, name="n")
        _require_positive_int(size, name="size")

        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Map flattened features [B, size] to suit probabilities [B, n]."""
        x = self.classifier(x)
        return self.softmax(x)


class SuitNet(nn.Module):
    """Single-card suit network for Approach A (DeepProbLog), faithful to MNIST_Net (Lab 03).

    Combines the shared CardEncoder (256 features with [-1, 1] normalization) and
    SuitClassifier (256 -> 120 -> 84 -> 3 + Softmax(dim=1)) over SUIT_CLASSES
    (hearts, clubs, spades).
    """

    def __init__(
        self,
        *,
        feature_dim: int = 16 * 4 * 4,
        encoder: CardEncoder | None = None,
    ) -> None:
        super().__init__()
        if encoder is not None and not isinstance(encoder, CardEncoder):
            raise CardSuitDuelError("encoder must be a CardEncoder instance or None.")

        self.encoder = (
            encoder
            if encoder is not None
            else CardEncoder(feature_dim=feature_dim)
        )
        self.size = self.encoder.feature_dim
        self.classifier = SuitClassifier(n=len(SUIT_CLASSES), size=self.size)

    def forward(self, x: Tensor) -> Tensor:
        """Return suit probabilities [B, 3] ordered as (hearts, clubs, spades).

        Accepts either batched [B, 3, H, W] tensors (batching=True in DeepProbLog)
        or a single unbatched [3, H, W] image tensor.
        """
        if isinstance(x, Tensor) and x.ndim == 3:
            x = x.unsqueeze(0)
        features = self.encoder(x)
        return self.classifier(features)
