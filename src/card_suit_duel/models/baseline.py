"""Purely neural baseline model (Approach B) following Lab 03's CNN+MLP structure."""

import torch
from torch import Tensor, nn

from card_suit_duel.data.mapping.definitions import OUTCOME_CLASSES
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.models.encoder import (
    CardEncoder,
    _require_image_batch,
    _require_positive_int,
)


def _require_dropout(dropout: float) -> None:
    """Require a dropout probability in [0.0, 1.0)."""
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise CardSuitDuelError(
            f"dropout must be a number in [0.0, 1.0); received {dropout!r}."
        )


class BaselineDuelNet(nn.Module):
    """End-to-end Siamese neural network predicting duel outcomes from two card images.

    Reuses the exact same CardEncoder backbone as SuitNet (Approach A):
    1. Shared LeNet-style CardEncoder (Lab 03: Conv2d(3,6,5) -> MaxPool -> ReLU ->
       Conv2d(6,16,5) -> MaxPool -> ReLU -> 16*4*4 = 256 features per card).
    2. Concatenation of the two card feature vectors [x1, x2] (512 features).
    3. MLP classifier head following the Lab 03 hidden widths (512 -> 120 -> ReLU ->
       84 -> ReLU -> 3 outcome logits for nn.CrossEntropyLoss).
    """

    def __init__(
        self,
        *,
        feature_dim: int = 16 * 4 * 4,
        hidden_dim: int = 120,
        second_hidden_dim: int = 84,
        dropout: float = 0.0,
        encoder: CardEncoder | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int(hidden_dim, name="hidden_dim")
        _require_positive_int(second_hidden_dim, name="second_hidden_dim")
        _require_dropout(dropout)

        if encoder is not None and not isinstance(encoder, CardEncoder):
            raise CardSuitDuelError("encoder must be a CardEncoder instance or None.")

        self.encoder = (
            encoder
            if encoder is not None
            else CardEncoder(feature_dim=feature_dim)
        )
        self.hidden_dim = hidden_dim
        self.second_hidden_dim = second_hidden_dim
        self.dropout_rate = float(dropout)

        layers: list[nn.Module] = [
            nn.Linear(self.encoder.feature_dim * 2, hidden_dim),
            nn.ReLU(),
        ]
        if self.dropout_rate > 0.0:
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.extend([
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, len(OUTCOME_CLASSES)),
        ])

        self.classifier = nn.Sequential(*layers)

    def forward(self, first_image: Tensor, second_image: Tensor) -> Tensor:
        """Compute outcome logits [B, 3] for a batch of ordered card pairs."""
        _require_image_batch(first_image, name="first_image")
        _require_image_batch(second_image, name="second_image")

        if first_image.shape != second_image.shape:
            raise CardSuitDuelError(
                "first_image and second_image must have identical shapes; "
                f"received {tuple(first_image.shape)} and {tuple(second_image.shape)}."
            )

        x1 = self.encoder(first_image)
        x2 = self.encoder(second_image)
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)
