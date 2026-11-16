"""Load ordered card duels on demand for neural models."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from card_suit_duel.data.image_io import load_card_image
from card_suit_duel.data.mapping.mappings import outcome_to_index
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.data.path_checks import resolve_dataset_root
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.exceptions import MetadataError


def _require_pair_collection(pairs: Sequence[CardPair]) -> None:
    """Require a non-empty collection of duels from one dataset split."""
    if not pairs:
        raise MetadataError("The pair dataset cannot be empty.")

    if any(not isinstance(pair, CardPair) for pair in pairs):
        raise MetadataError(
            "The dataset must contain CardPair instances."
        )

    if len({pair.split for pair in pairs}) != 1:
        raise MetadataError(
            "A pair dataset must contain only one split."
        )


def _require_image_tensor(value: Tensor) -> None:
    """Require a float32 image tensor with three channels."""
    if not isinstance(value, Tensor):
        raise MetadataError(
            "The image transform must return a PyTorch tensor."
        )

    if value.ndim != 3 or value.shape[0] != 3:
        raise MetadataError(
            "Expected an image tensor with shape [3, H, W]."
        )

    if value.dtype != torch.float32:
        raise MetadataError(
            "Expected an image tensor with dtype float32."
        )


class CardPairDataset(Dataset):
    """Load two image tensors and an outcome for each ordered duel.

    Only metadata is stored in memory. Images are opened when an example
    is requested and closed after transformation.

    Identifiers are returned for diagnostics, not as model input features.
    """

    def __init__(
        self,
        pairs: Sequence[CardPair],
        dataset_root: Path,
        *,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        """Configure the source duels, image root and image transform."""
        _require_pair_collection(pairs)

        self._pairs = tuple(pairs)
        self._dataset_root = resolve_dataset_root(dataset_root)
        self._transform = (
            build_image_transform()
            if transform is None
            else transform
        )

        if not callable(self._transform):
            raise MetadataError(
                "The image transform must be callable."
            )

    @property
    def split(self):
        """Return the split shared by all dataset examples."""
        return self._pairs[0].split

    def __len__(self) -> int:
        """Return the number of ordered duels."""
        return len(self._pairs)

    def _load_tensor(self, card: CardImage) -> Tensor:
        """Load, transform and close one image before returning its tensor."""
        with load_card_image(card, self._dataset_root) as image:
            tensor = self._transform(image)

        _require_image_tensor(tensor)
        return tensor

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """Return the image tensors, target and identifiers of one duel."""
        pair = self._pairs[index]

        return {
            "first_image": self._load_tensor(pair.first_card),
            "second_image": self._load_tensor(pair.second_card),
            "target": torch.tensor(
                outcome_to_index(pair.outcome),
                dtype=torch.long,
            ),
            "pair_id": pair.pair_id,
            "first_image_id": pair.first_card.image_id,
            "second_image_id": pair.second_card.image_id,
        }