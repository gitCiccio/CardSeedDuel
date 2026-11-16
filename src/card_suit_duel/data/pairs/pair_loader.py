"""Build batch loaders with split-aware ordering."""

import torch
from torch.utils.data import DataLoader

from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pair_dataset import CardPairDataset
from card_suit_duel.exceptions import MetadataError


def _require_batch_size(batch_size: int) -> None:
    """Require a positive integer batch size."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise MetadataError(
            "batch_size must be a positive integer."
        )


def _require_loader_seed(seed: int) -> None:
    """Require a non-negative seed supported by the local generator."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
    ):
        raise MetadataError(
            "The loader seed must be an integer in [0, 2**63)."
        )


def build_pair_loader(
    dataset: CardPairDataset,
    *,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """Build a loader that shuffles training and preserves evaluation order.

    The final incomplete batch is retained for every split.
    Single-process loading simplifies initial Windows/PyCharm debugging.

    The seed controls sampling order. It does not make arbitrary random
    image transformations deterministic.
    """
    _require_batch_size(batch_size)
    _require_loader_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.split is DatasetSplit.TRAIN,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )