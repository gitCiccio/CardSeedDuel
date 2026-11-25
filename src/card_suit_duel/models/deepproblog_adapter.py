"""Bridge between prepared card duels and DeepProbLog tensor sources and queries."""

from collections.abc import Sequence
from pathlib import Path

import torch
from deepproblog.dataset import Dataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from problog.logic import Constant, Term
from torch import Tensor

from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.logic import DUEL_PREDICATE, PROGRAM_PATH, SUIT_NETWORK_NAME
from card_suit_duel.models.baseline_trainer import CardTensorCache
from card_suit_duel.models.encoder import SuitNet

_FIRST_SLOT = Term("a")
_SECOND_SLOT = Term("b")


def _require_pairs(pairs: Sequence[CardPair]) -> None:
    """Require a non-empty sequence of CardPair objects."""
    if not pairs:
        raise CardSuitDuelError("At least one card pair is required.")
    for pair in pairs:
        if not isinstance(pair, CardPair):
            raise CardSuitDuelError("Every element of pairs must be a CardPair instance.")


class CardImages:
    """DeepProbLog tensor source mapping tensor(<split>(Index)) to cached card tensors.

    Uses the same CardTensorCache as Approach B, so decoding, resizing and
    pre-warming are identical across the two approaches.
    """

    def __init__(self, cards: Sequence[CardImage], tensor_cache: CardTensorCache) -> None:
        if not isinstance(tensor_cache, CardTensorCache):
            raise CardSuitDuelError("tensor_cache must be a CardTensorCache instance.")
        self._cards = tuple(cards)
        self._tensor_cache = tensor_cache

    @property
    def cards(self) -> tuple[CardImage, ...]:
        return self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __getitem__(self, item: tuple[Constant, ...]) -> Tensor:
        return self._tensor_cache.get(self._cards[int(item[0])])

    def prewarm(self) -> float:
        """Decode and cache every card of this source in advance; return elapsed seconds."""
        return self._tensor_cache.prewarm(self._cards)


class DuelQueries(Dataset):
    """DeepProbLog dataset of duel(tensor(split(A)), tensor(split(B)), Outcome) queries.

    Image indices are passed through a substitution so every query with the
    same outcome shares one compiled arithmetic circuit in the ExactEngine cache.
    """

    def __init__(self, pairs: Sequence[CardPair], split: DatasetSplit) -> None:
        super().__init__()
        _require_pairs(pairs)
        if not isinstance(split, DatasetSplit):
            raise CardSuitDuelError("split must be a DatasetSplit member.")

        self._pairs = tuple(pairs)
        self._split = split
        cards: dict[str, CardImage] = {}
        for pair in self._pairs:
            for card in (pair.first_card, pair.second_card):
                if card.split is not split:
                    raise CardSuitDuelError(
                        f"Card {card.image_id!r} belongs to split {card.split.value!r}, expected {split.value!r}."
                    )
                cards.setdefault(card.image_id, card)
        self._cards = tuple(cards.values())
        self._index = {card.image_id: position for position, card in enumerate(self._cards)}

    @property
    def split(self) -> DatasetSplit:
        return self._split

    @property
    def pairs(self) -> tuple[CardPair, ...]:
        return self._pairs

    @property
    def cards(self) -> tuple[CardImage, ...]:
        """Unique cards in first-appearance order; positions match tensor source indices."""
        return self._cards

    def __len__(self) -> int:
        return len(self._pairs)

    def to_query(self, i: int) -> Query:
        pair = self._pairs[i]
        source = self._split.value
        return Query(
            Term(
                DUEL_PREDICATE,
                Term("tensor", Term(source, _FIRST_SLOT)),
                Term("tensor", Term(source, _SECOND_SLOT)),
                Term(pair.outcome.value),
            ),
            substitution={
                _FIRST_SLOT: Constant(self._index[pair.first_card.image_id]),
                _SECOND_SLOT: Constant(self._index[pair.second_card.image_id]),
            },
        )

    def tensor_source(self, tensor_cache: CardTensorCache) -> CardImages:
        """Return the tensor source whose indices match this dataset's queries."""
        return CardImages(self._cards, tensor_cache)


def build_deepproblog_model(
    network: SuitNet,
    *,
    lr: float = 1e-3,
    program_path: Path = PROGRAM_PATH,
) -> Model:
    """Wrap SuitNet as the suit/2 neural predicate and compile the duel program."""
    if not isinstance(network, SuitNet):
        raise CardSuitDuelError("network must be a SuitNet instance.")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
        raise CardSuitDuelError(f"lr must be a positive number; received {lr!r}.")

    net = Network(network, SUIT_NETWORK_NAME, batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=float(lr))
    model = Model(str(program_path), [net])
    model.set_engine(ExactEngine(model), cache=True)
    return model


def register_split(
    model: Model,
    dataset: DuelQueries,
    tensor_cache: CardTensorCache,
) -> float:
    """Register the dataset's tensor source on the model and pre-warm it; return seconds."""
    source = dataset.tensor_source(tensor_cache)
    elapsed = source.prewarm()
    model.add_tensor_source(dataset.split.value, source)
    return elapsed
