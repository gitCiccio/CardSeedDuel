"""Define the metadata contract for an ordered card duel."""

import hashlib
import json
from dataclasses import dataclass

from card_suit_duel.data.mapping.definitions import (
    DatasetSplit,
    GameOutcome,
)
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.game.oracle import determine_outcome


def _require_card_instance(card: CardImage, *, position: str) -> None:
    """Require card metadata before accessing its identifier or split."""
    if not isinstance(card, CardImage):
        raise MetadataError(
            f"The {position} card must be a CardImage instance."
        )


def _require_distinct_images(
    first_card: CardImage,
    second_card: CardImage,
) -> None:
    """Reject a duel that references the same image twice."""
    if first_card.image_id == second_card.image_id:
        raise MetadataError(
            "A duel must contain two distinct images."
        )


def _require_same_split(
    first_card: CardImage,
    second_card: CardImage,
) -> None:
    """Prevent a duel from mixing training, validation or test images."""
    if first_card.split != second_card.split:
        raise MetadataError(
            "Both cards in a duel must belong to the same split."
        )


def _require_outcome_instance(outcome: GameOutcome) -> None:
    """Require an outcome enum instead of a string or another value."""
    if not isinstance(outcome, GameOutcome):
        raise MetadataError(
            "The duel outcome must be a GameOutcome member."
        )


def _require_matching_outcome(
    outcome: GameOutcome,
    expected: GameOutcome,
) -> None:
    """Reject a supplied label that disagrees with the oracle's result."""
    if outcome is not expected:
        raise MetadataError(
            f"Incorrect duel outcome: expected {expected.value!r}, "
            f"received {outcome.value!r}."
        )


def _validate_pair_cards(
    first_card: CardImage,
    second_card: CardImage,
) -> None:
    """Coordinate type, distinct-image and shared-split checks in order."""
    _require_card_instance(first_card, position="first")
    _require_card_instance(second_card, position="second")
    _require_distinct_images(first_card, second_card)
    _require_same_split(first_card, second_card)


def _validate_pair_outcome(
    first_card: CardImage,
    second_card: CardImage,
    outcome: GameOutcome,
) -> None:
    """Coordinate outcome type validation and comparison with the oracle."""
    _require_outcome_instance(outcome)
    expected = determine_outcome(first_card.suit, second_card.suit)
    _require_matching_outcome(outcome, expected)


def _build_pair_id(
    first_card: CardImage,
    second_card: CardImage,
) -> str:
    """Build a stable identifier from the ordered image identifiers.

    JSON encoding avoids ambiguities caused by separators inside paths.
    Reversing the images produces a different identifier.
    """
    payload = json.dumps(
        [first_card.image_id, second_card.image_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CardPair:
    """Represent an ordered, labeled duel between two distinct images.

    Both cards must belong to the same original dataset split.
    The outcome is expressed from the first player's perspective and
    must agree with the oracle.

    This object stores metadata only. It does not open image files.

    Example:
        outcome = determine_outcome(first_card.suit, second_card.suit)
        pair = CardPair(first_card, second_card, outcome)
        assert pair.split == first_card.split
    """

    first_card: CardImage
    second_card: CardImage
    outcome: GameOutcome

    def __post_init__(self) -> None:
        """Validate card compatibility and the supplied ground-truth label."""
        _validate_pair_cards(
            self.first_card,
            self.second_card,
        )
        _validate_pair_outcome(
            self.first_card,
            self.second_card,
            self.outcome,
        )

    @property
    def split(self) -> DatasetSplit:
        """Return the shared split without storing redundant metadata."""
        return self.first_card.split

    @property
    def pair_id(self) -> str:
        """Return the stable identifier of this ordered image pair."""
        return _build_pair_id(
            self.first_card,
            self.second_card,
        )