"""Shared class orders and conversions for both learning approaches."""

from types import MappingProxyType

from card_suit_duel.data.schema import CardSuit, GameOutcome
from card_suit_duel.exceptions import (
    InvalidMappingIndexError,
    InvalidMappingValueError,
)


# These tuples are the single source of truth for model output ordering.
SUIT_CLASSES = (
    CardSuit.HEARTS,
    CardSuit.CLUBS,
    CardSuit.SPADES,
)

OUTCOME_CLASSES = (
    GameOutcome.P1_WIN,
    GameOutcome.P2_WIN,
    GameOutcome.DRAW,
)


# Read-only mappings derived from the shared class orders.
SUIT_TO_INDEX = MappingProxyType({
    suit: index
    for index, suit in enumerate(SUIT_CLASSES)
})

OUTCOME_TO_INDEX = MappingProxyType({
    outcome: index
    for index, outcome in enumerate(OUTCOME_CLASSES)
})


# Symbolic names in the same order as the neural network outputs.
SUIT_DOMAIN = tuple(suit.value for suit in SUIT_CLASSES)
OUTCOME_DOMAIN = tuple(outcome.value for outcome in OUTCOME_CLASSES)


def suit_to_index(suit: CardSuit | str) -> int:
    """Convert a supported suit into its model output index.

    Raises:
        InvalidMappingValueError: If the value is unknown or excluded
            from the game's three-suit domain.
    """
    try:
        normalized_suit = CardSuit(suit)
    except (ValueError, TypeError) as exc:
        raise InvalidMappingValueError(
            f"Unknown suit {suit!r}. "
            f"Expected one of {SUIT_DOMAIN}."
        ) from exc

    if normalized_suit not in SUIT_TO_INDEX:
        raise InvalidMappingValueError(
            f"Suit {normalized_suit.value!r} is excluded from the duel. "
            f"Expected one of {SUIT_DOMAIN}."
        )

    return SUIT_TO_INDEX[normalized_suit]


def index_to_suit(index: int) -> CardSuit:
    """Convert a valid model output index into its suit.

    Raises:
        InvalidMappingIndexError: If the index is not an integer
            within the configured range. Booleans are rejected.
    """
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(SUIT_CLASSES)
    ):
        raise InvalidMappingIndexError(
            f"Invalid suit index {index!r}. "
            f"Expected an integer from 0 to {len(SUIT_CLASSES) - 1}."
        )

    return SUIT_CLASSES[index]


def outcome_to_index(outcome: GameOutcome | str) -> int:
    """Convert a game outcome into its model output index.

    Raises:
        InvalidMappingValueError: If the outcome is unknown.
    """
    try:
        normalized_outcome = GameOutcome(outcome)
    except (ValueError, TypeError) as exc:
        raise InvalidMappingValueError(
            f"Unknown outcome {outcome!r}. "
            f"Expected one of {OUTCOME_DOMAIN}."
        ) from exc

    return OUTCOME_TO_INDEX[normalized_outcome]


def index_to_outcome(index: int) -> GameOutcome:
    """Convert a valid model output index into its outcome.

    Raises:
        InvalidMappingIndexError: If the index is not an integer
            within the configured range. Booleans are rejected.
    """
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(OUTCOME_CLASSES)
    ):
        raise InvalidMappingIndexError(
            f"Invalid outcome index {index!r}. "
            f"Expected an integer from 0 to {len(OUTCOME_CLASSES) - 1}."
        )

    return OUTCOME_CLASSES[index]