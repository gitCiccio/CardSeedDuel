"""Provide deterministic ground-truth outcomes for CardSeedDuel."""

from card_suit_duel.data.mapping.definitions import (
    CardSuit,
    GameOutcome,
    SUIT_CLASSES,
)
from card_suit_duel.exceptions import InvalidMappingValueError


_WINNING_PAIRS = frozenset({
    (CardSuit.HEARTS, CardSuit.CLUBS),
    (CardSuit.CLUBS, CardSuit.SPADES),
    (CardSuit.SPADES, CardSuit.HEARTS),
})


def _require_game_suit(suit: CardSuit, *, player: str) -> None:
    """Require an enum member belonging to the game's allowed suits."""
    if not isinstance(suit, CardSuit):
        raise InvalidMappingValueError(
            f"{player} suit must be a CardSuit member."
        )

    if suit not in SUIT_CLASSES:
        raise InvalidMappingValueError(
            f"{player} suit is not allowed in the game: {suit.value!r}."
        )


def determine_outcome(
    first_suit: CardSuit,
    second_suit: CardSuit,
) -> GameOutcome:
    """Return the duel outcome from the first player's perspective.

    Hearts beats clubs, clubs beats spades, and spades beats hearts.
    Equal suits produce a draw. Card ranks do not affect the outcome.

    Args:
        first_suit: Allowed suit of the first player's card.
        second_suit: Allowed suit of the second player's card.

    Returns:
        P1_WIN, P2_WIN or DRAW.

    Raises:
        InvalidMappingValueError: Either argument is not an allowed
            CardSuit member. Strings, Diamonds and None are rejected.

    Examples:
        >>> determine_outcome(CardSuit.HEARTS, CardSuit.CLUBS).value
        'p1_win'
        >>> determine_outcome(CardSuit.CLUBS, CardSuit.HEARTS).value
        'p2_win'
        >>> determine_outcome(CardSuit.SPADES, CardSuit.SPADES).value
        'draw'
    """
    _require_game_suit(first_suit, player="First player")
    _require_game_suit(second_suit, player="Second player")

    if first_suit == second_suit:
        return GameOutcome.DRAW

    if (first_suit, second_suit) in _WINNING_PAIRS:
        return GameOutcome.P1_WIN

    return GameOutcome.P2_WIN