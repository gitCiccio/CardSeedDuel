"""Verify the game's ground-truth oracle independently of its implementation.

Run with: python -m unittest discover -s tests -p "test_*.py"
No dataset images or trained models are required.
"""

import doctest
import unittest
from enum import Enum
from itertools import product

from card_suit_duel.data.mapping.definitions import CardSuit, GameOutcome
from card_suit_duel.exceptions import InvalidMappingValueError
from card_suit_duel.game import oracle
from card_suit_duel.game.oracle import determine_outcome


class OracleTests(unittest.TestCase):
    """Cover all rule outcomes, player reversal and strict input boundaries."""

    def test_all_nine_combinations_match_the_game_rules(self):
        expected = (
            (CardSuit.HEARTS, CardSuit.HEARTS, GameOutcome.DRAW),
            (CardSuit.HEARTS, CardSuit.CLUBS, GameOutcome.P1_WIN),
            (CardSuit.HEARTS, CardSuit.SPADES, GameOutcome.P2_WIN),
            (CardSuit.CLUBS, CardSuit.HEARTS, GameOutcome.P2_WIN),
            (CardSuit.CLUBS, CardSuit.CLUBS, GameOutcome.DRAW),
            (CardSuit.CLUBS, CardSuit.SPADES, GameOutcome.P1_WIN),
            (CardSuit.SPADES, CardSuit.HEARTS, GameOutcome.P1_WIN),
            (CardSuit.SPADES, CardSuit.CLUBS, GameOutcome.P2_WIN),
            (CardSuit.SPADES, CardSuit.SPADES, GameOutcome.DRAW),
        )
        for first, second, outcome in expected:
            with self.subTest(first=first, second=second):
                self.assertIs(determine_outcome(first, second), outcome)

    def test_reversing_players_reverses_winner_and_preserves_draw(self):
        inverse = {
            GameOutcome.P1_WIN: GameOutcome.P2_WIN,
            GameOutcome.P2_WIN: GameOutcome.P1_WIN,
            GameOutcome.DRAW: GameOutcome.DRAW,
        }
        suits = (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES)
        for first, second in product(suits, repeat=2):
            with self.subTest(first=first, second=second):
                forward = determine_outcome(first, second)
                self.assertIs(determine_outcome(second, first), inverse[forward])

    def test_draw_occurs_exactly_when_allowed_suits_match(self):
        suits = (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES)
        for first, second in product(suits, repeat=2):
            with self.subTest(first=first, second=second):
                self.assertEqual(determine_outcome(first, second) is GameOutcome.DRAW, first is second)

    def test_invalid_values_are_rejected_for_either_player(self):
        class OtherSuit(str, Enum):
            HEARTS = "hearts"

        values = (
            CardSuit.DIAMONDS, None, "hearts", "clubs", "spades",
            "diamonds", "", " HEARTS ", 0, 1, True, False,
            [], {}, object(), OtherSuit.HEARTS,
        )
        for value in values:
            for position in (0, 1):
                with self.subTest(value=value, position=position):
                    arguments = [CardSuit.HEARTS, CardSuit.CLUBS]
                    arguments[position] = value
                    with self.assertRaises(InvalidMappingValueError):
                        determine_outcome(*arguments)

    def test_invalid_equal_values_cannot_bypass_validation_as_draws(self):
        for value in (CardSuit.DIAMONDS, None, "hearts"):
            with self.subTest(value=value):
                with self.assertRaises(InvalidMappingValueError):
                    determine_outcome(value, value)

    def test_error_identifies_the_player_with_invalid_input(self):
        with self.assertRaisesRegex(InvalidMappingValueError, "First player"):
            determine_outcome(None, CardSuit.HEARTS)
        with self.assertRaisesRegex(InvalidMappingValueError, "Second player"):
            determine_outcome(CardSuit.HEARTS, CardSuit.DIAMONDS)


def load_tests(loader, tests, pattern):
    """Include executable oracle documentation in normal test discovery."""
    tests.addTests(doctest.DocTestSuite(oracle))
    return tests


if __name__ == "__main__":
    unittest.main()
