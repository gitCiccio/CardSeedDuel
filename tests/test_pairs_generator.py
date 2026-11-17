"""Verify pair indexing, balanced sampling and the single-duel contract.

Tests use synthetic metadata and do not open original dataset images.
"""

from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import product
import random
import unittest

from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, GameOutcome
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.pairs_generator import (
    _count_available_pairs, _pair_from_index, generate_pairs_for_split,
)
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import MetadataError, InvalidMappingValueError
from card_suit_duel.game.oracle import determine_outcome

SUITS = (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES)


def make_card(index, suit=CardSuit.HEARTS, split=DatasetSplit.TRAIN, rank="ace"):
    """Construct valid card metadata without creating an image file."""
    label = f"{rank} of {suit.value}"
    path = f"{split.value}/{label}/{index:03d}.jpg"
    return CardImage(path, path, split, label, rank, suit)


def make_cards(count=5):
    return [make_card(i, suit, split) for split in DatasetSplit for suit in SUITS for i in range(count)]


class PairIndexTests(unittest.TestCase):
    """Compare every decoded pair against explicit small Cartesian products."""

    def test_different_suit_decoder_is_complete_and_unique(self):
        for n, m in product(range(6), repeat=2):
            with self.subTest(n=n, m=m):
                first = tuple(make_card(i) for i in range(n))
                second = tuple(make_card(i, CardSuit.CLUBS) for i in range(m))
                capacity = _count_available_pairs(first, second, same_suit=False)
                actual = [_pair_from_index(i, first, second, same_suit=False) for i in range(capacity)]
                self.assertEqual(capacity, n * m)
                self.assertEqual(actual, list(product(first, second)))
                self.assertEqual(len(actual), len(set(actual)))

    def test_same_suit_decoder_skips_self_pairs_without_losing_reversals(self):
        for n in range(8):
            with self.subTest(n=n):
                cards = tuple(make_card(i) for i in range(n))
                capacity = _count_available_pairs(cards, cards, same_suit=True)
                actual = [_pair_from_index(i, cards, cards, same_suit=True) for i in range(capacity)]
                expected = [(a, b) for a in cards for b in cards if a.image_id != b.image_id]
                self.assertEqual(capacity, n * (n - 1))
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), len(set(actual)))


class PairSamplingTests(unittest.TestCase):
    """Test the public generator's guarantees rather than particular RNG draws."""

    def setUp(self):
        self.cards = make_cards()

    def generate(self, cards=None, *, split=DatasetSplit.TRAIN, count=10, seed=42):
        return generate_pairs_for_split(
            self.cards if cards is None else cards,
            split=split, pairs_per_outcome=count, seed=seed,
        )

    def test_balanced_outcomes_and_nearly_equal_suit_combination_quotas(self):
        for requested in (1, 2, 3, 10, 11, 12):
            with self.subTest(requested=requested):
                pairs = self.generate(count=requested)
                self.assertIsInstance(pairs, tuple)
                self.assertEqual(len(pairs), 3 * requested)
                self.assertEqual(Counter(p.outcome for p in pairs), {o: requested for o in GameOutcome})
                counts = Counter((p.first_card.suit, p.second_card.suit) for p in pairs)
                for outcome in GameOutcome:
                    quotas = [counts[(a, b)] for a in SUITS for b in SUITS if determine_outcome(a, b) is outcome]
                    self.assertEqual(sum(quotas), requested)
                    self.assertLessEqual(max(quotas) - min(quotas), 1)

    def test_pairs_are_unique_distinct_and_labeled_by_oracle(self):
        pairs = self.generate(count=30)
        self.assertEqual(len({p.pair_id for p in pairs}), len(pairs))
        for pair in pairs:
            self.assertNotEqual(pair.first_card.image_id, pair.second_card.image_id)
            self.assertIs(pair.outcome, determine_outcome(pair.first_card.suit, pair.second_card.suit))

    def test_only_requested_split_is_used_and_splits_share_no_images(self):
        used = {}
        for split in DatasetSplit:
            pairs = self.generate(split=split)
            for pair in pairs:
                self.assertIs(pair.first_card.split, split)
                self.assertIs(pair.second_card.split, split)
            used[split] = {c.image_id for p in pairs for c in (p.first_card, p.second_card)}
        for first, second in product(DatasetSplit, repeat=2):
            if first is not second:
                self.assertFalse(used[first] & used[second])

    def test_same_seed_and_reordered_input_produce_identical_results(self):
        pairs = self.generate()
        self.assertEqual(pairs, self.generate())
        self.assertEqual(pairs, self.generate(list(reversed(self.cards))))

    def test_different_seeds_can_produce_different_samples(self):
        self.assertNotEqual(self.generate(seed=42), self.generate(seed=43))

    def test_global_random_state_and_input_order_are_preserved(self):
        state = random.getstate()
        before = list(self.cards)
        self.generate()
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.cards, before)

    def test_changing_training_request_does_not_change_separate_validation_call(self):
        valid = self.generate(split=DatasetSplit.VALID, seed=123)
        self.generate(count=20)
        self.assertEqual(valid, self.generate(split=DatasetSplit.VALID, seed=123))

    def test_invalid_counts_seeds_and_splits_are_rejected(self):
        for count in (0, -1, True, False, 1.5, "3", None):
            with self.subTest(count=count), self.assertRaises(MetadataError):
                self.generate(count=count)
        for seed in (True, False, 1.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(MetadataError):
                self.generate(seed=seed)
        for split in ("train", None, 0):
            with self.subTest(split=split), self.assertRaises(MetadataError):
                self.generate(split=split)

    def test_zero_and_negative_integer_seeds_are_supported(self):
        for seed in (0, -42):
            with self.subTest(seed=seed):
                self.assertEqual(self.generate(seed=seed), self.generate(seed=seed))

    def test_duplicate_identifiers_and_disallowed_suits_are_rejected(self):
        with self.assertRaisesRegex(MetadataError, "repeated image"):
            self.generate([*self.cards, self.cards[0]])
        with self.assertRaisesRegex(MetadataError, "outside the game domain"):
            self.generate([*self.cards, make_card(0, CardSuit.DIAMONDS)])

    def test_empty_and_absent_split_fail_with_capacity_error(self):
        for cards in ([], [c for c in self.cards if c.split is DatasetSplit.TEST]):
            with self.subTest(cards=len(cards)), self.assertRaisesRegex(MetadataError, "Insufficient pair capacity"):
                self.generate(cards)

    def test_insufficient_capacity_does_not_silently_reduce_or_reallocate_quota(self):
        cards = [make_card(i, suit) for suit in SUITS for i in range(2)]
        # Each same-suit group has two ordered pairs. Quota 7 needs 3/2/2.
        with self.assertRaisesRegex(MetadataError, "requested 3, available 2"):
            self.generate(cards, count=7)

    def test_exact_same_suit_capacity_is_allowed_and_reversed_pairs_exist(self):
        cards = [make_card(i, suit) for suit in SUITS for i in range(2)]
        pairs = self.generate(cards, count=6)
        draws = [p for p in pairs if p.outcome is GameOutcome.DRAW]
        self.assertEqual(len(draws), 6)
        ordered = {(p.first_card.image_id, p.second_card.image_id) for p in draws}
        self.assertTrue(all((b, a) in ordered for a, b in ordered))


class CardPairTests(unittest.TestCase):
    """Verify public metadata invariants enforced by the pair dataclass."""

    def setUp(self):
        self.first = make_card(0)
        self.second = make_card(0, CardSuit.CLUBS)

    def test_valid_pair_has_stable_order_sensitive_identifier(self):
        pair = CardPair(self.first, self.second, GameOutcome.P1_WIN)
        self.assertIs(pair.split, DatasetSplit.TRAIN)
        self.assertRegex(pair.pair_id, r"^[0-9a-f]{64}$")
        self.assertEqual(pair.pair_id, CardPair(self.first, self.second, GameOutcome.P1_WIN).pair_id)
        self.assertNotEqual(pair.pair_id, CardPair(self.second, self.first, GameOutcome.P2_WIN).pair_id)

    def test_self_pair_and_cross_split_pair_are_rejected(self):
        with self.assertRaisesRegex(MetadataError, "distinct images"):
            CardPair(self.first, self.first, GameOutcome.DRAW)
        with self.assertRaisesRegex(MetadataError, "same split"):
            CardPair(self.first, make_card(0, CardSuit.CLUBS, DatasetSplit.TEST), GameOutcome.P1_WIN)

    def test_wrong_outcome_and_non_enum_are_rejected(self):
        for outcome in (GameOutcome.DRAW, GameOutcome.P2_WIN, "p1_win", None):
            with self.subTest(outcome=outcome), self.assertRaises(MetadataError):
                CardPair(self.first, self.second, outcome)

    def test_invalid_card_types_are_rejected_before_attribute_access(self):
        with self.assertRaises(MetadataError):
            CardPair(None, self.second, GameOutcome.P1_WIN)
        with self.assertRaises(MetadataError):
            CardPair(self.first, "card", GameOutcome.P1_WIN)

    def test_pair_is_immutable(self):
        pair = CardPair(self.first, self.second, GameOutcome.P1_WIN)
        with self.assertRaises(FrozenInstanceError):
            pair.outcome = GameOutcome.P2_WIN

    def test_distinct_images_of_same_suit_can_draw(self):
        pair = CardPair(self.first, make_card(1), GameOutcome.DRAW)
        self.assertIs(pair.outcome, GameOutcome.DRAW)

    def test_disallowed_suit_is_rejected_by_oracle(self):
        with self.assertRaises(InvalidMappingValueError):
            CardPair(self.first, make_card(0, CardSuit.DIAMONDS), GameOutcome.P1_WIN)


if __name__ == "__main__":
    unittest.main()
