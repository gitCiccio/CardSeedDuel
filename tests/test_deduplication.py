"""Test exact-duplicate grouping and deterministic retention decisions."""
import unittest
from itertools import permutations
from dataclasses import FrozenInstanceError

from card_suit_duel.data.deduplication.deduplication import deduplicate_cards
from card_suit_duel.data.deduplication.fingerprints import ImageFingerprint
from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError


class DeduplicationTests(unittest.TestCase):
    """Synthetic fingerprints isolate decision logic from image decoding."""

    def card(self, name, split=DatasetSplit.TRAIN, suit=CardSuit.HEARTS, rank="ace"):
        label = f"{rank} of {suit.value}"
        path = f"{split.value}/{label}/{name}.png"
        return CardImage(path, path, split, label, rank, suit)

    def fp(self, card, pixels="a", file="b"):
        return ImageFingerprint(card.image_id, file * 64, pixels * 64)

    def test_empty_input(self):
        result = deduplicate_cards([], [])
        self.assertEqual((result.retained, result.removed, result.review, result.groups), ((), (), (), ()))

    def test_unique_cards_preserve_input_order(self):
        cards = [self.card("z"), self.card("a")]
        result = deduplicate_cards(cards, [self.fp(cards[0]), self.fp(cards[1], "c")])
        self.assertEqual(result.retained, tuple(cards))
        self.assertFalse(result.groups)

    def test_same_split_uses_lexical_identifier(self):
        z, a = self.card("z"), self.card("a")
        result = deduplicate_cards([z, a], [self.fp(z), self.fp(a)])
        self.assertEqual(result.retained, (a,))
        self.assertEqual(result.removed, (z,))
        self.assertEqual(result.groups[0].members, (a, z))
        self.assertFalse(result.groups[0].requires_review)

    def test_test_has_priority_over_valid_and_train(self):
        cards = [self.card("a", split) for split in DatasetSplit]
        result = deduplicate_cards(cards, [self.fp(c) for c in cards])
        self.assertEqual(result.retained[0].split, DatasetSplit.TEST)
        self.assertEqual(len(result.removed), 2)

    def test_valid_has_priority_over_train(self):
        cards = [self.card("a"), self.card("z", DatasetSplit.VALID)]
        result = deduplicate_cards(cards, [self.fp(c) for c in cards])
        self.assertEqual(result.retained, (cards[1],))

    def test_equal_pixels_different_files_are_grouped(self):
        a, b = self.card("a"), self.card("b")
        result = deduplicate_cards([a, b], [self.fp(a), self.fp(b, file="c")])
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.removed, (b,))

    def test_suit_and_rank_conflicts_require_whole_group_review(self):
        a = self.card("a")
        for b in (self.card("b", suit=CardSuit.CLUBS), self.card("b", rank="two")):
            with self.subTest(card=b):
                result = deduplicate_cards([a, b], [self.fp(a), self.fp(b)])
                self.assertEqual(result.review, (a, b))
                self.assertFalse(result.retained)
                self.assertFalse(result.removed)
                self.assertTrue(result.groups[0].requires_review)

    def test_selection_and_groups_are_independent_of_input_order(self):
        cards = [self.card("a"), self.card("z", DatasetSplit.TEST), self.card("a", DatasetSplit.TEST)]
        expected = deduplicate_cards(cards, [self.fp(c) for c in cards])
        for order in permutations(cards):
            result = deduplicate_cards(order, [self.fp(c) for c in reversed(order)])
            self.assertEqual(result.groups, expected.groups)
            self.assertEqual(result.retained, expected.retained)

    def test_partition_is_disjoint_complete_and_order_preserving(self):
        cards = [self.card(str(i)) for i in range(5)]
        cards[4] = self.card("4", suit=CardSuit.CLUBS)
        fps = [self.fp(c, key) for c, key in zip(cards, "aabcc")]
        original = list(cards)
        result = deduplicate_cards(cards, fps)
        partitions = [result.retained, result.removed, result.review]
        ids = [{c.image_id for c in part} for part in partitions]
        self.assertEqual(set.union(*ids), {c.image_id for c in cards})
        self.assertEqual(sum(map(len, ids)), len(cards))
        for part, selected in zip(partitions, ids):
            self.assertEqual(part, tuple(c for c in cards if c.image_id in selected))
        self.assertEqual(cards, original)

    def test_duplicate_card_identifiers_are_rejected(self):
        card = self.card("a")
        with self.assertRaisesRegex(MetadataError, "repeated image"):
            deduplicate_cards([card, card], [self.fp(card)])

    def test_duplicate_fingerprint_identifiers_are_rejected(self):
        card = self.card("a")
        with self.assertRaisesRegex(MetadataError, "Repeated fingerprint"):
            deduplicate_cards([card], [self.fp(card), self.fp(card)])

    def test_missing_and_unexpected_fingerprints_are_rejected(self):
        a, b = self.card("a"), self.card("b")
        for fingerprints in ([], [self.fp(b)], [self.fp(a), self.fp(b)]):
            with self.subTest(fingerprints=fingerprints), self.assertRaisesRegex(MetadataError, "do not match"):
                deduplicate_cards([a], fingerprints)

    def test_results_are_immutable(self):
        a, b = self.card("a"), self.card("b")
        result = deduplicate_cards([a,b], [self.fp(a), self.fp(b)])
        with self.assertRaises(FrozenInstanceError):
            result.retained = ()
        with self.assertRaises(FrozenInstanceError):
            result.groups[0].representative = b


if __name__ == "__main__":
    unittest.main()
