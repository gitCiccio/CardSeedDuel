"""Test nested training subsets without reading original image files."""
from collections import Counter
from dataclasses import replace
import random
import unittest

from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, GameOutcome
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.pairs_generator import generate_pairs_for_split
from card_suit_duel.data.pairs.subset import build_nested_subsets
from card_suit_duel.exceptions import MetadataError


class NestedSubsetTests(unittest.TestCase):
    """Check membership, exact counts, balance, ordering and invalid inputs."""

    @classmethod
    def setUpClass(cls):
        cards = []
        for suit in (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES):
            label = f"ace of {suit.value}"
            for i in range(8):
                path = f"train/{label}/{i}.jpg"
                cards.append(CardImage(path, path, DatasetSplit.TRAIN, label, "ace", suit))
        cls.pairs = generate_pairs_for_split(cards, split=DatasetSplit.TRAIN, pairs_per_outcome=60, seed=42)

    def test_exact_sizes_outcomes_and_nearly_equal_combination_quotas(self):
        subsets = build_nested_subsets(self.pairs, seed=123)
        self.assertEqual(set(subsets), {20, 50, 100})
        for percentage, pairs in subsets.items():
            target = 60 * percentage // 100
            self.assertIsInstance(pairs, tuple)
            self.assertEqual(len(pairs), target * 3)
            self.assertEqual(Counter(p.outcome for p in pairs), {o: target for o in GameOutcome})
            for outcome in GameOutcome:
                counts = Counter((p.first_card.suit, p.second_card.suit) for p in pairs if p.outcome is outcome)
                self.assertEqual(len(counts), 3)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_memberships_are_nested_and_unique(self):
        subsets = build_nested_subsets(self.pairs, seed=123)
        ids = {k: {p.pair_id for p in v} for k, v in subsets.items()}
        self.assertLess(ids[20], ids[50])
        self.assertLess(ids[50], ids[100])
        for percentage, pairs in subsets.items():
            self.assertEqual(len(pairs), len(ids[percentage]))

    def test_original_order_labels_and_pair_objects_are_preserved(self):
        for pairs in build_nested_subsets(self.pairs, seed=123).values():
            selected = {p.pair_id for p in pairs}
            expected = tuple(p for p in self.pairs if p.pair_id in selected)
            self.assertEqual(pairs, expected)
            self.assertTrue(all(a is b for a, b in zip(pairs, expected)))

    def test_full_subset_equals_source_exactly(self):
        self.assertEqual(build_nested_subsets(self.pairs, seed=123)[100], self.pairs)

    def test_same_seed_reproduces_selection(self):
        self.assertEqual(build_nested_subsets(self.pairs, seed=123), build_nested_subsets(self.pairs, seed=123))

    def test_reordering_source_changes_presentation_not_selected_members(self):
        first = build_nested_subsets(self.pairs, seed=123)
        second = build_nested_subsets(tuple(reversed(self.pairs)), seed=123)
        for percentage in first:
            self.assertEqual({p.pair_id for p in first[percentage]}, {p.pair_id for p in second[percentage]})
            self.assertEqual(second[percentage], tuple(reversed(first[percentage])))

    def test_different_seed_changes_reduced_selection_not_full_subset(self):
        first = build_nested_subsets(self.pairs, seed=123)
        second = build_nested_subsets(self.pairs, seed=124)
        self.assertNotEqual({p.pair_id for p in first[20]}, {p.pair_id for p in second[20]})
        self.assertEqual(first[100], second[100])

    def test_global_random_state_and_input_are_not_modified(self):
        pairs = list(self.pairs)
        before = pairs.copy()
        state = random.getstate()
        build_nested_subsets(pairs, seed=0)
        self.assertEqual(pairs, before)
        self.assertEqual(random.getstate(), state)

    def test_invalid_seeds_are_rejected(self):
        for seed in (True, False, None, "123", 1.5):
            with self.subTest(seed=seed), self.assertRaises(MetadataError):
                build_nested_subsets(self.pairs, seed=seed)

    def test_empty_and_unbalanced_sources_are_rejected(self):
        for pairs in ([], self.pairs[:-1]):
            with self.subTest(size=len(pairs)), self.assertRaises(MetadataError):
                build_nested_subsets(pairs, seed=123)

    def test_balanced_counts_not_divisible_by_ten_are_rejected(self):
        pairs = tuple(next(p for p in self.pairs if p.outcome is o) for o in GameOutcome)
        with self.assertRaisesRegex(MetadataError, "divisible by 10"):
            build_nested_subsets(pairs, seed=123)

    def test_repeated_pair_identifiers_are_rejected(self):
        # Repeating the whole collection preserves balanced outcome counts.
        with self.assertRaisesRegex(MetadataError, "repeated pair"):
            build_nested_subsets(self.pairs + self.pairs, seed=123)

    def test_validation_and_test_pairs_are_rejected(self):
        for split in (DatasetSplit.VALID, DatasetSplit.TEST):
            converted = []
            for p in self.pairs:
                def move(card):
                    path = split.value + card.path[len("train"):]
                    return replace(card, image_id=path, path=path, split=split)
                converted.append(replace(p, first_card=move(p.first_card), second_card=move(p.second_card)))
            with self.subTest(split=split), self.assertRaisesRegex(MetadataError, "only training"):
                build_nested_subsets(converted, seed=123)

    def test_balanced_outcomes_without_combination_capacity_are_rejected(self):
        selected = []
        for outcome in GameOutcome:
            members = [p for p in self.pairs if p.outcome is outcome]
            combo = (members[0].first_card.suit, members[0].second_card.suit)
            selected.extend(p for p in members if (p.first_card.suit, p.second_card.suit) == combo)
        with self.assertRaisesRegex(MetadataError, "Insufficient training pairs"):
            build_nested_subsets(selected, seed=123)


if __name__ == "__main__":
    unittest.main()
