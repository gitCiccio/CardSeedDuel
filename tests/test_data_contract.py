"""Behavioral tests for the shared metadata and class-index contract.

Run from the repository root after installing the project with ``python -m pip install -e .``::

    python -m unittest discover -s tests -p "test_*.py" -v

No dataset, Pillow, PyTorch or DeepProbLog installation is required.
"""

import unittest
from dataclasses import FrozenInstanceError, replace

from card_suit_duel.data.schema import (
    CardImage, CardSuit, DatasetSplit, GameOutcome,
)
from card_suit_duel.data.mappings import (
    SUIT_DOMAIN, OUTCOME_DOMAIN, SUIT_TO_INDEX, OUTCOME_TO_INDEX,
    suit_to_index, index_to_suit, outcome_to_index, index_to_outcome,
)
from card_suit_duel.exceptions import (
    MetadataError, MappingError,
    InvalidMappingValueError, InvalidMappingIndexError,
)


class CardImageTests(unittest.TestCase):
    """Validate metadata boundaries without reading any image files."""

    def setUp(self):
        self.card = CardImage(
            image_id="train/ace of hearts/001.jpg",
            path="train/ace of hearts/001.jpg",
            split=DatasetSplit.TRAIN,
            original_label="ace of hearts",
            rank="ace",
            suit=CardSuit.HEARTS,
        )

    def test_all_original_suits_and_splits_are_representable(self):
        for split in DatasetSplit:
            for suit in CardSuit:
                with self.subTest(split=split, suit=suit):
                    label = f"ace of {suit.value}"
                    path = f"{split.value}/{label}/001.jpg"
                    card = replace(self.card, image_id=path, path=path,
                                   split=split, original_label=label, suit=suit)
                    self.assertIs(card.suit, suit)
                    self.assertIs(card.split, split)

    def test_joker_has_no_suit(self):
        card = replace(self.card, image_id="train/joker/001.jpg",
                       path="train/joker/001.jpg", original_label="joker",
                       rank="xxx", suit=None)
        self.assertIsNone(card.suit)

    def test_card_cannot_be_reassigned_to_another_split(self):
        with self.assertRaises(FrozenInstanceError):
            self.card.split = DatasetSplit.TEST

    def test_required_text_fields_reject_empty_or_invalid_values(self):
        for field in ("image_id", "path", "original_label", "rank"):
            for value in ("", "   ", None, 123):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(MetadataError):
                        replace(self.card, **{field: value})

    def test_required_text_fields_reject_outer_whitespace(self):
        for field in ("image_id", "path", "original_label", "rank"):
            for value in (" leading", "trailing "):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(MetadataError):
                        replace(self.card, **{field: value})

    def test_split_requires_an_enum_not_a_string(self):
        for value in ("train", "training", None):
            with self.subTest(value=value):
                with self.assertRaises(MetadataError):
                    replace(self.card, split=value)

    def test_suit_requires_an_enum_not_a_string(self):
        with self.assertRaises(MetadataError):
            replace(self.card, suit="hearts")

    def test_rejects_absolute_traversing_and_noncanonical_paths(self):
        paths = (
            "/train/ace of hearts/001.jpg",
            "C:/dataset/train/ace of hearts/001.jpg",
            "train\\ace of hearts\\001.jpg",
            "train/../test/001.jpg",
            "./train/ace of hearts/001.jpg",
            "train//ace of hearts/001.jpg",
            "train/ace of hearts/001.jpg/",
            "train/001.jpg",
        )
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(MetadataError):
                    replace(self.card, image_id=path, path=path)

    def test_path_must_match_original_split(self):
        path = "test/ace of hearts/001.jpg"
        with self.assertRaises(MetadataError):
            replace(self.card, image_id=path, path=path)

    def test_identifier_must_equal_path_in_current_schema(self):
        with self.assertRaises(MetadataError):
            replace(self.card, image_id="another-id")

    def test_joker_cannot_have_a_suit(self):
        with self.assertRaises(MetadataError):
            replace(self.card, original_label="joker")

    def test_non_joker_requires_a_suit(self):
        with self.assertRaises(MetadataError):
            replace(self.card, suit=None)


    def test_label_and_suit_must_agree(self):
        for suit in (CardSuit.CLUBS, CardSuit.SPADES, CardSuit.DIAMONDS):
            with self.subTest(suit=suit):
                with self.assertRaisesRegex(MetadataError, 'match rank and suit'):
                    replace(self.card, suit=suit)

    def test_label_and_rank_must_agree(self):
        with self.assertRaisesRegex(MetadataError, 'match rank and suit'):
            replace(self.card, rank='king')

    def test_joker_rank_uses_the_original_marker(self):
        with self.assertRaisesRegex(MetadataError, 'rank marker'):
            replace(self.card, original_label='joker', suit=None, rank='ace')


class MappingTests(unittest.TestCase):
    """Protect class ordering shared by neural and symbolic components."""

    def test_suit_order_and_bidirectional_conversion(self):
        expected = (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES)
        self.assertEqual(SUIT_DOMAIN, ("hearts", "clubs", "spades"))
        for index, suit in enumerate(expected):
            with self.subTest(suit=suit):
                self.assertEqual(suit_to_index(suit), index)
                self.assertEqual(suit_to_index(suit.value), index)
                self.assertIs(index_to_suit(index), suit)

    def test_outcome_order_and_bidirectional_conversion(self):
        expected = (GameOutcome.P1_WIN, GameOutcome.P2_WIN, GameOutcome.DRAW)
        self.assertEqual(OUTCOME_DOMAIN, ("p1_win", "p2_win", "draw"))
        for index, outcome in enumerate(expected):
            with self.subTest(outcome=outcome):
                self.assertEqual(outcome_to_index(outcome), index)
                self.assertEqual(outcome_to_index(outcome.value), index)
                self.assertIs(index_to_outcome(index), outcome)

    def test_diamonds_is_original_metadata_but_not_a_model_class(self):
        for value in (CardSuit.DIAMONDS, "diamonds"):
            with self.subTest(value=value):
                with self.assertRaises(InvalidMappingValueError):
                    suit_to_index(value)

    def test_unknown_labels_are_rejected_without_silent_cleanup(self):
        cases = (
            (suit_to_index, (None, "joker", "HEARTS", " hearts ", "", 0, [])),
            (outcome_to_index, (None, "win", "P1_WIN", " draw ", "", 0, [])),
        )
        for convert, values in cases:
            for value in values:
                with self.subTest(converter=convert.__name__, value=value):
                    with self.assertRaises(InvalidMappingValueError):
                        convert(value)

    def test_indices_reject_negative_out_of_range_and_wrong_types(self):
        for convert in (index_to_suit, index_to_outcome):
            for value in (-1, 3, 100, True, False, 1.0, "1", None):
                with self.subTest(converter=convert.__name__, value=value):
                    with self.assertRaises(InvalidMappingIndexError):
                        convert(value)

    def test_shared_mappings_cannot_be_changed_accidentally(self):
        with self.assertRaises(TypeError):
            SUIT_TO_INDEX[CardSuit.HEARTS] = 2
        with self.assertRaises(TypeError):
            OUTCOME_TO_INDEX[GameOutcome.DRAW] = 0

    def test_mapping_failures_can_be_caught_as_a_group(self):
        self.assertTrue(issubclass(InvalidMappingValueError, MappingError))
        self.assertTrue(issubclass(InvalidMappingIndexError, MappingError))


if __name__ == "__main__":
    unittest.main()
