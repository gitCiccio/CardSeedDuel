"""Exercise lazy pair loading and split-aware batches on temporary images."""
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from PIL import Image
from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, SUIT_CLASSES
from card_suit_duel.data.mapping.mappings import outcome_to_index
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import MetadataError, DatasetStructureError, ImageValidationError
from card_suit_duel.game.oracle import determine_outcome

if torch is not None:
    from card_suit_duel.data.pairs.pair_dataset import CardPairDataset
    from card_suit_duel.data.pairs.pair_loader import build_pair_loader
    from card_suit_duel.data.preprocessing import build_image_transform


@unittest.skipIf(torch is None, "Install the neural extra to test pair loading")
class PairLoadingTests(unittest.TestCase):
    """No real dataset is required; sample colors identify the source cards."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        colors = {CardSuit.HEARTS: (255, 0, 0), CardSuit.CLUBS: (0, 255, 0), CardSuit.SPADES: (0, 0, 255)}
        self.cards = {}
        for split in DatasetSplit:
            for suit in SUIT_CLASSES:
                for i in range(2):
                    label = f"ace of {suit.value}"
                    relative = f"{split.value}/{label}/{i}.png"
                    card = CardImage(relative, relative, split, label, "ace", suit)
                    self.cards[split, suit, i] = card
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.new("RGB", (7, 5), colors[suit]) as image:
                        image.save(path)
        self.pairs = self.make_pairs(DatasetSplit.TRAIN)

    def make_pairs(self, split):
        return tuple(CardPair(self.cards[split, a, 0], self.cards[split, b, 1], determine_outcome(a, b)) for a, b in product(SUIT_CLASSES, repeat=2))

    def dataset(self, split=DatasetSplit.TRAIN, **kwargs):
        return CardPairDataset(self.make_pairs(split), self.root, **kwargs)

    def batch_ids(self, loader):
        return [identifier for batch in loader for identifier in batch["pair_id"]]

    def test_dataset_constructor_does_not_read_images(self):
        with patch("card_suit_duel.data.pairs.pair_dataset.load_card_image") as load:
            dataset = self.dataset()
            self.assertEqual(len(dataset), 9)
            self.assertIs(dataset.split, DatasetSplit.TRAIN)
            load.assert_not_called()

    def test_item_contract_and_all_outcome_mappings(self):
        dataset = self.dataset()
        for i, pair in enumerate(self.pairs):
            sample = dataset[i]
            self.assertEqual(set(sample), {"first_image", "second_image", "target", "pair_id", "first_image_id", "second_image_id"})
            self.assertEqual(sample["pair_id"], pair.pair_id)
            self.assertEqual(sample["first_image_id"], pair.first_card.image_id)
            self.assertEqual(sample["second_image_id"], pair.second_card.image_id)
            self.assertEqual(sample["target"].dtype, torch.long)
            self.assertEqual(sample["target"].ndim, 0)
            self.assertEqual(sample["target"].item(), outcome_to_index(pair.outcome))
            for key in ("first_image", "second_image"):
                tensor = sample[key]
                self.assertEqual(tuple(tensor.shape), (3, 5, 7))
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.isfinite(tensor).all())

    def test_image_tensors_correspond_to_correct_card_colors(self):
        sample = self.dataset()[1]  # Hearts first, clubs second.
        torch.testing.assert_close(sample["first_image"][:, 0, 0], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(sample["second_image"][:, 0, 0], torch.tensor([0., 1., 0.]))

    def test_custom_resize_is_applied_to_both_cards(self):
        sample = self.dataset(transform=build_image_transform(image_size=(4, 6)))[0]
        self.assertEqual(tuple(sample["first_image"].shape), (3, 4, 6))
        self.assertEqual(tuple(sample["second_image"].shape), (3, 4, 6))

    def test_normalized_values_outside_zero_one_are_allowed(self):
        base = build_image_transform()
        sample = self.dataset(transform=lambda image: base(image) * 2 - 1)[0]
        self.assertEqual(sample["first_image"].min().item(), -1.)

    def test_dataset_copies_pair_sequence_and_leaves_inputs_unchanged(self):
        pairs = list(self.pairs)
        dataset = CardPairDataset(pairs, self.root)
        pairs.clear()
        self.assertEqual(len(dataset), 9)
        self.assertEqual(dataset[0]["pair_id"], self.pairs[0].pair_id)

    def test_empty_invalid_and_mixed_split_collections_are_rejected(self):
        for pairs in ([], [None], [self.pairs[0], self.make_pairs(DatasetSplit.TEST)[0]]):
            with self.subTest(pairs=pairs), self.assertRaises(MetadataError):
                CardPairDataset(pairs, self.root)

    def test_missing_dataset_root_is_rejected(self):
        with self.assertRaises(DatasetStructureError):
            CardPairDataset(self.pairs, self.root / "missing")

    def test_non_callable_transform_is_rejected(self):
        with self.assertRaises(MetadataError):
            self.dataset(transform=123)

    def test_invalid_transform_return_types_shapes_and_dtypes_are_rejected(self):
        values = ["not a tensor", torch.zeros(5, 7), torch.zeros(1, 5, 7), torch.zeros(3, 5, 7, dtype=torch.float64)]
        for value in values:
            with self.subTest(value=type(value).__name__), self.assertRaises(MetadataError):
                self.dataset(transform=lambda image, result=value: result)[0]

    def test_missing_image_is_not_silently_skipped(self):
        dataset = self.dataset()
        (self.root / self.pairs[0].first_card.path).unlink()
        with self.assertRaises(ImageValidationError) as caught:
            dataset[0]
        self.assertEqual(caught.exception.reason, "missing_file")

    def test_corrupt_image_is_reported(self):
        dataset = self.dataset()
        (self.root / self.pairs[0].first_card.path).write_bytes(b"broken")
        with self.assertRaises(ImageValidationError) as caught:
            dataset[0]
        self.assertEqual(caught.exception.reason, "unreadable_image")

    def test_source_files_are_closed_and_unchanged(self):
        paths = {self.root / c.path for p in self.pairs for c in (p.first_card, p.second_card)}
        before = {p: p.read_bytes() for p in paths}
        dataset = self.dataset()
        for i in range(len(dataset)):
            dataset[i]
        self.assertEqual(before, {p: p.read_bytes() for p in paths})
        sample = dataset[0]
        path = self.root / self.pairs[0].first_card.path
        path.unlink()  # On Windows an open source handle would prevent this.
        self.assertEqual(tuple(sample["first_image"].shape), (3, 5, 7))

    def test_out_of_range_index_raises_index_error(self):
        dataset = self.dataset()
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_batches_have_expected_dimensions_and_keep_last_partial_batch(self):
        loader = build_pair_loader(self.dataset(), batch_size=4, seed=42)
        batches = list(loader)
        self.assertEqual([b["target"].shape[0] for b in batches], [4, 4, 1])
        self.assertEqual(tuple(batches[0]["first_image"].shape), (4, 3, 5, 7))
        self.assertEqual(tuple(batches[0]["second_image"].shape), (4, 3, 5, 7))
        self.assertEqual(batches[0]["target"].dtype, torch.long)
        self.assertEqual(len(loader), 3)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.drop_last)

    def test_all_splits_keep_every_example_exactly_once_per_epoch(self):
        for split in DatasetSplit:
            with self.subTest(split=split):
                ids = self.batch_ids(build_pair_loader(self.dataset(split), batch_size=4))
                self.assertEqual(Counter(ids), Counter(p.pair_id for p in self.make_pairs(split)))

    def test_evaluation_order_is_stable_and_matches_csv_order(self):
        for split in (DatasetSplit.VALID, DatasetSplit.TEST):
            expected = [p.pair_id for p in self.make_pairs(split)]
            loader = build_pair_loader(self.dataset(split), batch_size=4, seed=42)
            self.assertEqual(self.batch_ids(loader), expected)
            self.assertEqual(self.batch_ids(loader), expected)

    def test_training_is_shuffled_and_reproducible_for_fresh_same_seed_loaders(self):
        first = self.batch_ids(build_pair_loader(self.dataset(), batch_size=4, seed=42))
        second = self.batch_ids(build_pair_loader(self.dataset(), batch_size=4, seed=42))
        self.assertEqual(first, second)
        self.assertNotEqual(first, [p.pair_id for p in self.pairs])

    def test_reusing_training_loader_advances_epoch_shuffle(self):
        loader = build_pair_loader(self.dataset(), batch_size=4, seed=42)
        first, second = self.batch_ids(loader), self.batch_ids(loader)
        self.assertNotEqual(first, second)
        self.assertEqual(set(first), set(second))

    def test_different_loader_seeds_can_change_training_order(self):
        self.assertNotEqual(
            self.batch_ids(build_pair_loader(self.dataset(), seed=42)),
            self.batch_ids(build_pair_loader(self.dataset(), seed=43)),
        )

    def test_loader_does_not_modify_global_random_states(self):
        dataset = self.dataset()
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        list(build_pair_loader(dataset, batch_size=4, seed=42))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_invalid_batch_sizes_and_seeds_are_rejected(self):
        dataset = self.dataset()
        for size in (0, -1, True, 1.5, None, "4"):
            with self.subTest(size=size), self.assertRaises(MetadataError):
                build_pair_loader(dataset, batch_size=size)
        for seed in (-1, True, 1.5, None, "42", 2**63):
            with self.subTest(seed=seed), self.assertRaises(MetadataError):
                build_pair_loader(dataset, seed=seed)

    def test_batch_larger_than_dataset_returns_one_complete_dataset_batch(self):
        batches = list(build_pair_loader(self.dataset(), batch_size=32))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]["pair_id"]), 9)


if __name__ == "__main__":
    unittest.main()
