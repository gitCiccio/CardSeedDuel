"""Behavioral and unit tests for shared encoders, SuitNet, BaselineDuelNet, cache, metrics and trainer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, GameOutcome
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.models import (
    BaselineDuelNet,
    CachedCardPairDataset,
    CardEncoder,
    CardTensorCache,
    SuitClassifier,
    SuitNet,
    collect_hardware_info,
    compute_classification_metrics,
    compute_confusion_matrix,
    count_trainable_parameters,
    evaluate_baseline,
    fit_baseline,
    train_one_epoch,
)


class ArchitectureTests(unittest.TestCase):
    """Verify CardEncoder, SuitNet and BaselineDuelNet contracts."""

    def test_encoder_produces_fixed_feature_dim_across_resolutions(self):
        encoder = CardEncoder(feature_dim=64)
        for height, width in ((64, 64), (128, 128), (224, 224)):
            with self.subTest(resolution=(height, width)):
                batch = torch.rand(4, 3, height, width, dtype=torch.float32)
                features = encoder(batch)
                self.assertEqual(tuple(features.shape), (4, 64))
                self.assertEqual(features.dtype, torch.float32)

    def test_suit_net_outputs_normalized_suit_probabilities_and_shares_encoder(self):
        encoder = CardEncoder()
        net = SuitNet(encoder=encoder)
        self.assertIs(net.encoder, encoder)
        self.assertIsInstance(net.classifier, SuitClassifier)

        batch = torch.rand(5, 3, 64, 64, dtype=torch.float32)
        probs = net(batch)
        self.assertEqual(tuple(probs.shape), (5, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5))
        self.assertTrue(((probs >= 0.0) & (probs <= 1.0)).all().item())

        single = torch.rand(3, 64, 64, dtype=torch.float32)
        single_probs = net(single)
        self.assertEqual(tuple(single_probs.shape), (1, 3))
        self.assertTrue(torch.allclose(single_probs.sum(dim=1), torch.ones(1), atol=1e-5))

    def test_baseline_duel_net_outputs_three_outcome_logits_and_shares_encoder(self):
        encoder = CardEncoder(feature_dim=48)
        model = BaselineDuelNet(encoder=encoder, hidden_dim=32, dropout=0.1)
        self.assertIs(model.encoder, encoder)
        self.assertGreater(count_trainable_parameters(model), 0)

        first = torch.rand(6, 3, 64, 64, dtype=torch.float32)
        second = torch.rand(6, 3, 64, 64, dtype=torch.float32)
        logits = model(first, second)
        self.assertEqual(tuple(logits.shape), (6, 3))
        self.assertEqual(logits.dtype, torch.float32)

    def test_rejects_mismatched_or_invalid_image_batches(self):
        model = BaselineDuelNet(feature_dim=32, hidden_dim=16)
        valid = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        other_size = torch.rand(2, 3, 32, 32, dtype=torch.float32)
        wrong_channels = torch.rand(2, 1, 64, 64, dtype=torch.float32)

        with self.assertRaises(CardSuitDuelError):
            model(valid, other_size)
        with self.assertRaises(CardSuitDuelError):
            model(wrong_channels, wrong_channels)
        with self.assertRaises(CardSuitDuelError):
            BaselineDuelNet(feature_dim=32, hidden_dim=0)
        with self.assertRaises(CardSuitDuelError):
            BaselineDuelNet(feature_dim=32, hidden_dim=16, dropout=1.0)


class MetricsAndHardwareTests(unittest.TestCase):
    """Verify accuracy, macro-F1, confusion matrix and hardware metadata."""

    def test_perfect_predictions_yield_unit_accuracy_and_macro_f1(self):
        targets = [0, 1, 2, 0, 1, 2]
        metrics = compute_classification_metrics(targets, targets)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        )

    def test_confusion_matrix_orientation_and_known_f1(self):
        targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        predictions = torch.tensor([0, 1, 1, 1, 2, 0], dtype=torch.long)
        matrix = compute_confusion_matrix(predictions, targets)
        self.assertEqual(matrix, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

        metrics = compute_classification_metrics(predictions, targets)
        self.assertAlmostEqual(metrics["accuracy"], 4.0 / 6.0)
        self.assertAlmostEqual(metrics["per_class"]["p1_win"]["f1"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["p2_win"]["f1"], 0.8)
        self.assertAlmostEqual(metrics["per_class"]["draw"]["f1"], 2.0 / 3.0)

    def test_rejects_invalid_metric_inputs(self):
        with self.assertRaises(CardSuitDuelError):
            compute_confusion_matrix([], [])
        with self.assertRaises(CardSuitDuelError):
            compute_confusion_matrix([0, 1], [0])
        with self.assertRaises(CardSuitDuelError):
            compute_confusion_matrix([3], [0])

    def test_collect_hardware_info_reports_required_keys(self):
        info = collect_hardware_info("cpu")
        for key in (
            "device",
            "platform",
            "python_version",
            "cpu_count",
            "torch_num_threads",
            "torch_version",
            "cuda_available",
            "gpu_name",
        ):
            self.assertIn(key, info)
        self.assertEqual(info["device"], "cpu")
        self.assertIsInstance(info["torch_num_threads"], int)


class CacheTests(unittest.TestCase):
    """Verify CardTensorCache and CachedCardPairDataset pre-warming behavior."""

    def test_prewarm_cache_decodes_each_unique_image_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "train" / "ace of hearts" / "1.jpg"
            p2 = root / "train" / "two of clubs" / "1.jpg"
            p1.parent.mkdir(parents=True, exist_ok=True)
            p2.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(p1)
            Image.new("RGB", (64, 64), color=(0, 0, 0)).save(p2)

            c1 = CardImage(
                image_id="train/ace of hearts/1.jpg",
                path="train/ace of hearts/1.jpg",
                split=DatasetSplit.TRAIN,
                original_label="ace of hearts",
                rank="ace",
                suit=CardSuit.HEARTS,
            )
            c2 = CardImage(
                image_id="train/two of clubs/1.jpg",
                path="train/two of clubs/1.jpg",
                split=DatasetSplit.TRAIN,
                original_label="two of clubs",
                rank="two",
                suit=CardSuit.CLUBS,
            )
            pairs = (
                CardPair(first_card=c1, second_card=c2, outcome=GameOutcome.P1_WIN),
                CardPair(first_card=c2, second_card=c1, outcome=GameOutcome.P2_WIN),
            )

            cache = CardTensorCache(root, transform=build_image_transform(image_size=(32, 32)))
            dataset = CachedCardPairDataset(pairs, root, tensor_cache=cache)

            from card_suit_duel.models import baseline_trainer

            with patch.object(
                baseline_trainer,
                "load_card_image",
                wraps=baseline_trainer.load_card_image,
            ) as mock_load:
                elapsed = dataset.prewarm_cache()
                self.assertGreaterEqual(elapsed, 0.0)
                self.assertEqual(dataset.cached_image_count, 2)
                self.assertEqual(mock_load.call_count, 2)

                # Subsequent reads through __getitem__ must hit RAM cache with zero disk loads.
                item0 = dataset[0]
                item1 = dataset[1]
                self.assertEqual(mock_load.call_count, 2)
                self.assertEqual(tuple(item0["first_image"].shape), (3, 32, 32))
                self.assertEqual(tuple(item1["second_image"].shape), (3, 32, 32))


class BaselineTrainerTests(unittest.TestCase):
    """Verify the 5-step training loop and validation checkpoint restoration."""

    def setUp(self):
        torch.manual_seed(7)
        first = torch.zeros(6, 3, 28, 28, dtype=torch.float32)
        second = torch.zeros(6, 3, 28, 28, dtype=torch.float32)
        first[0:2, 0] = 1.0
        second[2:4, 1] = 1.0
        first[4:6, 2] = 1.0
        second[4:6, 2] = 1.0
        targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        self.batches = [
            {
                "first_image": first,
                "second_image": second,
                "target": targets,
            }
        ]

    def test_fit_baseline_reduces_loss_on_synthetic_batch(self):
        torch.manual_seed(7)
        model = BaselineDuelNet(feature_dim=32, hidden_dim=16)
        initial_eval = evaluate_baseline(model, self.batches)

        result = fit_baseline(
            model,
            self.batches,
            self.batches,
            epochs=8,
            lr=1e-2,
        )
        final_eval = evaluate_baseline(model, self.batches)

        self.assertEqual(len(result["history"]), 8)
        self.assertLess(final_eval["loss"], initial_eval["loss"])
        self.assertGreaterEqual(final_eval["macro_f1"], initial_eval["macro_f1"])
        self.assertIsNotNone(result["best_state_dict"])

    def test_trainer_rejects_empty_loader(self):
        model = BaselineDuelNet(feature_dim=16, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with self.assertRaises(CardSuitDuelError):
            train_one_epoch(model, [], optimizer)
        with self.assertRaises(CardSuitDuelError):
            evaluate_baseline(model, [])


if __name__ == "__main__":
    unittest.main()
