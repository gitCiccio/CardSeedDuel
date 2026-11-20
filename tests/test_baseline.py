"""Behavioral and unit tests for the neural architectures, metrics and baseline trainer."""

import unittest

import torch

from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.models import (
    BaselineDuelNet,
    CardEncoder,
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


class MetricsTests(unittest.TestCase):
    """Verify accuracy, macro-F1 and confusion matrix calculations."""

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
