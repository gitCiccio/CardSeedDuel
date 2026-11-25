"""Tests for the ProbLog duel rules, the DeepProbLog adapter and the Approach A trainer."""

import copy
import itertools
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

with warnings.catch_warnings():
    # ApproximateEngine needs PySwip; only ExactEngine is used in this project.
    warnings.simplefilter("ignore", UserWarning)
    from problog import get_evaluatable
    from problog.program import PrologString

from card_suit_duel.data.mapping.definitions import SUIT_CLASSES, CardSuit, DatasetSplit
from card_suit_duel.data.mapping.mappings import outcome_to_index
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.game.oracle import determine_outcome
from card_suit_duel.logic import PROGRAM_PATH, SUIT_NETWORK_NAME
from card_suit_duel.models import CardTensorCache, SuitNet
from card_suit_duel.models.deepproblog_adapter import (
    DuelQueries,
    build_deepproblog_model,
    register_split,
)
from card_suit_duel.models.deepproblog_trainer import (
    checkpoint_path,
    evaluate_deepproblog,
    evaluate_suit_predictions,
    fit_deepproblog,
    load_checkpoint,
    suit_network,
)

_SUIT_COLORS = {
    CardSuit.HEARTS: (255, 0, 0),
    CardSuit.CLUBS: (0, 255, 0),
    CardSuit.SPADES: (0, 0, 255),
}


def _symbolic_rules() -> str:
    """Return the duel program without the neural predicate declaration."""
    lines = PROGRAM_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.startswith("nn("))


def _make_cards(root: Path, split: DatasetSplit) -> dict[CardSuit, tuple[CardImage, CardImage]]:
    """Write two solid-color images per suit and return their metadata."""
    cards: dict[CardSuit, tuple[CardImage, CardImage]] = {}
    for suit in SUIT_CLASSES:
        pair: list[CardImage] = []
        for number, rank in enumerate(("ace", "two")):
            label = f"{rank} of {suit.value}"
            relative = f"{split.value}/{label}/{number}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=_SUIT_COLORS[suit]).save(path)
            pair.append(
                CardImage(
                    image_id=relative,
                    path=relative,
                    split=split,
                    original_label=label,
                    rank=rank,
                    suit=suit,
                )
            )
        cards[suit] = (pair[0], pair[1])
    return cards


def _all_suit_pairs(cards: dict[CardSuit, tuple[CardImage, CardImage]]) -> tuple[CardPair, ...]:
    """Build one duel for each of the nine ordered suit combinations."""
    return tuple(
        CardPair(
            first_card=cards[first][0],
            second_card=cards[second][1],
            outcome=determine_outcome(first, second),
        )
        for first, second in itertools.product(SUIT_CLASSES, repeat=2)
    )


class SymbolicRuleTests(unittest.TestCase):
    """The ProbLog rules must agree with the deterministic oracle."""

    def test_rules_match_oracle_on_all_nine_suit_combinations(self):
        for first, second in itertools.product(SUIT_CLASSES, repeat=2):
            with self.subTest(first=first.value, second=second.value):
                program = (
                    _symbolic_rules()
                    + f"\nsuit(c1, {first.value}).\nsuit(c2, {second.value}).\n"
                    + "query(duel(c1, c2, O)).\n"
                )
                results = get_evaluatable().create_from(PrologString(program)).evaluate()
                true_outcomes = {str(term.args[2]) for term, p in results.items() if p > 0.5}
                self.assertEqual(true_outcomes, {determine_outcome(first, second).value})


class DeepProbLogAdapterTests(unittest.TestCase):
    """Verify query construction, tensor sources and exact inference."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cards = _make_cards(self.root, DatasetSplit.TRAIN)
        self.pairs = _all_suit_pairs(self.cards)
        self.cache = CardTensorCache(self.root, transform=build_image_transform(image_size=(32, 32)))

    def tearDown(self):
        self._tmp.cleanup()

    def test_queries_reference_split_source_and_share_circuit_structure(self):
        dataset = DuelQueries(self.pairs, DatasetSplit.TRAIN)
        self.assertEqual(len(dataset), 9)
        self.assertEqual(len(dataset.cards), 6)

        query = dataset.to_query(0)
        self.assertEqual(str(query.query), "duel(tensor(train(a)),tensor(train(b)),draw)")
        substituted = str(query.substitute().query)
        self.assertTrue(substituted.startswith("duel(tensor(train(0)),tensor(train(1)),"))

        # Queries with the same outcome differ only by substitution, so they share one circuit.
        draws = [dataset.to_query(i).query for i in range(9) if self.pairs[i].outcome.value == "draw"]
        self.assertEqual(len(set(draws)), 1)

    def test_rejects_pairs_from_another_split(self):
        with self.assertRaises(CardSuitDuelError):
            DuelQueries(self.pairs, DatasetSplit.VALID)
        with self.assertRaises(CardSuitDuelError):
            DuelQueries((), DatasetSplit.TRAIN)

    def test_exact_inference_matches_closed_form_outcome_probabilities(self):
        torch.manual_seed(3)
        network = SuitNet()
        model = build_deepproblog_model(network)
        dataset = DuelQueries(self.pairs, DatasetSplit.TRAIN)
        register_split(model, dataset, self.cache)
        model.eval()
        network.eval()

        beats = {(0, 1), (1, 2), (2, 0)}
        with torch.no_grad():
            for i, pair in enumerate(self.pairs):
                with self.subTest(pair=i):
                    p1 = network(self.cache.get(pair.first_card))[0]
                    p2 = network(self.cache.get(pair.second_card))[0]
                    expected = [0.0, 0.0, 0.0]
                    for a, b in itertools.product(range(3), repeat=2):
                        outcome = 2 if a == b else (0 if (a, b) in beats else 1)
                        expected[outcome] += float(p1[a] * p2[b])

                    result = model.solve([dataset.to_query(i).variable_output()])[0].result
                    actual = [0.0, 0.0, 0.0]
                    for term, probability in result.items():
                        actual[outcome_to_index(str(term.args[-1]))] = float(probability)
                    for got, want in zip(actual, expected):
                        self.assertAlmostEqual(got, want, places=5)

    def test_fit_deepproblog_reduces_validation_loss(self):
        torch.manual_seed(7)
        model = build_deepproblog_model(SuitNet(), lr=1e-2)
        dataset = DuelQueries(self.pairs, DatasetSplit.TRAIN)
        register_split(model, dataset, self.cache)

        initial = evaluate_deepproblog(model, dataset)
        result = fit_deepproblog(model, dataset, dataset, epochs=15, batch_size=9, seed=0)
        final = evaluate_deepproblog(model, dataset)

        self.assertEqual(len(result["history"]), 15)
        self.assertLess(final["loss"], initial["loss"])
        self.assertIsNotNone(result["best_state_dict"])
        self.assertEqual(result["trainable_parameters"], sum(p.numel() for p in SuitNet().parameters()))

    def _fresh_model(self):
        torch.manual_seed(11)
        model = build_deepproblog_model(SuitNet(), lr=1e-2)
        dataset = DuelQueries(self.pairs, DatasetSplit.TRAIN)
        register_split(model, dataset, self.cache)
        return model, dataset

    def test_checkpoints_are_written_and_resume_matches_uninterrupted_run(self):
        run_dir = self.root / "run"
        checkpoint_dir = run_dir / "checkpoints"

        model, dataset = self._fresh_model()
        full = fit_deepproblog(
            model, dataset, dataset, epochs=4, batch_size=3, seed=0, checkpoint_dir=checkpoint_dir
        )
        full_weights = copy.deepcopy(suit_network(model).state_dict())

        for epoch in range(1, 5):
            self.assertTrue(checkpoint_path(checkpoint_dir, epoch).is_file())
        self.assertTrue((run_dir / "best_model.pt").is_file())
        self.assertEqual(load_checkpoint(checkpoint_path(checkpoint_dir, 4))["epoch"], 4)

        resumed_model, dataset = self._fresh_model()
        resumed = fit_deepproblog(
            resumed_model,
            dataset,
            dataset,
            epochs=4,
            batch_size=3,
            seed=0,
            resume_state=load_checkpoint(checkpoint_path(checkpoint_dir, 2)),
        )

        self.assertEqual([h["epoch"] for h in resumed["history"]], [1, 2, 3, 4])
        self.assertEqual(resumed["best_epoch"], full["best_epoch"])
        for key, value in full_weights.items():
            # The restored best checkpoint must match bit-for-bit up to float noise.
            self.assertTrue(torch.allclose(value, suit_network(resumed_model).state_dict()[key], atol=1e-6))

        with self.assertRaises(CardSuitDuelError):
            fit_deepproblog(
                resumed_model,
                dataset,
                dataset,
                epochs=4,
                resume_state=load_checkpoint(checkpoint_path(checkpoint_dir, 4)),
            )


class _ColorToSuit(nn.Module):
    """Map red/green/blue images to one-hot suits, rotated by `shift` positions."""

    def __init__(self, shift: int) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        channel = images.mean(dim=(2, 3)).argmax(dim=1)
        return nn.functional.one_hot((channel + self.shift) % 3, num_classes=3).float()


class SuitDiagnosticsTests(unittest.TestCase):
    """Suit accuracy must be reported both directly and up to the rules' cyclic symmetry."""

    def test_rotated_suit_naming_is_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = [card for pair in _make_cards(root, DatasetSplit.VALID).values() for card in pair]
            cache = CardTensorCache(root, transform=build_image_transform(image_size=(32, 32)))

            for shift in range(3):
                with self.subTest(shift=shift):
                    network = SimpleNamespace(network_module=_ColorToSuit(shift))
                    model = SimpleNamespace(networks={SUIT_NETWORK_NAME: network})
                    report = evaluate_suit_predictions(model, cards, cache)
                    self.assertEqual(report["accuracy"], 1.0 if shift == 0 else 0.0)
                    self.assertEqual(report["accuracy_up_to_rotation"], 1.0)
                    self.assertEqual(set(report["per_class"]), {"hearts", "clubs", "spades"})


if __name__ == "__main__":
    unittest.main()
