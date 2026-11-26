"""Training and evaluation loops for Approach A (DeepProbLog) on card duels."""

import copy
import math
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from deepproblog.model import Model
from deepproblog.query import Query

from card_suit_duel.data.mapping.mappings import SUIT_DOMAIN, outcome_to_index, suit_to_index
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.logic import SUIT_NETWORK_NAME
from card_suit_duel.models.baseline_trainer import CardTensorCache, build_pair_suit_lookup
from card_suit_duel.models.deepproblog_adapter import DuelQueries
from card_suit_duel.models.metrics import (
    compute_classification_metrics,
    compute_suit_pair_accuracy,
    count_trainable_parameters,
)

_EPS = 1e-12


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CardSuitDuelError(f"{name} must be a positive integer; received {value!r}.")


def _require_optional_limit(value: int | None, *, name: str) -> None:
    if value is not None:
        _require_positive_int(value, name=name)


def _iter_query_batches(
    dataset: DuelQueries,
    *,
    batch_size: int,
    rng: random.Random | None,
    max_batches: int | None,
) -> Iterator[list[Query]]:
    """Yield query batches, shuffled with rng when given, in order otherwise."""
    indices = list(range(len(dataset)))
    if rng is not None:
        rng.shuffle(indices)
    for batch_number, start in enumerate(range(0, len(indices), batch_size)):
        if max_batches is not None and batch_number >= max_batches:
            return
        yield [dataset.to_query(i) for i in indices[start:start + batch_size]]


def suit_network(model: Model) -> torch.nn.Module:
    """Return the SuitNet module wrapped by the suit/2 neural predicate."""
    return model.networks[SUIT_NETWORK_NAME].network_module


def train_one_epoch_deepproblog(
    model: Model,
    dataset: DuelQueries,
    *,
    batch_size: int,
    rng: random.Random,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one epoch: exact inference on duel/3, cross-entropy on the true outcome, Adam step."""
    _require_positive_int(batch_size, name="batch_size")
    _require_optional_limit(max_batches, name="max_batches")

    loss_fn = model.solver.semiring.cross_entropy
    model.train()

    total_loss = 0.0
    query_count = 0
    batch_count = 0
    start_time = time.perf_counter()

    for batch in _iter_query_batches(dataset, batch_size=batch_size, rng=rng, max_batches=max_batches):
        model.optimizer.zero_grad()
        results = model.solve(batch)
        for result, query in zip(results, batch):
            # cross_entropy also runs backward() on the weighted loss.
            total_loss += float(
                loss_fn(result, query.p, weight=1.0 / len(batch), q=query.substitute().query)
            ) * len(batch)
        model.optimizer.step()
        query_count += len(batch)
        batch_count += 1

    elapsed = time.perf_counter() - start_time
    if batch_count == 0:
        raise CardSuitDuelError("Training dataset produced no batches.")

    return {
        "loss": total_loss / query_count,
        "sample_count": query_count,
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
    }


def evaluate_deepproblog(
    model: Model,
    dataset: DuelQueries,
    *,
    batch_size: int = 256,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Predict argmax_O P(duel(A, B, O)) for every pair and compute outcome metrics.

    Like evaluate_baseline, also reports ms_per_pair and post-hoc accuracy across
    the 9 ordered suit pairs (suit labels are never used for training).
    """
    _require_positive_int(batch_size, name="batch_size")
    _require_optional_limit(max_batches, name="max_batches")

    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    batch_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in _iter_query_batches(dataset, batch_size=batch_size, rng=None, max_batches=max_batches):
            results = model.solve([query.variable_output() for query in batch])
            for result, query in zip(results, batch):
                target = outcome_to_index(str(query.output_values()[0]))
                probabilities = [0.0, 0.0, 0.0]
                for term, probability in result.result.items():
                    probabilities[outcome_to_index(str(term.args[-1]))] = float(probability)
                predictions.append(max(range(3), key=probabilities.__getitem__))
                targets.append(target)
                total_loss += -math.log(max(probabilities[target], 0.0) + _EPS)
            batch_count += 1

    elapsed = time.perf_counter() - start_time
    if batch_count == 0 or not targets:
        raise CardSuitDuelError("Evaluation dataset produced no batches.")

    metrics = compute_classification_metrics(predictions, targets)
    result: dict[str, Any] = {
        "loss": total_loss / len(targets),
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        "ms_per_pair": (elapsed * 1000.0) / len(targets),
        **metrics,
    }
    # Batches iterate in dataset order, so the first len(targets) pairs were evaluated.
    evaluated_pairs = dataset.pairs[:len(targets)]
    suit_lookup = build_pair_suit_lookup(evaluated_pairs)
    if all(pair.pair_id in suit_lookup for pair in evaluated_pairs):
        result["per_suit_combination"] = compute_suit_pair_accuracy(
            predictions,
            targets,
            [suit_lookup[pair.pair_id] for pair in evaluated_pairs],
        )
    return result


def evaluate_suit_predictions(
    model: Model,
    cards: Sequence[CardImage],
    tensor_cache: CardTensorCache,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Diagnostic only: compare SuitNet's argmax with the true suit of each card.

    Suit labels are never used for training. The duel rules are invariant under
    the cyclic rotation hearts -> clubs -> spades -> hearts, so the network may
    learn a rotated naming; accuracy is therefore also reported up to rotation.
    """
    _require_positive_int(batch_size, name="batch_size")
    if not cards:
        raise CardSuitDuelError("At least one card is required for suit diagnostics.")

    network = suit_network(model)
    network.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(cards), batch_size):
            images = torch.stack([tensor_cache.get(card) for card in cards[start:start + batch_size]])
            predictions.extend(torch.argmax(network(images), dim=1).tolist())
    targets = [suit_to_index(card.suit) for card in cards]

    rotation_accuracy = []
    for shift in range(3):
        correct = sum((prediction + shift) % 3 == target for prediction, target in zip(predictions, targets))
        rotation_accuracy.append(correct / len(targets))

    return {
        **compute_classification_metrics(predictions, targets, class_names=SUIT_DOMAIN),
        "accuracy_up_to_rotation": max(rotation_accuracy),
        "best_rotation_shift": max(range(3), key=rotation_accuracy.__getitem__),
    }


def checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    """Return the path of the per-epoch checkpoint file."""
    return checkpoint_dir / f"epoch_{epoch:03d}.pt"


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write a checkpoint atomically so an interrupted save never leaves a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by fit_deepproblog."""
    if not path.is_file():
        raise CardSuitDuelError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "epoch" not in state or "model_state_dict" not in state:
        raise CardSuitDuelError(f"Not a DeepProbLog training checkpoint: {path}")
    return state


def fit_deepproblog(
    model: Model,
    train_dataset: DuelQueries,
    valid_dataset: DuelQueries,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    checkpoint_dir: Path | None = None,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the DeepProbLog model across epochs and restore the best validation checkpoint.

    With checkpoint_dir, a full checkpoint (weights, Adam state, RNG state, history
    and best-so-far) is written after every epoch, and best_model.pt is refreshed
    whenever validation improves. resume_state (from load_checkpoint) continues
    training from the epoch after the one it was saved at.
    """
    _require_positive_int(epochs, name="epochs")
    network = suit_network(model)
    optimizer = model.networks[SUIT_NETWORK_NAME].optimizer
    rng = random.Random(seed)

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_valid_macro_f1 = -1.0
    best_valid_loss = float("inf")
    best_state_dict: dict[str, Any] | None = None
    total_train_seconds = 0.0
    start_epoch = 1

    if resume_state is not None:
        if resume_state["epoch"] >= epochs:
            raise CardSuitDuelError(
                f"Checkpoint is already at epoch {resume_state['epoch']}; epochs must exceed it (got {epochs})."
            )
        network.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        rng.setstate(resume_state["rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        history = list(resume_state["history"])
        best_epoch = resume_state["best_epoch"]
        best_valid_macro_f1 = resume_state["best_valid_macro_f1"]
        best_valid_loss = resume_state["best_valid_loss"]
        best_state_dict = resume_state["best_state_dict"]
        total_train_seconds = resume_state["total_training_seconds"]
        start_epoch = resume_state["epoch"] + 1

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch_deepproblog(
            model,
            train_dataset,
            batch_size=batch_size,
            rng=rng,
            max_batches=max_train_batches,
        )
        valid_metrics = evaluate_deepproblog(
            model,
            valid_dataset,
            max_batches=max_valid_batches,
        )

        total_train_seconds += train_metrics["elapsed_seconds"]
        valid_f1 = valid_metrics["macro_f1"]
        valid_loss = valid_metrics["loss"]

        improved = (valid_f1 > best_valid_macro_f1) or (
            valid_f1 == best_valid_macro_f1 and valid_loss < best_valid_loss
        )
        if improved:
            best_valid_macro_f1 = valid_f1
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(network.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_path(checkpoint_dir, epoch),
                {
                    "epoch": epoch,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_valid_macro_f1": best_valid_macro_f1,
                    "best_valid_loss": best_valid_loss,
                    "best_state_dict": best_state_dict,
                    "total_training_seconds": total_train_seconds,
                },
            )
            if improved:
                _save_checkpoint(checkpoint_dir.parent / "best_model.pt", best_state_dict)

    if best_state_dict is not None:
        network.load_state_dict(best_state_dict)

    # Read from history so runs resumed from older checkpoints report it too.
    best_valid_metrics = next(
        (entry["valid"] for entry in history if entry["epoch"] == best_epoch),
        None,
    )

    return {
        "epochs": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_valid_loss": best_valid_loss,
        "best_valid_metrics": best_valid_metrics,
        "total_training_seconds": total_train_seconds,
        "trainable_parameters": count_trainable_parameters(network),
        "history": history,
        "best_state_dict": best_state_dict,
    }
