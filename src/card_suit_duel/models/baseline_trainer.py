"""Five-step PyTorch training, in-memory image caching and evaluation loops."""

import copy
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import Optimizer

from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.pair_dataset import CardPairDataset
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.models.baseline import BaselineDuelNet
from card_suit_duel.models.metrics import (
    compute_classification_metrics,
    count_trainable_parameters,
)


class CachedCardPairDataset(CardPairDataset):
    """CardPairDataset that caches decoded and transformed card tensors in RAM.

    Since many duel pairs reuse the same underlying cards across an epoch
    (e.g. 3,000 validation pairs come from only 195 unique images), caching
    each transformed tensor by image_id eliminates redundant disk I/O.
    """

    def __init__(
        self,
        pairs: Sequence[CardPair],
        dataset_root: Path,
        *,
        transform: Callable[[Image.Image], Tensor] | None = None,
        shared_cache: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__(pairs, dataset_root, transform=transform)
        self._cache: dict[str, Tensor] = {} if shared_cache is None else shared_cache

    @property
    def cached_image_count(self) -> int:
        """Return the number of unique card tensors currently cached in memory."""
        return len(self._cache)

    def _load_tensor(self, card: CardImage) -> Tensor:
        cached = self._cache.get(card.image_id)
        if cached is not None:
            return cached
        tensor = super()._load_tensor(card)
        self._cache[card.image_id] = tensor
        return tensor


def _iter_batches(loader: Iterable[dict[str, Any]], max_batches: int | None):
    """Yield all batches or at most max_batches when a positive limit is set."""
    if max_batches is None:
        return loader
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0:
        raise CardSuitDuelError(f"max_batches must be None or a positive integer; received {max_batches!r}.")
    return islice(loader, max_batches)


def train_one_epoch(
    model: BaselineDuelNet,
    loader: Iterable[dict[str, Any]],
    optimizer: Optimizer,
    criterion: nn.Module | None = None,
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one training epoch using the standard 5-step PyTorch backpropagation loop."""
    if not isinstance(model, nn.Module):
        raise CardSuitDuelError("model must be an nn.Module instance.")

    loss_fn = nn.CrossEntropyLoss() if criterion is None else criterion
    target_device = torch.device(device)
    model.to(target_device)
    model.train()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    batch_count = 0

    start_time = time.perf_counter()

    for batch in _iter_batches(loader, max_batches):
        first_image = batch["first_image"].to(target_device)
        second_image = batch["second_image"].to(target_device)
        targets = batch["target"].to(target_device)

        # 1. Reset gradients
        optimizer.zero_grad()
        # 2. Forward pass
        logits = model(first_image, second_image)
        # 3. Compute loss
        loss = loss_fn(logits, targets)
        # 4. Backward pass
        loss.backward()
        # 5. Update weights
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        predictions = torch.argmax(logits.detach(), dim=1)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        batch_count += 1

    elapsed = time.perf_counter() - start_time
    if batch_count == 0 or not all_targets:
        raise CardSuitDuelError("Training loader produced no batches.")

    metrics = compute_classification_metrics(all_predictions, all_targets)
    return {
        "loss": total_loss / len(all_targets),
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        **metrics,
    }


def evaluate_baseline(
    model: BaselineDuelNet,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module | None = None,
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a baseline model with gradients disabled."""
    if not isinstance(model, nn.Module):
        raise CardSuitDuelError("model must be an nn.Module instance.")

    loss_fn = nn.CrossEntropyLoss() if criterion is None else criterion
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    batch_count = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in _iter_batches(loader, max_batches):
            first_image = batch["first_image"].to(target_device)
            second_image = batch["second_image"].to(target_device)
            targets = batch["target"].to(target_device)

            logits = model(first_image, second_image)
            loss = loss_fn(logits, targets)

            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            batch_count += 1

    elapsed = time.perf_counter() - start_time
    if batch_count == 0 or not all_targets:
        raise CardSuitDuelError("Evaluation loader produced no batches.")

    metrics = compute_classification_metrics(all_predictions, all_targets)
    return {
        "loss": total_loss / len(all_targets),
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        **metrics,
    }


def fit_baseline(
    model: BaselineDuelNet,
    train_loader: Iterable[dict[str, Any]],
    valid_loader: Iterable[dict[str, Any]],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
) -> dict[str, Any]:
    """Train BaselineDuelNet across epochs and restore the best validation checkpoint."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise CardSuitDuelError(f"epochs must be a positive integer; received {epochs!r}.")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
        raise CardSuitDuelError(f"lr must be a positive number; received {lr!r}.")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0.0:
        raise CardSuitDuelError(f"weight_decay must be a non-negative number; received {weight_decay!r}.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_valid_macro_f1 = -1.0
    best_valid_loss = float("inf")
    best_state_dict: dict[str, Any] | None = None
    total_train_seconds = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            max_batches=max_train_batches,
        )
        valid_metrics = evaluate_baseline(
            model,
            valid_loader,
            criterion,
            device=device,
            max_batches=max_valid_batches,
        )

        total_train_seconds += train_metrics["elapsed_seconds"]
        valid_f1 = valid_metrics["macro_f1"]
        valid_loss = valid_metrics["loss"]

        if (valid_f1 > best_valid_macro_f1) or (
            valid_f1 == best_valid_macro_f1 and valid_loss < best_valid_loss
        ):
            best_valid_macro_f1 = valid_f1
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_valid_loss": best_valid_loss,
        "total_training_seconds": total_train_seconds,
        "trainable_parameters": count_trainable_parameters(model),
        "history": history,
        "best_state_dict": best_state_dict,
    }
