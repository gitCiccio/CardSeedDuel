"""Five-step PyTorch training, shared image tensor caching and evaluation loops."""

import copy
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import Optimizer

from card_suit_duel.data.image_io import load_card_image
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.pairs.pair_dataset import CardPairDataset
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.data.path_checks import resolve_dataset_root
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.exceptions import CardSuitDuelError, MetadataError
from card_suit_duel.models.baseline import BaselineDuelNet
from card_suit_duel.models.metrics import (
    compute_classification_metrics,
    compute_suit_pair_accuracy,
    count_trainable_parameters,
)


def _sync_device(device: torch.device) -> None:
    """Synchronize CUDA stream before timing boundaries so GPU timings are exact."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def build_pair_suit_lookup(pairs: Sequence[CardPair]) -> dict[str, tuple[str, str]]:
    """Map pair_id to (first_suit, second_suit) for post-hoc evaluation diagnostics."""
    return {
        pair.pair_id: (pair.first_card.suit.value, pair.second_card.suit.value)
        for pair in pairs
        if pair.first_card.suit is not None and pair.second_card.suit is not None
    }


def _require_cached_tensor(value: Tensor) -> None:
    """Validate that a transformed tensor matches the [3, H, W] float32 contract."""
    if not isinstance(value, Tensor):
        raise MetadataError("The image transform must return a PyTorch tensor.")
    if value.ndim != 3 or value.shape[0] != 3:
        raise MetadataError("Expected an image tensor with shape [3, H, W].")
    if value.dtype != torch.float32:
        raise MetadataError("Expected an image tensor with dtype float32.")


class CardTensorCache:
    """Reusable in-memory cache of decoded and transformed card tensors.

    Shared between Approach A (DeepProbLog CardImages) and Approach B
    (CachedCardPairDataset) so both approaches perform identical decoding,
    resizing and pre-warming outside the timed training loop.
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self._dataset_root = resolve_dataset_root(dataset_root)
        self._transform = (
            build_image_transform()
            if transform is None
            else transform
        )
        if not callable(self._transform):
            raise MetadataError("The image transform must be callable.")
        self._tensors: dict[str, Tensor] = {}

    @property
    def dataset_root(self) -> Path:
        return self._dataset_root

    @property
    def transform(self) -> Callable[[Image.Image], Tensor]:
        return self._transform

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._tensors

    def get(self, card: CardImage) -> Tensor:
        """Return the cached [3, H, W] float32 tensor for card, loading it once if needed."""
        cached = self._tensors.get(card.image_id)
        if cached is not None:
            return cached
        with load_card_image(card, self._dataset_root) as image:
            tensor = self._transform(image)
        _require_cached_tensor(tensor)
        self._tensors[card.image_id] = tensor
        return tensor

    def prewarm(self, cards: Iterable[CardImage]) -> float:
        """Decode and cache all unique cards in advance; return elapsed seconds."""
        start_time = time.perf_counter()
        for card in cards:
            if card.image_id not in self._tensors:
                self.get(card)
        return time.perf_counter() - start_time


class CachedCardPairDataset(CardPairDataset):
    """CardPairDataset backed by a pre-warmable in-memory tensor cache.

    Call prewarm_cache() before training so epoch timings reflect only
    forward/backward/update computation rather than disk image decoding.
    """

    def __init__(
        self,
        pairs: Sequence[CardPair],
        dataset_root: Path,
        *,
        transform: Callable[[Image.Image], Tensor] | None = None,
        tensor_cache: CardTensorCache | None = None,
    ) -> None:
        resolved_transform = (
            tensor_cache.transform
            if tensor_cache is not None and transform is None
            else transform
        )
        super().__init__(pairs, dataset_root, transform=resolved_transform)
        self._tensor_cache = (
            CardTensorCache(dataset_root, transform=self._transform)
            if tensor_cache is None
            else tensor_cache
        )

    @property
    def cached_image_count(self) -> int:
        """Return the number of unique card tensors currently cached in memory."""
        return len(self._tensor_cache)

    def unique_cards(self) -> tuple[CardImage, ...]:
        """Return all unique CardImage objects referenced by this dataset."""
        seen: dict[str, CardImage] = {}
        for pair in self._pairs:
            seen.setdefault(pair.first_card.image_id, pair.first_card)
            seen.setdefault(pair.second_card.image_id, pair.second_card)
        return tuple(seen.values())

    def prewarm_cache(self) -> float:
        """Decode and cache every unique card referenced by this dataset."""
        return self._tensor_cache.prewarm(self.unique_cards())

    def _load_tensor(self, card: CardImage) -> Tensor:
        return self._tensor_cache.get(card)


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

    _sync_device(target_device)
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

    _sync_device(target_device)
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
    pair_suit_lookup: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate a baseline model with gradients disabled.

    When pair_suit_lookup is provided and batches contain 'pair_id', also
    computes post-hoc accuracy broken down across the 9 ordered suit pairs.
    """
    if not isinstance(model, nn.Module):
        raise CardSuitDuelError("model must be an nn.Module instance.")

    loss_fn = nn.CrossEntropyLoss() if criterion is None else criterion
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    all_suit_pairs: list[tuple[str, str]] = []
    batch_count = 0

    _sync_device(target_device)
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
            if pair_suit_lookup is not None and "pair_id" in batch:
                all_suit_pairs.extend(pair_suit_lookup[pid] for pid in batch["pair_id"])
            batch_count += 1

    _sync_device(target_device)
    elapsed = time.perf_counter() - start_time
    if batch_count == 0 or not all_targets:
        raise CardSuitDuelError("Evaluation loader produced no batches.")

    metrics = compute_classification_metrics(all_predictions, all_targets)
    result: dict[str, Any] = {
        "loss": total_loss / len(all_targets),
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        "ms_per_pair": (elapsed * 1000.0) / len(all_targets),
        **metrics,
    }
    if all_suit_pairs and len(all_suit_pairs) == len(all_targets):
        result["per_suit_combination"] = compute_suit_pair_accuracy(
            all_predictions,
            all_targets,
            all_suit_pairs,
        )
    return result


def fit_baseline(
    model: BaselineDuelNet,
    train_loader: Iterable[dict[str, Any]],
    valid_loader: Iterable[dict[str, Any]],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int | None = None,
    device: str | torch.device = "cpu",
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    valid_pair_suit_lookup: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Train BaselineDuelNet across epochs and restore the best validation checkpoint.

    Early stopping is disabled by default (patience=None). When patience is set
    to a positive integer, training stops if validation Macro-F1 (tie-broken on
    lower validation loss) does not improve for that many consecutive epochs.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise CardSuitDuelError(f"epochs must be a positive integer; received {epochs!r}.")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
        raise CardSuitDuelError(f"lr must be a positive number; received {lr!r}.")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0.0:
        raise CardSuitDuelError(f"weight_decay must be a non-negative number; received {weight_decay!r}.")
    if patience is not None and (isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0):
        raise CardSuitDuelError(f"patience must be None or a positive integer; received {patience!r}.")

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
    best_valid_metrics: dict[str, Any] | None = None
    best_state_dict: dict[str, Any] | None = None
    total_train_seconds = 0.0
    epochs_without_improvement = 0
    stopped_early = False

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
            pair_suit_lookup=valid_pair_suit_lookup,
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
            best_valid_metrics = valid_metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )

        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "epochs": epochs,
        "epochs_completed": len(history),
        "patience": patience,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_valid_loss": best_valid_loss,
        "best_valid_metrics": best_valid_metrics,
        "total_training_seconds": total_train_seconds,
        "trainable_parameters": count_trainable_parameters(model),
        "history": history,
        "best_state_dict": best_state_dict,
    }
