"""Evaluation metrics, suit-pair diagnostics, multi-seed aggregation and hardware reporting."""

import math
import os
import platform
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from card_suit_duel.data.mapping.mappings import OUTCOME_DOMAIN, SUIT_DOMAIN
from card_suit_duel.exceptions import CardSuitDuelError


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the total number of trainable scalar parameters in a PyTorch module."""
    if not isinstance(model, nn.Module):
        raise CardSuitDuelError("model must be an nn.Module instance.")
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def collect_hardware_info(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Collect CPU, thread, PyTorch and GPU metadata for timing comparisons."""
    target_device = torch.device(device)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    if cuda_available:
        gpu_index = 0 if target_device.index is None else target_device.index
        gpu_name = torch.cuda.get_device_name(gpu_index)

    return {
        "device": str(target_device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def _to_index_list(values: Sequence[int] | Tensor, *, name: str, num_classes: int) -> list[int]:
    """Convert a 1D tensor or sequence of class indices into a validated list of ints."""
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise CardSuitDuelError(f"{name} tensor must be 1D; received shape {tuple(values.shape)}.")
        raw_list = values.detach().cpu().tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_list = list(values)
    else:
        raise CardSuitDuelError(f"{name} must be a 1D Tensor or sequence of integers.")

    result: list[int] = []
    for item in raw_list:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < num_classes:
            raise CardSuitDuelError(
                f"Invalid class index {item!r} in {name}; expected int in [0, {num_classes - 1}]."
            )
        result.append(item)
    return result


def compute_confusion_matrix(
    predictions: Sequence[int] | Tensor,
    targets: Sequence[int] | Tensor,
    *,
    num_classes: int = len(OUTCOME_DOMAIN),
) -> list[list[int]]:
    """Build a [num_classes, num_classes] confusion matrix (rows=target, cols=pred)."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise CardSuitDuelError("num_classes must be a positive integer.")

    pred_list = _to_index_list(predictions, name="predictions", num_classes=num_classes)
    target_list = _to_index_list(targets, name="targets", num_classes=num_classes)

    if len(pred_list) != len(target_list):
        raise CardSuitDuelError(
            f"predictions and targets must have equal length; received {len(pred_list)} and {len(target_list)}."
        )
    if not pred_list:
        raise CardSuitDuelError("Cannot compute metrics on empty predictions.")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target_idx, pred_idx in zip(target_list, pred_list):
        matrix[target_idx][pred_idx] += 1
    return matrix


def compute_classification_metrics(
    predictions: Sequence[int] | Tensor,
    targets: Sequence[int] | Tensor,
    *,
    class_names: Sequence[str] = OUTCOME_DOMAIN,
) -> dict[str, Any]:
    """Compute accuracy, macro-precision, macro-recall, macro-F1 and confusion matrix."""
    if not class_names:
        raise CardSuitDuelError("class_names cannot be empty.")

    num_classes = len(class_names)
    matrix = compute_confusion_matrix(predictions, targets, num_classes=num_classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_classes))
    accuracy = correct / total

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for idx, name in enumerate(class_names):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted_positive = sum(matrix[row][idx] for row in range(num_classes))

        precision = tp / predicted_positive if predicted_positive > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "sample_count": total,
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1_scores) / num_classes,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def compute_suit_pair_accuracy(
    predictions: Sequence[int] | Tensor,
    targets: Sequence[int] | Tensor,
    suit_pairs: Sequence[tuple[str, str]],
    *,
    num_classes: int = len(OUTCOME_DOMAIN),
) -> dict[str, dict[str, float | int]]:
    """Compute diagnostic outcome accuracy broken down by the 9 ordered suit pairs.

    Suit labels are used strictly for post-hoc error analysis (e.g. spotting
    clubs-vs-spades confusion), never as model inputs or training targets.
    """
    pred_list = _to_index_list(predictions, name="predictions", num_classes=num_classes)
    target_list = _to_index_list(targets, name="targets", num_classes=num_classes)
    if len(pred_list) != len(target_list) or len(pred_list) != len(suit_pairs):
        raise CardSuitDuelError(
            "predictions, targets and suit_pairs must have equal length."
        )
    if not pred_list:
        raise CardSuitDuelError("Cannot compute suit-pair diagnostics on empty predictions.")

    counts: dict[tuple[str, str], list[int]] = {
        (s1, s2): [0, 0] for s1 in SUIT_DOMAIN for s2 in SUIT_DOMAIN
    }
    for pred_idx, target_idx, pair in zip(pred_list, target_list, suit_pairs):
        if pair not in counts:
            raise CardSuitDuelError(f"Unsupported suit pair for diagnostics: {pair!r}.")
        counts[pair][1] += 1
        if pred_idx == target_idx:
            counts[pair][0] += 1

    return {
        f"{s1}_vs_{s2}": {
            "correct": correct,
            "support": support,
            "accuracy": (correct / support) if support > 0 else 0.0,
        }
        for (s1, s2), (correct, support) in counts.items()
    }


def aggregate_metric_runs(values: Sequence[float]) -> dict[str, float | int]:
    """Compute mean, population std, min and max across multi-seed runs."""
    if not values:
        raise CardSuitDuelError("Cannot aggregate an empty sequence of metric values.")
    floats = [float(v) for v in values]
    n = len(floats)
    mean = sum(floats) / n
    variance = sum((x - mean) ** 2 for x in floats) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(floats),
        "max": max(floats),
    }
