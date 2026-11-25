"""Shared neural building blocks, metrics and Approach B baseline trainer."""

from card_suit_duel.models.baseline import BaselineDuelNet
from card_suit_duel.models.baseline_trainer import (
    CachedCardPairDataset,
    CardTensorCache,
    build_pair_suit_lookup,
    evaluate_baseline,
    fit_baseline,
    train_one_epoch,
)
from card_suit_duel.models.encoder import CardEncoder, SuitClassifier, SuitNet
from card_suit_duel.models.metrics import (
    aggregate_metric_runs,
    collect_hardware_info,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_suit_pair_accuracy,
    count_trainable_parameters,
)

__all__ = [
    "BaselineDuelNet",
    "CachedCardPairDataset",
    "CardEncoder",
    "CardTensorCache",
    "SuitClassifier",
    "SuitNet",
    "aggregate_metric_runs",
    "build_pair_suit_lookup",
    "collect_hardware_info",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_suit_pair_accuracy",
    "count_trainable_parameters",
    "evaluate_baseline",
    "fit_baseline",
    "train_one_epoch",
]
