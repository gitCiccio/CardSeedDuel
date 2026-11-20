"""Shared neural building blocks, metrics and Approach B baseline trainer."""

from card_suit_duel.models.baseline import BaselineDuelNet
from card_suit_duel.models.baseline_trainer import (
    CachedCardPairDataset,
    CardTensorCache,
    evaluate_baseline,
    fit_baseline,
    train_one_epoch,
)
from card_suit_duel.models.encoder import CardEncoder, SuitClassifier, SuitNet
from card_suit_duel.models.metrics import (
    collect_hardware_info,
    compute_classification_metrics,
    compute_confusion_matrix,
    count_trainable_parameters,
)

__all__ = [
    "BaselineDuelNet",
    "CachedCardPairDataset",
    "CardEncoder",
    "CardTensorCache",
    "SuitClassifier",
    "SuitNet",
    "collect_hardware_info",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "count_trainable_parameters",
    "evaluate_baseline",
    "fit_baseline",
    "train_one_epoch",
]
