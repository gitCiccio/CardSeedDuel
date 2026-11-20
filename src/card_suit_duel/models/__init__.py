"""Neural baseline architecture, metrics and training loops for Approach B."""

from card_suit_duel.models.baseline import BaselineDuelNet
from card_suit_duel.models.baseline_trainer import (
    CachedCardPairDataset,
    evaluate_baseline,
    fit_baseline,
    train_one_epoch,
)
from card_suit_duel.models.encoder import CardEncoder
from card_suit_duel.models.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    count_trainable_parameters,
)

__all__ = [
    "BaselineDuelNet",
    "CachedCardPairDataset",
    "CardEncoder",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "count_trainable_parameters",
    "evaluate_baseline",
    "fit_baseline",
    "train_one_epoch",
]
