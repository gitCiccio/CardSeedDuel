"""Build reproducible nested subsets of the training-pair dataset."""

import random
from collections import Counter
from collections.abc import Sequence

from card_suit_duel.data.mapping.definitions import (
    CardSuit,
    DatasetSplit,
    SUIT_CLASSES,
)

from card_suit_duel.data.mapping.definitions import GameOutcome
from card_suit_duel.game.oracle import determine_outcome
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import MetadataError


def _require_training_pair(pair: CardPair) -> None:
    """Prevent validation or test duels from entering training subsets."""
    if pair.split is not DatasetSplit.TRAIN:
        raise MetadataError(
            "Training subsets can contain only training pairs."
        )


def _require_unique_pair_ids(
    pairs: Sequence[CardPair],
) -> None:
    """Reject repeated ordered duels before selecting training subsets."""
    identifiers = {pair.pair_id for pair in pairs}

    if len(identifiers) != len(pairs):
        raise MetadataError(
            "Training subset input contains repeated pair identifiers."
        )


def _require_integer_seed(seed: int) -> None:
    """Require an integer seed, explicitly excluding booleans."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetadataError(
            "The subset seed must be an integer."
        )


def _prepare_pair_groups(
    pairs: Sequence[CardPair],
    *,
    seed: int,
) -> dict[tuple[CardSuit, CardSuit], tuple[CardPair, ...]]:
    """Group and shuffle training duels by ordered suit combination.

    Each group is sorted by pair_id before shuffling, making the result
    independent of the input CSV order.

    Groups are shuffled once. Nested subsets must reuse these sequences
    rather than shuffle independently for each percentage.
    """
    _require_integer_seed(seed)
    _require_unique_pair_ids(pairs)

    groups: dict[
        tuple[CardSuit, CardSuit],
        list[CardPair],
    ] = {
        (first_suit, second_suit): []
        for first_suit in SUIT_CLASSES
        for second_suit in SUIT_CLASSES
    }

    for pair in pairs:
        _require_training_pair(pair)

        combination = (
            pair.first_card.suit,
            pair.second_card.suit,
        )

        groups[combination].append(pair)

    rng = random.Random(seed)

    prepared: dict[
        tuple[CardSuit, CardSuit],
        tuple[CardPair, ...],
    ] = {}

    for combination, members in groups.items():
        ordered = sorted(
            members,
            key=lambda pair: pair.pair_id,
        )

        rng.shuffle(ordered)
        prepared[combination] = tuple(ordered)

    return prepared

def _require_balanced_source(
    pairs: Sequence[CardPair],
) -> int:
    """Require balanced outcomes supporting exact 20% and 50% subsets.

    Returns the number of pairs per outcome in the full training dataset.
    """
    if not pairs:
        raise MetadataError(
            "Cannot build subsets from an empty training dataset."
        )

    counts = Counter(pair.outcome for pair in pairs)

    outcome_counts = [
        counts[outcome]
        for outcome in GameOutcome
    ]

    if len(set(outcome_counts)) != 1:
        raise MetadataError(
            "Training outcomes must have equal counts."
        )

    full_count = outcome_counts[0]

    if full_count % 10 != 0:
        raise MetadataError(
            "The count per outcome must be divisible by 10 "
            "to build exact 20% and 50% subsets."
        )

    return full_count

def _plan_subset_quotas(
    groups: dict[
        tuple[CardSuit, CardSuit],
        tuple[CardPair, ...],
    ],
    *,
    target_per_outcome: int,
) -> dict[tuple[CardSuit, CardSuit], int]:
    """Distribute each outcome target across its three suit combinations.

    A fixed combination order assigns any remainder consistently across
    subset sizes. Requests exceeding a group's capacity are rejected.
    """
    quotas: dict[tuple[CardSuit, CardSuit], int] = {}

    for outcome in GameOutcome:
        combinations = [
            combination
            for combination in groups
            if determine_outcome(*combination) is outcome
        ]

        base_count, remainder = divmod(
            target_per_outcome,
            len(combinations),
        )

        for position, combination in enumerate(combinations):
            requested = base_count + (
                1 if position < remainder else 0
            )

            if requested > len(groups[combination]):
                raise MetadataError(
                    "Insufficient training pairs for subset combination "
                    f"{combination[0].value}/{combination[1].value}."
                )

            quotas[combination] = requested

    return quotas

def build_nested_subsets(
    pairs: Sequence[CardPair],
    *,
    seed: int,
) -> dict[int, tuple[CardPair, ...]]:
    """Build nested 20%, 50% and 100% training-pair collections.

    Reduced subsets preserve outcome balance and distribute examples
    nearly equally across the three combinations for each outcome.

    All subsets preserve the relative order of the original input.
    Pair identifiers and labels are unchanged.

    The full subset contains every input pair in its original order.
    No images or files are read or modified.

    Args:
        pairs: Unique training duels with balanced outcomes.
        seed: Seed used to shuffle each suit-combination group once.

    Returns:
        Training-pair collections indexed by percentage.

    Raises:
        MetadataError: Source counts, split membership, identifiers
            or available combination capacities violate the policy.
    """
    full_count = _require_balanced_source(pairs)
    groups = _prepare_pair_groups(pairs, seed=seed)

    subsets: dict[int, tuple[CardPair, ...]] = {}

    for percentage in (20, 50):
        target_per_outcome = full_count * percentage // 100

        quotas = _plan_subset_quotas(
            groups,
            target_per_outcome=target_per_outcome,
        )

        selected_ids = {
            pair.pair_id
            for combination, members in groups.items()
            for pair in members[:quotas[combination]]
        }

        subsets[percentage] = tuple(
            pair
            for pair in pairs
            if pair.pair_id in selected_ids
        )

    subsets[100] = tuple(pairs)

    return subsets