"""Generate reproducible, balanced card duels within each dataset split."""
import random
from collections.abc import Sequence

from card_suit_duel.data.mapping.definitions import (
    CardSuit,
    DatasetSplit,
    GameOutcome,
    SUIT_CLASSES,
)
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.data.metadata_checks import require_unique_identifiers
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.game.oracle import determine_outcome

def _require_unique_images(cards: Sequence[CardImage]) -> None:
    """Apply the shared identifier check to pair-generation card metadata."""
    require_unique_identifiers(
        (card.image_id for card in cards),
        source_name="Pair generation input",
    )


def _require_allowed_suit(card: CardImage) -> None:
    """Reject cards outside the game's supported suits."""
    if card.suit not in SUIT_CLASSES:
        raise MetadataError(
            f"Card {card.image_id!r} has a suit outside the game domain."
        )

def _group_cards(
        cards: Sequence[CardImage]
) -> dict[DatasetSplit, dict[CardSuit, tuple[CardImage, ...]]]:
    """Group cards by split and suit using a stable image ordering.

        Empty groups are included explicitly. Sorting by image_id makes later
        sampling independent of the input CSV row order.
    """

    _require_unique_images(cards)

    buckets: dict[
        DatasetSplit,
        dict[CardSuit, list[CardImage]],
    ] = {
        split: {
            suit: []
            for suit in SUIT_CLASSES
        }
        for split in DatasetSplit
    }

    for card in cards:
        _require_allowed_suit(card)
        buckets[card.split][card.suit].append(card)

    return {
        split: {
            suit: tuple(
                sorted(members, key=lambda card: card.image_id)
            )
            for suit, members in suits.items()
        }
        for split, suits in buckets.items()
    }

def _count_available_pairs(
    first_cards: Sequence[CardImage],
    second_cards: Sequence[CardImage],
    *,
    same_suit: bool,
) -> int:
    """Count ordered pairs, excluding self-pairs for the same suit.

    For same-suit combinations, both arguments must represent the same
    ordered group produced by _group_cards.
    """
    if same_suit:
        # Each of N first images has N-1 possible partners: its own
        # position is excluded. Order matters, so (A, B) and (B, A) both exist.
        count = len(first_cards)
        return count * (count - 1)
    # Different suits are disjoint groups. Each first image can be paired
    # with every second image: N rows times M columns (Cartesian product).
    return len(first_cards) * len(second_cards)

def _pair_from_index(
    index: int,
    first_cards: Sequence[CardImage],
    second_cards: Sequence[CardImage],
    *,
    same_suit: bool,
) -> tuple[CardImage, CardImage]:
    """Decode an index into an ordered pair without listing all pairs.

    The index must come from the range defined by _count_available_pairs.
    Empty capacities must be rejected or skipped before calling this helper.
    This numeric index is local to one split and ordered suit combination;
    it is not the global CardPair.pair_id.
    For same-suit combinations, both sequences represent the same group.
    """
    if same_suit:
        group_size = len(first_cards)

        # Each first-image row contains N-1 legal partners. The quotient
        # selects the row; the remainder selects a partner before skipping self.
        first_position, second_position = divmod(
            index,
            group_size - 1,
        )

        # Skip the first image's position to prevent self-pairing.
        if second_position >= first_position:
            second_position += 1

        return (
            first_cards[first_position],
            second_cards[second_position],
        )

    # Flatten the N-by-M grid row by row: index = row * M + column.
    # divmod returns that row (quotient) and column (remainder).
    first_position, second_position = divmod(
        index,
        len(second_cards),
    )

    return (
        first_cards[first_position],
        second_cards[second_position],
    )


def _validate_split(split: DatasetSplit) -> None:
    """Require the split enum without shadowing its type as a parameter."""
    if not isinstance(split, DatasetSplit):
        raise MetadataError(
            "The requested split must be a DatasetSplit member."
        )


def _validate_pairs_per_outcome(pairs_per_outcome: int) -> None:
    """Require a positive integer count, explicitly rejecting booleans."""
    if (
            isinstance(pairs_per_outcome, bool)
            or not isinstance(pairs_per_outcome, int)
            or pairs_per_outcome <= 0
    ):
        raise MetadataError(
            "pairs_per_outcome must be a positive integer."
        )


def _validate_seed(seed: int) -> None:
    """Require an integer seed without accepting bool as an integer."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetadataError(
            "The sampling seed must be an integer."
        )


def _require_sampling_arguments(
    split: DatasetSplit,
    pairs_per_outcome: int,
    seed: int,
) -> None:
    """Validate the requested split, positive sample count and integer seed."""
    _validate_split(split)

    _validate_pairs_per_outcome(pairs_per_outcome)


    _validate_seed(seed)

def _require_pair_capacity(
    requested: int,
    capacity: int,
    value_first_suit: str,
    value_second_suit: str,
) -> None:
    """Reject a quota exceeding one combination's distinct ordered pairs.

    Quotas are not silently transferred to another combination: balanced
    suit-combination coverage is part of the current sampling policy.
    """
    if requested > capacity:
        raise MetadataError(
            "Insufficient pair capacity for "
            f"{value_first_suit}/{value_second_suit}: "
            f"requested {requested}, available {capacity}."
        )


def _plan_combination_counts(
    suits: dict[CardSuit, tuple[CardImage, ...]],
    pairs_per_outcome: int,
    rng: random.Random,
) -> list[tuple[CardSuit, CardSuit, int]]:
    """Allocate each outcome evenly across its three suit combinations.

    Remainders are assigned using the local random generator.
    Insufficient capacity raises an error before any pairs are sampled.
    """
    combinations: dict[
        GameOutcome,
        list[tuple[CardSuit, CardSuit]],
    ] = {
        outcome: []
        for outcome in GameOutcome
    }

    for first_suit in SUIT_CLASSES:
        for second_suit in SUIT_CLASSES:
            outcome = determine_outcome(first_suit, second_suit)
            combinations[outcome].append(
                (first_suit, second_suit)
            )

    plan: list[tuple[CardSuit, CardSuit, int]] = []

    for outcome in GameOutcome:
        outcome_combinations = combinations[outcome].copy()
        # When the requested count is not divisible by three, some suit
        # combinations receive one extra pair. Shuffle which ones receive it
        # to avoid always favoring the first combination in enum order.
        # The seeded local RNG keeps this choice reproducible.
        rng.shuffle(outcome_combinations)

        # Divide the outcome quota among its three suit combinations.
        # Example: divmod(10_000, 3) gives (3333, 1): a base quota
        # of 3333 for each combination, plus one pair to distribute.
        base_count, remainder = divmod(
            pairs_per_outcome,
            len(outcome_combinations),
        )

        for position, (first_suit, second_suit) in enumerate(
            outcome_combinations
        ):
            # Each iteration computes a fresh quota for this combination.
            # The first `remainder` combinations receive base_count + 1;
            # the others receive base_count. For 10_000: 3334, 3333, 3333.
            # Their sum remains exactly pairs_per_outcome.
            requested = base_count + (
                1 if position < remainder else 0
            )

            capacity = _count_available_pairs(
                suits[first_suit],
                suits[second_suit],
                same_suit=first_suit == second_suit,
            )

            _require_pair_capacity(requested, capacity, first_suit.value, second_suit.value)

            plan.append(
                (first_suit, second_suit, requested)
            )

    return plan

def generate_pairs_for_split(
    cards: Sequence[CardImage],
    *,
    split: DatasetSplit,
    pairs_per_outcome: int,
    seed: int,
) -> tuple[CardPair, ...]:
    """Sample unique ordered duels within one original dataset split.

    Outcomes are equally represented. Each outcome is distributed across
    its three suit combinations with counts differing by at most one.

    Sampling is reproducible for the same metadata, seed and configuration,
    independently of input row order. It does not modify global random state.

    Reversed pairs are allowed; repeated ordered pairs and self-pairs are not.
    The number of returned pairs is three times pairs_per_outcome.

    Args:
        cards: Validated, deduplicated metadata, possibly from all splits.
        split: Original split from which both images must be selected.
        pairs_per_outcome: Requested count for each of the three outcomes.
        seed: Explicit seed for this split's local random generator.

    Returns:
        An immutable sequence of labeled duels in shuffled order.

    Raises:
        MetadataError: Arguments, input metadata or sampling capacities
            do not satisfy the generation policy.

    Example:
        pairs = generate_pairs_for_split(
            cards,
            split=DatasetSplit.TRAIN,
            pairs_per_outcome=10_000,
            seed=42,
        )
        assert len(pairs) == 30_000
    """
    _require_sampling_arguments(split, pairs_per_outcome, seed)

    grouped = _group_cards(cards)
    suits = grouped[split]
    rng = random.Random(seed)

    plan = _plan_combination_counts(
        suits,
        pairs_per_outcome,
        rng,
    )

    pairs: list[CardPair] = []

    for first_suit, second_suit, requested in plan:
        first_cards = suits[first_suit]
        second_cards = suits[second_suit]
        same_suit = first_suit == second_suit

        capacity = _count_available_pairs(
            first_cards,
            second_cards,
            same_suit=same_suit,
        )

        sampled_indices = rng.sample(
            range(capacity),
            requested,
        )

        outcome = determine_outcome(first_suit, second_suit)

        for index in sampled_indices:
            first_card, second_card = _pair_from_index(
                index,
                first_cards,
                second_cards,
                same_suit=same_suit,
            )

            pairs.append(
                CardPair(
                    first_card=first_card,
                    second_card=second_card,
                    outcome=outcome,
                )
            )

    rng.shuffle(pairs)

    return tuple(pairs)

