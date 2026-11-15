"""Load exported duels using the validated image metadata contract."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from card_suit_duel.data.mapping.definitions import (
    DatasetSplit,
    GameOutcome,
    PAIR_COLUMNS,
)
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.metadata_checks import (
    require_text_field,
    require_unique_identifiers,
    validate_record_shape,
    validate_required_columns,
)
from card_suit_duel.data.pairs.schema import CardPair
from card_suit_duel.exceptions import MetadataError

def _index_cards(
        cards: Sequence[CardImage]
) -> dict[str, CardImage]:
    """Associate each unique image identifier with its card metadata."""
    require_unique_identifiers(
        (card.image_id for card in cards),
        source_name="Pair reader image index"
    )

    return {
        card.image_id: card for card in cards
    }

def _parse_pair_row(
    row: Mapping[str | None, object],
    indexed_cards: Mapping[str, CardImage],
) -> CardPair:
    """Reconstruct a duel and verify its exported identity and split."""
    validate_record_shape(row, source_name="Pair index")

    pair_id = require_text_field(row, "pair_id")
    raw_split = require_text_field(row, "split")
    first_id = require_text_field(row, "first_image_id")
    second_id = require_text_field(row, "second_image_id")
    raw_outcome = require_text_field(row, "outcome")

    try:
        first_card = indexed_cards[first_id]
        second_card = indexed_cards[second_id]
    except KeyError as exc:
        raise MetadataError(
            f"Pair references an unknown image: {exc.args[0]!r}."
        ) from exc

    try:
        split = DatasetSplit(raw_split)
    except ValueError as exc:
        raise MetadataError(
            f"Unknown pair split: {raw_split!r}."
        ) from exc

    try:
        outcome = GameOutcome(raw_outcome)
    except ValueError as exc:
        raise MetadataError(
            f"Unknown pair outcome: {raw_outcome!r}."
        ) from exc

    pair = CardPair(
        first_card=first_card,
        second_card=second_card,
        outcome=outcome,
    )

    if pair.split is not split:
        raise MetadataError(
            "The exported pair split does not match its images."
        )

    if pair.pair_id != pair_id:
        raise MetadataError(
            "The exported pair_id does not match its ordered images."
        )

    return pair

def load_pairs(
    pairs_path: Path,
    cards: Sequence[CardImage],
    *,
    expected_split: DatasetSplit,
) -> tuple[CardPair, ...]:
    """Load unique ordered duels belonging to one expected dataset split.

    Image files are not opened. Every referenced image must exist in the
    supplied metadata index. Invalid records stop the operation.

    Args:
        pairs_path: Exported train, validation or test pair CSV.
        cards: Metadata from the corresponding deduplicated image index.
        expected_split: Split required for every loaded pair.

    Returns:
        Validated duels in their original CSV order.

    Raises:
        MetadataError: The input file or its records violate the contract.

    Example:
        pairs = load_pairs(
            train_pairs_path,
            cards,
            expected_split=DatasetSplit.TRAIN,
        )
        assert all(pair.split is DatasetSplit.TRAIN for pair in pairs)
    """
    if not isinstance(expected_split, DatasetSplit):
        raise MetadataError(
            "expected_split must be a DatasetSplit member."
        )

    indexed_cards = _index_cards(cards)
    pairs: list[CardPair] = []
    seen_ids: set[str] = set()

    try:
        with pairs_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream, strict=True)

            validate_required_columns(
                reader.fieldnames,
                PAIR_COLUMNS,
                source_name="Pair index",
            )

            for row_number, row in enumerate(reader, start=2):
                try:
                    pair = _parse_pair_row(row, indexed_cards)

                    if pair.split is not expected_split:
                        raise MetadataError(
                            "Pair belongs to an unexpected dataset split."
                        )

                    if pair.pair_id in seen_ids:
                        raise MetadataError(
                            "Repeated ordered pair in the input index."
                        )

                except MetadataError as exc:
                    raise MetadataError(
                        f"Invalid pair-index record {row_number}: {exc}"
                    ) from exc

                seen_ids.add(pair.pair_id)
                pairs.append(pair)

    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(
            "Cannot read or parse the pair index."
        ) from exc

    if not pairs:
        raise MetadataError(
            "The pair index contains no duels."
        )

    return tuple(pairs)