"""Load validated image metadata before applying game-domain filters."""

import csv
from collections.abc import Mapping
from pathlib import Path

from card_suit_duel.data.mapping.definitions import (
    CardSuit,
    DatasetSplit,
    INDEX_COLUMNS,
)
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.data.metadata_checks import (
    require_text_field,
    validate_required_columns,
    validate_record_shape,
)
from collections.abc import Sequence

from card_suit_duel.data.mapping.definitions import SUIT_CLASSES
def _parse_index_row(row: Mapping[str, str | None]) -> CardImage:
    """Reconstruct card metadata from one validated-index record.

        This function does not read image files. CardImage checks its own
        metadata invariants, including Joker and label/suit consistency.
    """
    validate_record_shape(row, source_name="Validated index")
    image_id = require_text_field(row, "image_id")
    path = require_text_field(row, "path")
    raw_split = require_text_field(row, "split")
    label = require_text_field(row, "original_label")
    rank = require_text_field(row, "rank")
    raw_suit = require_text_field(row,"suit", allow_empty=True)

    try:
        split = DatasetSplit(raw_split)
    except ValueError as exc:
        raise MetadataError(
            f"Unknown split in validated index: {raw_split!r}."
        ) from exc

    try:
        suit = CardSuit(raw_suit) if raw_suit else None
    except ValueError as exc:
        raise MetadataError(
            f"Unknown suit in validated index: {raw_suit!r}."
        ) from exc

    return CardImage(
        image_id=image_id,
        path=path,
        split=split,
        original_label=label,
        rank=rank,
        suit=suit,
    )


def _check_cards(cards: list[CardImage]) -> None:
    """Reject an empty index so downstream filtering cannot succeed silently."""
    if not cards:
        raise MetadataError(
            "The validated image index contains no cards."
        )


def _is_already_present(card: CardImage, seen_ids: set[str]) -> None:
    """Reject a repeated identifier, otherwise remember it for later rows."""
    if card.image_id in seen_ids:
        raise MetadataError(
            f"Repeated image_id: {card.image_id!r}."
        )

    seen_ids.add(card.image_id)


def load_validate_index(index_path: Path) -> list[CardImage]:
    """Load a validated CSV index into immutable card metadata objects.

        Unlike original-data validation, malformed index records stop the
        operation: filtering must not silently lose previously validated rows.

        Args:
            index_path: The validated_images.csv produced by F2.3.

        Returns:
            Card metadata in the same order as the index.

        Raises:
            MetadataError: If the file, header, records or identifiers
                violate the expected index contract.
    """
    cards: list[CardImage] = []
    seen_ids: set[str] = set()

    try:
        with index_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream, strict=True)
            validate_required_columns(
                reader.fieldnames, INDEX_COLUMNS, source_name="Validated index"
            )

            for row_number, row in enumerate(reader, start=2):
                try:
                    card = _parse_index_row(row)
                    _is_already_present(card, seen_ids)
                    cards.append(card)
                except MetadataError as exc:
                    raise MetadataError(
                        f"Invalid validated-index record "
                        f"{row_number}: {exc}"
                    ) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(
            "Cannot read or parse the validated image index."
        ) from exc

    _check_cards(cards)

    return cards

def filter_game_cards(
    cards: Sequence[CardImage],
) -> tuple[list[CardImage], list[CardImage]]:
    """Partition validated cards according to the game's allowed suits.

    Hearts, clubs and spades are included. Diamonds and Joker are excluded
    from the game, but remain valid records in the original dataset.

    Input order, metadata and original split membership are preserved.
    This function does not modify the input or access image files.

    Args:
        cards: Card metadata loaded from the validated image index.

    Returns:
        A pair containing the included cards and the excluded cards.

    Example:
        included, excluded = filter_game_cards(cards)
        assert len(included) + len(excluded) == len(cards)
        assert all(card.suit in SUIT_CLASSES for card in included)
    """
    included: list[CardImage] = []
    excluded: list[CardImage] = []

    for card in cards:
        if card.suit in SUIT_CLASSES:
            included.append(card)
        else:
            excluded.append(card)

    return included, excluded