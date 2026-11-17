"""Convert original CSV rows into the shared CardImage metadata contract.

Public API: parse_card_row(row), validate_card_file(card, dataset_root). Internal helpers and constants use a leading
underscore. In Python this means "internal implementation detail" by convention;
it does not enforce access restrictions. These are module functions, not methods.

Execution flow:
    1. Read required text fields through require_text_field.
    2. Validate the original class index and convert the split to DatasetSplit.
    3. Derive rank and suit, with a separate rule for Joker.
    4. Construct CardImage, which validates the shared metadata invariants.
    5. Check that the path's class directory matches the original label.
    6. Return metadata without reading or modifying an image file.

Callers should use the public functions rather than importing internal helpers.
File validation resolves the root and image path, checks containment and file
existence, then decodes pixels. It never modifies the source image.
Invalid rows raise MetadataError; collecting rejections is the caller's job.
"""
from collections.abc import Mapping
from card_suit_duel.data.metadata_checks import require_text_field
from card_suit_duel.data.path_checks import resolve_card_path
from card_suit_duel.data.mapping.definitions import (
    CARD_RANKS, REQUIRED_FIELDS, JOKER_LABEL, JOKER_RANK,
)

from card_suit_duel.data.mapping.schema import (
    CardImage,
    CardSuit,
    DatasetSplit,
)
from card_suit_duel.exceptions import (
    ImageValidationError,
    MetadataError,
)
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Immutable membership set for the thirteen original non-Joker ranks.




def _validate_card_rank(rank: str) -> None:
    """Require a recognized rank for a non-Joker card.

    Internal check used after required text validation. Joker is handled by
    its separate parsing branch and must not be passed here. Raises
    MetadataError for values outside the thirteen original ranks; no value
    is returned and no normalization is performed.
    """
    if rank not in CARD_RANKS:
        raise MetadataError(
            f"Unknown card rank: {rank!r}."
        )


def _validate_label_rank(label_rank: str, rank: str) -> None:
    """Require the rank parsed from the label to match the card type column.

    Internal consistency check used after splitting a non-Joker label at
    ' of '. A mismatch raises MetadataError with both observed values.
    Returns None on success; it does not choose which value is correct.
    """
    if label_rank != rank:
        raise MetadataError(
            f"Label rank {label_rank!r} does not match "
            f"card type {rank!r}."
        )


def _validate_card_path_prefix(card: CardImage) -> None:
    """Require the relative path to start with the split and original label.

    Internal check performed after CardImage construction, for normal cards
    and Joker alike. For example, an ace of hearts in train must live under
    'train/ace of hearts/'. Raises MetadataError on a mismatch. This only
    checks metadata; it does not access the filesystem or decode the image.
    """
    expected_prefix = f"{card.split.value}/{card.original_label}/"
    if not card.path.startswith(expected_prefix):
        raise MetadataError(
            f"Image path {card.path!r} must start with "
            f"{expected_prefix!r}."
        )


def parse_card_row(
        row: Mapping[str, str | None],
) -> CardImage:
    """Convert one original CSV row into validated card metadata.

    This is the public entry point for row parsing. It does not check file
    existence, decode pixels, remove duplicates or filter game suits. A valid
    Diamonds card remains representable; Joker has rank 'xxx' and suit=None.

    Args:
        row: Original CSV fields. Names and values are matched exactly;
            class index is checked but is not used as a model class index.

    Returns:
        A CardImage satisfying the shared contract and class-directory check.

    Raises:
        MetadataError: If required values are invalid or inconsistent.

    Examples:
        A normal card is parsed without accessing its referenced image:

        >>> row = {
        ...     "class index": "0",
        ...     "filepaths": "train/ace of hearts/001.jpg",
        ...     "labels": "ace of hearts",
        ...     "card type": "ace",
        ...     "data set": "train",
        ... }
        >>> card = parse_card_row(row)
        >>> card.suit.value, card.rank, card.split.value
        ('hearts', 'ace', 'train')
        >>> card.image_id == card.path
        True

        Joker follows the original dataset representation:

        >>> joker_row = {
        ...     "class index": "20",
        ...     "filepaths": "train/joker/001.jpg",
        ...     "labels": "joker",
        ...     "card type": "xxx",
        ...     "data set": "train",
        ... }
        >>> joker = parse_card_row(joker_row)
        >>> joker.original_label, joker.rank, joker.suit
        ('joker', 'xxx', None)

        Missing fields are rejected with a domain-specific error:

        >>> parse_card_row({})
        Traceback (most recent call last):
            ...
        card_suit_duel.exceptions.MetadataError: Metadata field 'class index' is missing or is not text.
    """

    values = {
        field: require_text_field(row, field)
        for field in REQUIRED_FIELDS
    }

    raw_class_index = values["class index"]

    if not raw_class_index.isascii() or not raw_class_index.isdecimal():
        raise MetadataError(
            f"Class index must be a nonnegative integer: "
            f"{raw_class_index!r}."
        )

    try:
        split = DatasetSplit(values["data set"])
    except ValueError as exc:
        raise MetadataError(
            f"Unknown dataset split: {values['data set']!r}."
        ) from exc

    label = values["labels"]
    rank = values["card type"]
    path = values["filepaths"]

    if label == JOKER_LABEL:
        suit = None

        if rank != JOKER_RANK:
            raise MetadataError(
                f"Joker must use rank 'xxx'; received {rank!r}."
            )
    else:
        _validate_card_rank(rank)

        label_rank, separator, suit_name = label.partition(" of ")

        if not separator:
            raise MetadataError(
                f"Invalid card label: {label!r}. "
                "Expected '<rank> of <suit>'."
            )

        _validate_label_rank(label_rank, rank)

        try:
            suit = CardSuit(suit_name)
        except ValueError as exc:
            raise MetadataError(
                f"Unknown suit in label {label!r}: "
                f"{suit_name!r}."
            ) from exc

    card = CardImage(
        image_id=path,
        path=path,
        split=split,
        original_label=label,
        rank=rank,
        suit=suit,
    )

    _validate_card_path_prefix(card)

    return card

def _decode_image(image_path: Path, original_path: str) -> None:
    """Attempt full pixel decoding without converting or saving the image.

    Internal check: Image.open identifies the format; load forces pixel reading.
    The context manager closes the file on success and failure. Known decoding
    failures are wrapped as ImageValidationError with reason 'unreadable_image',
    preserving the original exception as the cause.
    """
    try:
        with Image.open(image_path) as image:
            image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageValidationError(
            f"Cannot decode image {original_path}: {exc}",
            reason="unreadable_image",
            image_path=original_path,
        ) from exc


def validate_card_file(card: CardImage, dataset_root: Path) -> Path:
    """Accept a card's file only when its location and pixels are valid.

    Public entry point: call this after parse_card_row(row). It delegates
    filesystem checks to the shared resolve_card_path function, then decodes
    the image through its private helper. It does not resize images, filter suits or write
    files. Metadata consistency is the parser and CardImage's responsibility.

    Args:
        card: Previously validated card metadata.
        dataset_root: Directory containing the original dataset splits.

    Returns:
        The resolved absolute path of the readable image.

    Raises:
        DatasetStructureError: The root is invalid; the caller should stop.
        ImageValidationError: One sample is invalid; the caller may record
            reason/image_path and continue with other rows.

    Examples:
        This self-contained example creates a temporary image, not dataset data.
        The same two public calls will be used for real CSV rows:

        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     root = Path(directory)
        ...     path = root / "train/ace of hearts/001.jpg"
        ...     path.parent.mkdir(parents=True)
        ...     with Image.new("RGB", (8, 8)) as image:
        ...         image.save(path, format="JPEG")
        ...     row = {
        ...         "class index": "0",
        ...         "filepaths": "train/ace of hearts/001.jpg",
        ...         "labels": "ace of hearts",
        ...         "card type": "ace",
        ...         "data set": "train",
        ...     }
        ...     card = parse_card_row(row)
        ...     validated_path = validate_card_file(card, root)
        ...     print(validated_path == path.resolve())
        ...     path.unlink()
        ...     try:
        ...         validate_card_file(card, root)
        ...     except ImageValidationError as error:
        ...         print(error.reason, error.image_path)
        True
        missing_file train/ace of hearts/001.jpg

        In a real pipeline, store the successful CardImage in the valid index.
        On ImageValidationError, store the row number, reason and image_path in
        the rejection report. The caller handles logging to avoid duplicates.
    """
    image_path = resolve_card_path(card, dataset_root)
    _decode_image(image_path, card.path)
    return image_path
