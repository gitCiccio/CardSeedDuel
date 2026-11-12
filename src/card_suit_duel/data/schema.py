"""Shared categorical types for dataset metadata and game outcomes."""

from enum import Enum
from dataclasses import dataclass
from pathlib import PurePosixPath

from card_suit_duel.exceptions import MetadataError

class DatasetSplit(str, Enum):
    """Original dataset partitions, preserved during preparation."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class CardSuit(str, Enum):
    """Suits present in the original dataset.

    Diamonds belongs to the original dataset but is excluded from the duel.
    Joker has no suit and will be represented as None in card metadata.
    """

    HEARTS = "hearts"
    CLUBS = "clubs"
    SPADES = "spades"
    DIAMONDS = "diamonds"


class GameOutcome(str, Enum):
    """Possible outcomes from the first player's perspective."""

    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    DRAW = "draw"

@dataclass(frozen=True)
class CardImage:
    """Immutable metadata describing one original card image.

    Paths are relative to the dataset root and use forward slashes.
    This object contains metadata only; it does not load image pixels
    or verify that the referenced file exists.

    Attributes:
        image_id: Stable identifier, initially equal to the relative path.
        path: Image path relative to data/raw.
        split: Original dataset partition.
        original_label: Original class label, such as "ace of hearts".
        rank: Original card rank/type, preserved for analysis.
        suit: Original suit, or None for Joker.

    Raises:
        MetadataError: If a field violates the metadata contract.
    """

    image_id: str
    path: str
    split: DatasetSplit
    original_label: str
    rank: str
    suit: CardSuit | None

    def __post_init__(self) -> None:
        """Validate metadata without changing the supplied values."""
        text_fields = {
            "image_id": self.image_id,
            "path": self.path,
            "original_label": self.original_label,
            "rank": self.rank,
        }

        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise MetadataError(
                    f"{field_name} must be a non-empty string; "
                    f"received {value!r}."
                )

            if value != value.strip():
                raise MetadataError(
                    f"{field_name} contains leading or trailing "
                    f"whitespace: {value!r}."
                )

        if not isinstance(self.split, DatasetSplit):
            raise MetadataError(
                f"split must be a DatasetSplit; "
                f"received {self.split!r}."
            )

        if self.suit is not None and not isinstance(self.suit, CardSuit):
            raise MetadataError(
                f"suit must be a CardSuit or None; "
                f"received {self.suit!r}."
            )

        if "\\" in self.path or ":" in self.path:
            raise MetadataError(
                f"path must use a relative POSIX-style path: "
                f"{self.path!r}."
            )

        relative_path = PurePosixPath(self.path)

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != self.path
        ):
            raise MetadataError(
                f"path must be a normalized relative path "
                f"without parent traversal: {self.path!r}."
            )

        if (
            len(relative_path.parts) < 3
            or relative_path.parts[0] != self.split.value
        ):
            raise MetadataError(
                f"path must start with {self.split.value!r} "
                f"and contain a class directory and filename: "
                f"{self.path!r}."
            )

        if self.image_id != self.path:
            raise MetadataError(
                "image_id must equal path in the current schema; "
                f"received {self.image_id!r} and {self.path!r}."
            )

        is_joker = self.original_label == "joker"

        if is_joker and self.suit is not None:
            raise MetadataError("Joker must have suit=None.")

        if not is_joker and self.suit is None:
            raise MetadataError(
                "Only Joker can have suit=None."
            )

        if is_joker:
            if self.rank != "xxx":
                raise MetadataError("Joker must use the original rank marker 'xxx'.")
        else:
            expected_label = f"{self.rank} of {self.suit.value}"
            if self.original_label != expected_label:
                raise MetadataError(
                    f"original_label must match rank and suit: expected "
                    f"{expected_label!r}, received {self.original_label!r}."
                )
