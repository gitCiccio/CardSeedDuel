"""Single source of truth for dataset vocabulary and immutable constants.

Import these definitions rather than repeating column names, ranks or class
orders. No abstract class is needed: this module exposes immutable values.
Existing schema/result modules re-export enums for backward compatibility.
"""
from enum import Enum


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


class ValidationStatus(str, Enum):
    """Final classification of one original CSV row."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW = "review"

CARD_RANKS = frozenset({
    "ace",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "jack",
    "queen",
    "king",
})

REQUIRED_FIELDS = (
    "class index",
    "filepaths",
    "labels",
    "card type",
    "data set",
)

REQUIRED_COLUMNS = frozenset(REQUIRED_FIELDS)
DATASET_SPLITS = tuple(split.value for split in DatasetSplit)
INDEX_FIELDS = ("image_id", "path", "split", "original_label", "rank", "suit")
JOKER_LABEL = "joker"
JOKER_RANK = "xxx"
SUIT_CLASSES = (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES)
OUTCOME_CLASSES = (GameOutcome.P1_WIN, GameOutcome.P2_WIN, GameOutcome.DRAW)

IMAGE_EXTENSIONS = frozenset(['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'])
INDEX_COLUMNS = frozenset(INDEX_FIELDS)
