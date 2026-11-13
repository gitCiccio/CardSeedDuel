"""Shared CSV header and text checks, independent of card-specific rules."""

from collections.abc import Collection, Mapping, Sequence

from card_suit_duel.exceptions import MetadataError


def validate_required_columns(
    columns: Sequence[str] | None,
    required_columns: Collection[str],
    *,
    source_name: str = "CSV",
) -> None:
    """Require an unambiguous header containing every required column.

    Additional columns are allowed. No names are normalized.

    Example:
        >>> validate_required_columns(["path", "suit"], {"path"})
    """
    if not columns:
        raise MetadataError(f"{source_name} has no CSV header.")
    if len(columns) != len(set(columns)):
        raise MetadataError(f"{source_name} contains duplicate column names.")
    missing = set(required_columns) - set(columns)
    if missing:
        raise MetadataError(
            f"{source_name} is missing required columns: {sorted(missing)}."
        )


def require_text_field(
    row: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
    source_name: str = "Metadata",
) -> str:
    """Return unchanged text, rejecting missing values and outer whitespace.

    Empty strings require explicit permission from the caller. Domain rules,
    such as allowing an empty suit for Joker, belong to the calling parser.

    Examples:
        >>> require_text_field({"split": "train"}, "split")
        'train'
        >>> require_text_field({"suit": ""}, "suit", allow_empty=True)
        ''
    """
    value = row.get(field)
    if not isinstance(value, str):
        raise MetadataError(
            f"{source_name} field {field!r} is missing or is not text."
        )
    if value != value.strip():
        raise MetadataError(
            f"{source_name} field {field!r} contains outer whitespace."
        )
    if not allow_empty and not value:
        raise MetadataError(f"{source_name} field {field!r} cannot be empty.")
    return value
