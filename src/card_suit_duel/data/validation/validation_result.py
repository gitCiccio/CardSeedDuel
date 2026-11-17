"""Structured results produced by dataset validation.

Each original CSV row receives exactly one final status.
A row can have multiple issues without being counted multiple times.
"""

from dataclasses import dataclass
from card_suit_duel.data.mapping.definitions import ValidationStatus

from card_suit_duel.data.mapping.schema import CardImage





@dataclass(frozen=True)
class ValidationIssue:
    """Describe one problem without deciding the row's final status.

    Attributes:
        code: Stable machine-readable identifier, such as "missing_file".
        message: Human-readable explanation of the problem.

    The owning row result supplies the CSV row number and image path.
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        """Reject empty issue identifiers and explanations."""
        for field_name, value in (
            ("code", self.code),
            ("message", self.message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )


@dataclass(frozen=True)
class RowValidationResult:
    """Store the final validation result for one CSV data row.

    Attributes:
        row_number: Logical CSV record number, starting at 2 because
            record 1 contains the column names.
        original_path: Path as declared in the CSV, or None if absent.
        status: Exactly one final classification.
        card: Parsed metadata when available. Its presence does not mean
            that the row was accepted; always inspect status.
        class_index: Original numeric class identifier, when available.
        issues: All problems recorded for this row.

    A rejected or review row may retain parsed metadata for diagnosis.
    Only accepted rows can enter the validated image index.
    """

    row_number: int
    original_path: str | None
    status: ValidationStatus
    card: CardImage | None = None
    class_index: int | None = None
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        """Check that the result has a coherent final classification."""
        if (
            isinstance(self.row_number, bool)
            or not isinstance(self.row_number, int)
            or self.row_number < 2
        ):
            raise ValueError(
                "row_number must be an integer starting at 2."
            )

        if self.original_path is not None and not isinstance(
            self.original_path, str
        ):
            raise TypeError("original_path must be a string or None.")

        if not isinstance(self.status, ValidationStatus):
            raise TypeError("status must be a ValidationStatus.")

        if self.card is not None and not isinstance(self.card, CardImage):
            raise TypeError("card must be a CardImage or None.")

        if self.class_index is not None and (
            isinstance(self.class_index, bool)
            or not isinstance(self.class_index, int)
            or self.class_index < 0
        ):
            raise ValueError(
                "class_index must be a nonnegative integer or None."
            )

        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, ValidationIssue)
            for issue in self.issues
        ):
            raise TypeError(
                "issues must be a tuple of ValidationIssue objects."
            )

        if self.status is ValidationStatus.ACCEPTED:
            if self.card is None or self.class_index is None:
                raise ValueError(
                    "Accepted rows require card metadata and class_index."
                )

            if self.issues:
                raise ValueError(
                    "Accepted rows cannot contain validation issues."
                )

        elif not self.issues:
            raise ValueError(
                "Rejected and review rows require at least one issue."
            )

@dataclass(frozen=True)
class DatasetValidationResult:
    """Collect the final row decisions and unreferenced image paths.

    Rows retain their logical CSV record numbers.
    Unreferenced images are reported but never added automatically.
    """

    rows: tuple[RowValidationResult, ...]
    unreferenced_images: tuple[str, ...]
    source_sha256: str
