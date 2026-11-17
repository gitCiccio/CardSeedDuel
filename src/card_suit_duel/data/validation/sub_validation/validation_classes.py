"""Cross-row class consistency and final index invariants."""
from collections import defaultdict
from card_suit_duel.data.mapping.definitions import ValidationStatus
from card_suit_duel.data.validation.validation_result import RowValidationResult


def find_class_conflicts(candidates: dict[int, tuple]) -> set[int]:
    """Find rows violating the original class-index/label correspondence.

    Both directions must be unique:
    one index per label and one label per index.
    """
    labels_by_index = defaultdict(set)
    indices_by_label = defaultdict(set)

    for card, class_index in candidates.values():
        labels_by_index[class_index].add(card.original_label)
        indices_by_label[card.original_label].add(class_index)

    return {
        number
        for number, (card, class_index) in candidates.items()
        if len(labels_by_index[class_index]) > 1
        or len(indices_by_label[card.original_label]) > 1
    }


def check_result_integrity(
    results: tuple[RowValidationResult, ...],
    expected_rows: int,
) -> None:
    """Check pipeline invariants rather than silently exporting bad output."""
    if len(results) != expected_rows:
        raise RuntimeError(
            "Every CSV row must have exactly one final result."
        )

    expected_numbers = list(range(2, expected_rows + 2))

    if [row.row_number for row in results] != expected_numbers:
        raise RuntimeError(
            "Validation results lost or duplicated CSV record numbers."
        )

    accepted_ids = [
        row.card.image_id
        for row in results
        if row.status is ValidationStatus.ACCEPTED
    ]

    if len(accepted_ids) != len(set(accepted_ids)):
        raise RuntimeError(
            "The accepted image index contains duplicate identifiers."
        )
