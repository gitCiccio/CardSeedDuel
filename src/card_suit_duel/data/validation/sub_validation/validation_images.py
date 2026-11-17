"""File acceptance, repeated references and filesystem reconciliation."""
from pathlib import Path
from card_suit_duel.data.mapping.definitions import ValidationStatus
from card_suit_duel.data.inspection import scan_dataset_files
from card_suit_duel.data.validation.validation import validate_card_file
from card_suit_duel.data.validation.validation_result import RowValidationResult, ValidationIssue
from card_suit_duel.data.validation.sub_validation.validation_paths import path_key
from card_suit_duel.exceptions import ImageValidationError


def classify_candidates(
    candidates: dict[int, tuple],
    path_conflicts: set[str],
    class_conflicts: set[int],
    dataset_root: Path,
) -> dict[int, RowValidationResult]:
    """Assign review, rejected or accepted status to parsed candidates.

    Conflicts take precedence over file acceptance.
    Files are decoded at most once per identical relative reference.
    """
    results = {}
    file_outcomes = {}

    for row_number, (card, class_index) in candidates.items():
        issues = []

        if card.path in path_conflicts:
            issues.append(ValidationIssue(
                code="path_metadata_conflict",
                message=(
                    "The same image path has conflicting metadata."
                ),
            ))

        if row_number in class_conflicts:
            issues.append(ValidationIssue(
                code="class_index_conflict",
                message=(
                    "Original class indices and labels "
                    "do not have a one-to-one correspondence."
                ),
            ))

        if issues:
            status = ValidationStatus.REVIEW

        elif card.path in file_outcomes:
            previous_issue = file_outcomes[card.path]

            if previous_issue is None:
                issues.append(ValidationIssue(
                    code="duplicate_reference",
                    message=(
                        "An identical reference was already accepted."
                    ),
                ))
            else:
                issues.append(previous_issue)

            status = ValidationStatus.REJECTED

        else:
            try:
                validate_card_file(card, dataset_root)

            except ImageValidationError as exc:
                issue = ValidationIssue(
                    code=exc.reason,
                    message=(
                        "The referenced image failed file validation "
                        f"with reason: {exc.reason}."
                    ),
                )
                file_outcomes[card.path] = issue
                issues.append(issue)
                status = ValidationStatus.REJECTED

            else:
                file_outcomes[card.path] = None
                status = ValidationStatus.ACCEPTED

        results[row_number] = RowValidationResult(
            row_number=row_number,
            original_path=card.path,
            status=status,
            card=card,
            class_index=class_index,
            issues=tuple(issues),
        )

    return results


def find_unreferenced_images(
    rows: list[dict],
    dataset_root: Path,
) -> tuple[str, ...]:
    """Report scanned images absent from all original CSV references."""
    references = {
        key
        for row in rows
        if (key := path_key(row.get("filepaths"))) is not None
    }

    files_by_split = scan_dataset_files(dataset_root)

    disk_paths = {
        path.relative_to(dataset_root).as_posix()
        for paths in files_by_split.values()
        for path in paths
    }

    return tuple(sorted(disk_paths - references))
