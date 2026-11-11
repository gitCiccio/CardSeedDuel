"""Application exceptions with meaningful dataset error categories.

Raise an exception when a caller cannot complete an operation, and include the
affected path, expected condition and observed problem in its message. Preserve
an underlying exception with ``raise MetadataError(message) from exc``.

Example::

    from card_suit_duel.exceptions import MetadataError

    if "filepaths" not in columns:
        raise MetadataError(
            f"Missing required column 'filepaths' in {csv_path}; found {columns}"
        )

For dataset inspection, individual invalid images are findings to record in the
report and log as warnings; they need not abort the entire scan. Reserve raised
exceptions for failures that prevent the requested operation from continuing.
Catch errors at the boundary that can handle them. Avoid logging an error and
re-raising it at every layer, which would duplicate messages and tracebacks.
"""


class CardSuitDuelError(Exception):
    """Base class for application-specific failures.

    Catch this type at an entry point to handle known application errors together.
    Raise a more specific subclass when one describes the problem. Standard
    Python exceptions do not need wrapping unless extra domain context is useful.
    """


class DatasetError(CardSuitDuelError):
    """Base class for failures that prevent a dataset operation from completing.

    Catch this type when the same recovery applies to structure and metadata
    failures. Do not raise it merely because inspection found a bad sample.
    """


class DatasetStructureError(DatasetError):
    """A required dataset path is missing or has an incompatible structure.

    Use for a missing dataset root, required split directory or metadata file.
    Include the relevant path and what was expected in the exception message.
    """


class MetadataError(DatasetError):
    """Required dataset metadata cannot be parsed or does not match its schema.

    Use for malformed CSV input or missing mandatory columns. State the source
    path, expected schema and actual problem. Preserve parsing errors as causes.
    Individual invalid rows may instead be reported and skipped when the agreed
    inspection policy allows the operation to continue.
    """
