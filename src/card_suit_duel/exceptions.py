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

class MappingError(CardSuitDuelError):
    """Base class for failures in the shared class-to-index contract.

    Catch this type where an invalid mapping input can be reported or handled.
    Mapping failures can occur in data preparation, training or inference, so
    they are not restricted to dataset errors.
    """


class InvalidMappingValueError(MappingError):
    """A value cannot be mapped to one of the configured model classes.

    Use for unknown labels, unsupported suits (such as diamonds in the duel),
    or a missing suit. Include the supplied value and accepted values in the
    message. A valid original card may still be outside the game's domain.
    """


class InvalidMappingIndexError(MappingError):
    """An index is not a valid integer position in the shared class order.

    Use for out-of-range indices, negative indices or incorrect input types.
    Reject booleans explicitly: Python treats bool as an int subclass.
    Include the invalid index and permitted range in the message.
    """

class ImageValidationError(DatasetError):
    """An individual image reference cannot be accepted.

    Attributes:
        reason: Machine-readable rejection code for validation reports.
        image_path: Original image path relative to the dataset root.

    This error describes one rejected sample, not a failure of the
    entire dataset. The caller decides whether to collect it and continue.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        image_path: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.image_path = image_path
