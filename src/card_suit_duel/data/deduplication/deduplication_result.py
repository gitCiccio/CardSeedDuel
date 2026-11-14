"""Structured results of exact-image deduplication."""

from dataclasses import dataclass

from card_suit_duel.data.mapping.schema import CardImage

@dataclass(frozen=True)
class DuplicateGroup:
    """Describe an exact-pixel duplicate group and its decision.

        representative is the retained card when metadata is consistent.
        A missing representative means that the whole group requires review.

        members contains every card in the group, including the representative.
    """

    pixels_sha256: str
    members: tuple[CardImage, ...]
    representative: CardImage | None

    @property
    def requires_review(self) -> bool:
        """Return whether conflicting metadata prevents automatic selection."""
        return self.representative is None

@dataclass(frozen=True)
class DeduplicationResult:
    """Partition input cards into retained, removed and review categories.

        groups contains only groups with at least two members.
        Cards requiring review are excluded from the provisional retained index.
    """

    retained: tuple[CardImage, ...]
    removed: tuple[CardImage, ...]
    review: tuple[CardImage, ...]
    groups: tuple[DuplicateGroup, ...]

