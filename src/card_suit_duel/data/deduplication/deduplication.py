"""Group exact image duplicates and apply deterministic retention rules."""

from collections import defaultdict
from collections.abc import Sequence

from card_suit_duel.data.deduplication.deduplication_result import (
    DeduplicationResult,
    DuplicateGroup,
)
from card_suit_duel.data.deduplication.fingerprints import ImageFingerprint
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


_SPLIT_PRIORITY = {
    DatasetSplit.TEST: 0,
    DatasetSplit.VALID: 1,
    DatasetSplit.TRAIN: 2,
}


def _require_unique_card_ids(card_ids: set[str], cards: Sequence[CardImage]) -> None:
    """Reject repeated input identifiers before associating fingerprints."""
    if len(card_ids) != len(cards):
        raise MetadataError(
            "Deduplication input contains repeated image identifiers."
        )

def _require_new_fingerprint(image_id: str, indexed: dict[str, ImageFingerprint]) -> None:
    """Require a previously unseen image identifier in the fingerprint index."""
    if image_id in indexed:
        raise MetadataError(
            "Repeated fingerprint for image_id: "
            f"{image_id!r}."
        )

def _index_fingerprints(
    cards: Sequence[CardImage],
    fingerprints: Sequence[ImageFingerprint],
) -> dict[str, ImageFingerprint]:
    """Require exactly one fingerprint for each unique input image_id."""
    logger.info("Checking associations for %d cards and %d fingerprints", len(cards), len(fingerprints))
    card_ids = {card.image_id for card in cards}

    _require_unique_card_ids(card_ids, cards)

    indexed: dict[str, ImageFingerprint] = {}

    for fingerprint in fingerprints:
        _require_new_fingerprint(fingerprint.image_id, indexed)

        indexed[fingerprint.image_id] = fingerprint

    fingerprint_ids = set(indexed)

    if fingerprint_ids != card_ids:
        # Set differences identify cards without fingerprints and fingerprints
        # whose identifiers do not belong to the input index.
        missing = len(card_ids - fingerprint_ids)
        unexpected = len(fingerprint_ids - card_ids)

        raise MetadataError(
            "Fingerprints do not match input cards: "
            f"{missing} missing, {unexpected} unexpected."
        )

    logger.debug("Fingerprint associations are complete and unique")
    return indexed

def _has_metadata_conflicts(
    members: Sequence[CardImage],
) -> bool:
    """Detect disagreement about the identity of an exact-pixel image."""
    signatures = {
        (card.original_label, card.rank, card.suit)
        for card in members
    }

    return len(signatures) > 1


def _select_representative(
    members: Sequence[CardImage],
) -> CardImage:
    """Select by test, valid, train priority, then lexical image_id."""
    return min(
        members,
        key=lambda card: (
            _SPLIT_PRIORITY[card.split],
            card.image_id,
        ),
    )


def deduplicate_cards(
    cards: Sequence[CardImage],
    fingerprints: Sequence[ImageFingerprint],
) -> DeduplicationResult:
    """Partition cards using exact decoded-pixel identity.

    Consistent groups retain one representative according to split priority
    and lexical image_id. Conflicting groups require manual review.

    Retained, removed and review cards preserve their relative input order.
    Group descriptions and representative selection are deterministic.

    This function does not read, modify or delete image files.

    Args:
        cards: Metadata loaded from the filtered image index.
        fingerprints: Exactly one computed fingerprint for each input card.

    Returns:
        Retained cards, confirmed removals, review cards and duplicate groups.

    Raises:
        MetadataError: Input identifiers or fingerprint associations are invalid.

    Example:
        result = deduplicate_cards(cards, fingerprints)
        assert len(cards) == (
            len(result.retained)
            + len(result.removed)
            + len(result.review)
        )
    """
    indexed = _index_fingerprints(cards, fingerprints)

    buckets: dict[str, list[CardImage]] = defaultdict(list)

    for card in cards:
        fingerprint = indexed[card.image_id]
        buckets[fingerprint.pixels_sha256].append(card)

    retained_ids: set[str] = set()
    removed_ids: set[str] = set()
    review_ids: set[str] = set()
    groups: list[DuplicateGroup] = []

    for pixels_hash, members in sorted(buckets.items()):
        if len(members) == 1:
            retained_ids.add(members[0].image_id)
            continue

        ordered_members = tuple(
            sorted(members, key=lambda card: card.image_id)
        )

        if _has_metadata_conflicts(ordered_members):
            logger.warning("Duplicate group requires review: %d conflicting records", len(ordered_members))
            representative = None
            review_ids.update(
                card.image_id
                for card in ordered_members
            )
        else:
            representative = _select_representative(ordered_members)
            retained_ids.add(representative.image_id)

            removed_ids.update(
                card.image_id
                for card in ordered_members
                if card.image_id != representative.image_id
            )

        groups.append(
            DuplicateGroup(
                pixels_sha256=pixels_hash,
                members=ordered_members,
                representative=representative,
            )
        )

    logger.info("Deduplication decisions: %d retained, %d removed, %d review", len(retained_ids), len(removed_ids), len(review_ids))
    return DeduplicationResult(
        retained=tuple(
            card for card in cards
            if card.image_id in retained_ids
        ),
        removed=tuple(
            card for card in cards
            if card.image_id in removed_ids
        ),
        review=tuple(
            card for card in cards
            if card.image_id in review_ids
        ),
        groups=tuple(groups),
    )