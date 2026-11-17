"""Apply explicit, fingerprint-bound review decisions without changing raw files."""
from collections.abc import Mapping, Sequence
from dataclasses import replace

from card_suit_duel.data.deduplication.deduplication_result import DeduplicationResult
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError


def apply_manual_decisions(
    cards: Sequence[CardImage],
    result: DeduplicationResult,
    decisions: Sequence[Mapping],
) -> DeduplicationResult:
    """Resolve only known conflicting groups with an explicit retained member.

    A decision is bound to the group's pixel fingerprint, not a run timestamp.
    Unknown, repeated or incompatible decisions fail instead of being ignored.
    Every non-representative member of an approved group is excluded.
    """
    groups = {group.pixels_sha256: group for group in result.groups}
    updates = {}
    retained = {card.image_id for card in result.retained}
    removed = {card.image_id for card in result.removed}
    review = {card.image_id for card in result.review}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise MetadataError("Each manual decision must be an object.")
        key = decision.get("pixels_sha256")
        if not isinstance(key, str) or key not in groups or key in updates:
            raise MetadataError("Unknown or repeated manual-review fingerprint.")
        group = groups[key]
        if not group.requires_review:
            raise MetadataError("Manual decisions may resolve only review groups.")
        members = {card.image_id: card for card in group.members}
        representative = decision.get("representative_id")
        reason = decision.get("reason")
        if representative not in members:
            raise MetadataError("The manual representative is not a group member.")
        if not isinstance(reason, str) or not reason.strip():
            raise MetadataError("A manual decision requires a documented reason.")
        excluded = set(members) - {representative}
        if "removed_id" in decision and excluded != {decision["removed_id"]}:
            raise MetadataError("The explicit removed image does not match the group.")
        retained.add(representative)
        removed.update(excluded)
        review.difference_update(members)
        updates[key] = replace(group, representative=members[representative])
    return DeduplicationResult(
        retained=tuple(card for card in cards if card.image_id in retained),
        removed=tuple(card for card in cards if card.image_id in removed),
        review=tuple(card for card in cards if card.image_id in review),
        groups=tuple(updates.get(group.pixels_sha256, group) for group in result.groups),
    )
