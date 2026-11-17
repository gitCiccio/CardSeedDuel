"""Reusable exact-deduplication stage with reproducible manual-review input."""
import hashlib
import json
from pathlib import Path

from card_suit_duel.data.deduplication.deduplication import deduplicate_cards
from card_suit_duel.data.deduplication.fingerprints import fingerprint_card
from card_suit_duel.data.deduplication.manual_review import apply_manual_decisions
from card_suit_duel.data.deduplication.exporting import export_deduplication
from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.exceptions import MetadataError


def run_deduplication(index_path: Path, dataset_root: Path, output_directory: Path, *, manual_decisions: list) -> Path:
    """Apply exact grouping, reviewed decisions and export an approved index."""
    source_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    cards = load_validate_index(index_path)
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != source_hash:
        raise MetadataError("The deduplication input changed during loading.")
    fingerprints = [fingerprint_card(card, dataset_root) for card in cards]
    result = deduplicate_cards(cards, fingerprints)
    result = apply_manual_decisions(cards, result, manual_decisions)
    export_deduplication(cards, result, fingerprints, output_directory, source_hash, manual_decisions)
    if result.review or not result.retained:
        raise MetadataError("Deduplication requires manual review before continuing.")
    return output_directory
