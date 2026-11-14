"""Behavioral tests for exact file and pixel fingerprints.

Images are generated in temporary directories; no original data is required.
"""

import hashlib
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from card_suit_duel.data.deduplication.fingerprints import fingerprint_card
from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import DatasetStructureError, ImageValidationError


class FingerprintTests(unittest.TestCase):
    """Verify identity semantics, metadata independence and explicit failures."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "raw"
        self.root.mkdir()

    def card(self, filename="001.png", suit=CardSuit.HEARTS):
        label = f"ace of {suit.value}"
        path = f"train/{label}/{filename}"
        return CardImage(path, path, DatasetSplit.TRAIN, label, "ace", suit)

    def save(self, card, *, size=(8, 8), color=(12, 34, 56), mode="RGB", metadata=None):
        path = self.root / card.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new(mode, size, color) as image:
            image.save(path, format="PNG", pnginfo=metadata)
        return path

    def assert_reason(self, card, reason):
        with self.assertRaises(ImageValidationError) as caught:
            fingerprint_card(card, self.root)
        self.assertEqual(caught.exception.reason, reason)
        self.assertEqual(caught.exception.image_path, card.path)
        return caught.exception

    def test_result_is_stable_associated_and_immutable(self):
        card = self.card()
        path = self.save(card)
        before = path.read_bytes()
        result = fingerprint_card(card, self.root)
        self.assertEqual(result, fingerprint_card(card, self.root))
        self.assertEqual(result.image_id, card.image_id)
        self.assertEqual(result.file_sha256, hashlib.sha256(before).hexdigest())
        self.assertRegex(result.pixels_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(FrozenInstanceError):
            result.image_id = "changed"

    def test_identical_files_at_different_paths_match_both_hashes(self):
        first, second = self.card(), self.card("002.png")
        source = self.save(first)
        target = self.root / second.path
        target.write_bytes(source.read_bytes())
        a, b = (fingerprint_card(c, self.root) for c in (first, second))
        self.assertNotEqual(a.image_id, b.image_id)
        self.assertEqual(a.file_sha256, b.file_sha256)
        self.assertEqual(a.pixels_sha256, b.pixels_sha256)

    def test_different_file_metadata_preserves_pixel_hash(self):
        first, second = self.card(), self.card("002.png")
        self.save(first)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("description", "same pixels, different file bytes")
        self.save(second, metadata=metadata)
        a, b = (fingerprint_card(c, self.root) for c in (first, second))
        self.assertNotEqual(a.file_sha256, b.file_sha256)
        self.assertEqual(a.pixels_sha256, b.pixels_sha256)

    def test_different_pixels_have_different_hashes(self):
        first, second = self.card(), self.card("002.png")
        self.save(first)
        self.save(second, color=(12, 34, 57))
        a, b = (fingerprint_card(c, self.root) for c in (first, second))
        self.assertNotEqual(a.pixels_sha256, b.pixels_sha256)

    def test_dimensions_distinguish_equal_raw_pixel_sequences(self):
        first, second = self.card(), self.card("002.png")
        self.save(first, size=(4, 8))
        self.save(second, size=(8, 4))
        with Image.open(self.root / first.path) as a, Image.open(self.root / second.path) as b:
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotEqual(
            fingerprint_card(first, self.root).pixels_sha256,
            fingerprint_card(second, self.root).pixels_sha256,
        )

    def test_metadata_labels_do_not_affect_content_hashes(self):
        first = self.card()
        second = self.card(suit=CardSuit.CLUBS)
        source = self.save(first)
        target = self.root / second.path
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        self.assertEqual(
            fingerprint_card(first, self.root).pixels_sha256,
            fingerprint_card(second, self.root).pixels_sha256,
        )

    def test_non_rgb_modes_are_rejected_without_conversion(self):
        for mode in ("L", "P", "RGBA"):
            with self.subTest(mode=mode):
                card = self.card(f"{mode}.png")
                self.save(card, mode=mode, color=0)
                self.assert_reason(card, "unexpected_color_mode")

    def test_missing_file_preserves_domain_reason(self):
        self.assert_reason(self.card(), "missing_file")

    def test_corrupt_image_is_rejected(self):
        card = self.card()
        path = self.root / card.path
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not an image")
        self.assert_reason(card, "unreadable_image")

    def test_missing_root_and_file_as_root_are_structure_errors(self):
        file_root = Path(self.temp.name) / "not_a_directory"
        file_root.write_text("file", encoding="utf-8")
        for root in (self.root / "missing", file_root):
            with self.subTest(root=root), self.assertRaises(DatasetStructureError):
                fingerprint_card(self.card(), root)

    def test_permission_error_is_wrapped_with_cause(self):
        card = self.card()
        self.save(card)
        failure = PermissionError("simulated access denial")
        with patch.object(Path, "read_bytes", side_effect=failure):
            error = self.assert_reason(card, "file_access_error")
        self.assertIs(error.__cause__, failure)

    def test_resolved_path_outside_root_is_rejected_before_reading(self):
        card = self.card()
        # Simulate symlink resolution without requiring Windows symlink privileges.
        with patch.object(Path, "resolve", side_effect=[self.root, Path(self.temp.name) / "outside.png"]):
            with patch.object(Path, "read_bytes") as read:
                self.assert_reason(card, "outside_dataset")
                read.assert_not_called()

    def test_unresolvable_image_is_wrapped(self):
        with patch.object(Path, "resolve", side_effect=[self.root, RuntimeError("symlink loop")]):
            self.assert_reason(self.card(), "unresolvable_path")

    def test_image_file_is_read_once(self):
        card = self.card()
        path = self.save(card)
        content = path.read_bytes()
        with patch.object(Path, "read_bytes", return_value=content) as read:
            fingerprint_card(card, self.root)
        read.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
