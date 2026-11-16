"""Test shared path validation and detached RGB image loading."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from card_suit_duel.data.image_io import load_card_image, resolve_card_path as image_io_resolver
from card_suit_duel.data.path_checks import resolve_dataset_root, resolve_card_reference, resolve_card_path
from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import DatasetStructureError, ImageValidationError


class ImageAccessTests(unittest.TestCase):
    """Use temporary files and simulated link resolution on Windows."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "raw"
        self.root.mkdir()
        path = "train/ace of hearts/a.png"
        self.card = CardImage(path, path, DatasetSplit.TRAIN, "ace of hearts", "ace", CardSuit.HEARTS)
        self.path = self.root / path

    def save(self, mode="RGB"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new(mode, (8, 9), 0) as image:
            image.save(self.path)

    def assert_reason(self, function, reason):
        with self.assertRaises(ImageValidationError) as caught:
            function(self.card, self.root)
        self.assertEqual(caught.exception.reason, reason)
        self.assertEqual(caught.exception.image_path, self.card.path)
        return caught.exception

    def test_resolved_root_and_existing_card_path(self):
        self.save()
        self.assertEqual(resolve_dataset_root(self.root), self.root.resolve())
        self.assertEqual(resolve_card_path(self.card, self.root), self.path.resolve())

    def test_image_io_reexports_the_shared_resolver(self):
        self.assertIs(image_io_resolver, resolve_card_path)

    def test_missing_root_and_file_as_root_are_structure_errors(self):
        file = self.root / "file"
        file.write_bytes(b"file")
        for root in (self.root / "missing", file):
            with self.subTest(root=root), self.assertRaises(DatasetStructureError):
                resolve_card_path(self.card, root)

    def test_reference_resolution_does_not_require_image_existence(self):
        self.assertEqual(resolve_card_reference(self.card, self.root), self.path.resolve())
        self.assert_reason(resolve_card_path, "missing_file")

    def test_directory_cannot_be_used_as_image_file(self):
        self.path.mkdir(parents=True)
        self.assert_reason(resolve_card_path, "missing_file")

    def test_outside_target_is_rejected_before_file_access(self):
        with patch.object(Path, "resolve", side_effect=[self.root, Path(self.temp.name) / "outside.png"]):
            with patch.object(Path, "is_file") as access:
                self.assert_reason(resolve_card_path, "outside_dataset")
                access.assert_not_called()

    def test_unresolvable_image_has_explicit_reason(self):
        with patch.object(Path, "resolve", side_effect=[self.root, RuntimeError("loop")]):
            self.assert_reason(resolve_card_path, "unresolvable_path")

    def test_file_access_error_preserves_cause(self):
        error = PermissionError("denied")
        with patch.object(Path, "is_file", side_effect=error):
            caught = self.assert_reason(resolve_card_path, "file_access_error")
        self.assertIs(caught.__cause__, error)

    def test_load_preserves_pixels_dimensions_and_original_file(self):
        self.save()
        before = self.path.read_bytes()
        with load_card_image(self.card, self.root) as image:
            self.assertEqual((image.mode, image.size), ("RGB", (8, 9)))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(before, self.path.read_bytes())

    def test_returned_image_remains_usable_after_source_file_is_removed(self):
        self.save()
        with load_card_image(self.card, self.root) as image:
            self.path.unlink()
            self.assertEqual(image.getpixel((1, 1)), (0, 0, 0))
            self.assertEqual(image.size, (8, 9))

    def test_non_rgb_images_are_rejected_without_conversion(self):
        for mode in ("L", "P", "RGBA"):
            with self.subTest(mode=mode):
                self.save(mode)
                self.assert_reason(load_card_image, "unexpected_color_mode")

    def test_corrupt_image_is_rejected(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"not an image")
        self.assert_reason(load_card_image, "unreadable_image")

    def test_missing_image_is_rejected(self):
        self.assert_reason(load_card_image, "missing_file")

    def test_permission_failure_during_open_is_an_access_error(self):
        self.save()
        with patch("card_suit_duel.data.image_io.Image.open", side_effect=PermissionError("denied")):
            self.assert_reason(load_card_image, "file_access_error")

    def test_image_disappearing_between_check_and_open_is_reported(self):
        self.save()
        with patch("card_suit_duel.data.image_io.Image.open", side_effect=FileNotFoundError("gone")):
            self.assert_reason(load_card_image, "missing_file")


if __name__ == "__main__":
    unittest.main()
