"""Verify the deterministic RGB-to-tensor preprocessing contract."""
import unittest

try:
    import torch
    from card_suit_duel.data.preprocessing import build_image_transform
except ModuleNotFoundError as error:
    if error.name not in {"torch", "torchvision"}:
        raise
    torch = None

from PIL import Image
from card_suit_duel.exceptions import MetadataError


@unittest.skipIf(torch is None, "Install the project's neural extra to test preprocessing")
class ImagePreprocessingTests(unittest.TestCase):
    """Use in-memory images; source dataset files are never modified."""

    def test_native_dimensions_channels_and_dtype(self):
        with Image.new("RGB", (11, 7), (10, 20, 30)) as image:
            tensor = build_image_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, 7, 11))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item())

    def test_channel_order_and_eight_bit_scaling(self):
        with Image.new("RGB", (2, 3), (0, 127, 255)) as image:
            tensor = build_image_transform()(image)
        torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([0.0, 127 / 255, 1.0]))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_black_and_white_reach_expected_endpoints(self):
        for color, expected in (((0, 0, 0), 0.0), ((255, 255, 255), 1.0)):
            with self.subTest(color=color), Image.new("RGB", (3, 4), color) as image:
                tensor = build_image_transform()(image)
                torch.testing.assert_close(tensor, torch.full((3, 4, 3), expected))

    def test_explicit_size_uses_height_width_order(self):
        with Image.new("RGB", (11, 7), (10, 20, 30)) as image:
            tensor = build_image_transform(image_size=(5, 9))(image)
        self.assertEqual(tuple(tensor.shape), (3, 5, 9))

    def test_repeated_calls_are_deterministic(self):
        with Image.new("RGB", (11, 7), (10, 20, 30)) as image:
            image.putpixel((0, 0), (255, 0, 127))
            transform = build_image_transform(image_size=(5, 9))
            first, second = transform(image), transform(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_transform_does_not_modify_original_image(self):
        with Image.new("RGB", (11, 7), (10, 20, 30)) as image:
            before = image.tobytes()
            build_image_transform(image_size=(5, 9))(image)
            self.assertEqual(image.size, (11, 7))
            self.assertEqual(image.tobytes(), before)

    def test_tensor_remains_valid_after_image_is_closed(self):
        with Image.new("RGB", (2, 2), (255, 0, 0)) as image:
            tensor = build_image_transform()(image)
        torch.testing.assert_close(tensor[:, 1, 1], torch.tensor([1.0, 0.0, 0.0]))

    def test_invalid_size_shapes_are_rejected(self):
        for size in (224, [224, 224], (), (224,), (224, 224, 224), "224"):
            with self.subTest(size=size), self.assertRaises(MetadataError):
                build_image_transform(image_size=size)

    def test_invalid_dimension_values_are_rejected(self):
        for size in ((0, 224), (224, -1), (True, 224), (224, False), (1.5, 224), (224, "224"), (None, 224)):
            with self.subTest(size=size), self.assertRaises(MetadataError):
                build_image_transform(image_size=size)


if __name__ == "__main__":
    unittest.main()
