"""Exercise validated-index loading without requiring dataset images."""

import csv
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import INDEX_FIELDS, CardSuit, DatasetSplit
from card_suit_duel.exceptions import MetadataError


class ValidatedIndexTests(unittest.TestCase):
    """Check reconstruction, strict input rejection and read-only behavior."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "validated_images.csv"

    def row(self, suit="hearts", split="train"):
        label = f"ace of {suit}" if suit else "joker"
        path = f"{split}/{label}/001.jpg"
        return dict(zip(INDEX_FIELDS, (
            path, path, split, label, "ace" if suit else "xxx", suit,
        )))

    def write(self, rows, fields=INDEX_FIELDS):
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_preserves_order_splits_diamonds_and_joker_without_image_access(self):
        rows = [self.row("diamonds", "test"), self.row("", "valid"), self.row()]
        self.write(rows)
        before = self.path.read_bytes()
        cards = load_validate_index(self.path)
        self.assertEqual([card.image_id for card in cards], [r["image_id"] for r in rows])
        self.assertEqual(cards[0].suit, CardSuit.DIAMONDS)
        self.assertEqual(cards[1].split, DatasetSplit.VALID)
        self.assertIsNone(cards[1].suit)
        self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_identifier_reports_record_number(self):
        self.write([self.row(), self.row()])
        with self.assertRaisesRegex(MetadataError, "record 3: Repeated image_id"):
            load_validate_index(self.path)

    def test_invalid_metadata_is_not_silently_skipped(self):
        for field, value in (
            ("split", "validation"), ("suit", "unknown"),
            ("suit", "clubs"), ("suit", ""), ("rank", ""),
            ("path", " train/ace of hearts/001.jpg"),
            ("image_id", "different"),
        ):
            with self.subTest(field=field, value=value):
                row = self.row()
                row[field] = value
                self.write([row])
                with self.assertRaisesRegex(MetadataError, "record 2:"):
                    load_validate_index(self.path)

    def test_missing_and_duplicate_columns(self):
        for fields in (INDEX_FIELDS[:-1], (*INDEX_FIELDS, "path")):
            with self.subTest(fields=fields):
                self.write([], fields)
                with self.assertRaises(MetadataError):
                    load_validate_index(self.path)

    def test_extra_and_missing_record_fields(self):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(INDEX_FIELDS)
        header = buffer.getvalue()
        values = list(self.row().values())
        for record in (values[:-1], [*values, "extra"]):
            with self.subTest(record=record):
                buffer = io.StringIO(newline="")
                csv.writer(buffer).writerow(record)
                self.path.write_text(header + buffer.getvalue(), encoding="utf-8")
                with self.assertRaisesRegex(MetadataError, "record 2:.*header"):
                    load_validate_index(self.path)

    def test_empty_file_and_header_only_are_rejected(self):
        for text in ("", ",".join(INDEX_FIELDS) + "\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(MetadataError):
                    load_validate_index(self.path)

    def test_missing_file_and_invalid_utf8_are_domain_errors(self):
        with self.assertRaises(MetadataError):
            load_validate_index(self.path)
        self.path.write_bytes(b"\xff")
        with self.assertRaises(MetadataError):
            load_validate_index(self.path)

    def test_malformed_quoted_csv_is_rejected(self):
        self.path.write_text(
            ",".join(INDEX_FIELDS) + '\n"unterminated', encoding="utf-8"
        )
        with self.assertRaises(MetadataError):
            load_validate_index(self.path)

    def test_additional_named_columns_are_allowed(self):
        row = self.row()
        row["note"] = "original metadata"
        self.write([row], (*INDEX_FIELDS, "note"))
        self.assertEqual(len(load_validate_index(self.path)), 1)


if __name__ == "__main__":
    unittest.main()
