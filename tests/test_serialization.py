"""Test portable card serialization and CSV export contracts."""
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, INDEX_FIELDS
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.serialization import (
    serialize_card, export_filtered_images, export_excluded_images, write_json,
)


class SerializationTests(unittest.TestCase):
    """Use temporary outputs; no original dataset is read or modified."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def card(self, suit=CardSuit.HEARTS, split=DatasetSplit.TRAIN):
        label = f"ace of {suit.value}" if suit is not None else "joker"
        path = f"{split.value}/{label}/001.jpg"
        return CardImage(path, path, split, label, "ace" if suit else "xxx", suit)

    def read(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_card_serialization_matches_shared_schema(self):
        card = self.card(split=DatasetSplit.TEST)
        row = serialize_card(card)
        self.assertEqual(tuple(row), INDEX_FIELDS)
        self.assertTrue(all(type(value) is str for value in row.values()))
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["suit"], "hearts")
        row["path"] = "changed"
        self.assertNotEqual(card.path, row["path"])

    def test_joker_serializes_as_empty_suit(self):
        row = serialize_card(self.card(None))
        self.assertEqual(row["suit"], "")
        self.assertEqual(row["rank"], "xxx")

    def test_included_export_round_trips_preserving_order_and_splits(self):
        cards = [self.card(CardSuit.SPADES, DatasetSplit.TEST), self.card()]
        path = self.root / "filtered.csv"
        export_filtered_images(cards, path)
        self.assertEqual(load_validate_index(path), cards)
        self.assertEqual(self.read(path)[0], list(INDEX_FIELDS))

    def test_excluded_export_records_correct_reasons(self):
        cards = [self.card(None), self.card(CardSuit.DIAMONDS, DatasetSplit.VALID)]
        path = self.root / "excluded.csv"
        export_excluded_images(cards, path)
        fields, rows = self.read(path)
        self.assertEqual(fields, [*INDEX_FIELDS, "exclusion_reason"])
        self.assertEqual([r["exclusion_reason"] for r in rows], ["joker", "excluded_suit"])
        self.assertEqual([r["image_id"] for r in rows], [c.image_id for c in cards])
        self.assertEqual(rows[1]["split"], "valid")

    def test_all_allowed_suits_rejected_before_exclusion_file_creation(self):
        for suit in (CardSuit.HEARTS, CardSuit.CLUBS, CardSuit.SPADES):
            with self.subTest(suit=suit):
                path = self.root / "excluded.csv"
                with self.assertRaisesRegex(ValueError, "allowed card"):
                    export_excluded_images([self.card(None), self.card(suit)], path)
                self.assertFalse(path.exists())

    def test_empty_exports_retain_headers(self):
        for exporter, fields in (
            (export_filtered_images, INDEX_FIELDS),
            (export_excluded_images, (*INDEX_FIELDS, "exclusion_reason")),
        ):
            with self.subTest(exporter=exporter.__name__):
                path = self.root / exporter.__name__
                exporter([], path)
                self.assertEqual(self.read(path), (list(fields), []))

    def test_csv_exports_do_not_overwrite_existing_files(self):
        for exporter in (export_filtered_images, export_excluded_images):
            with self.subTest(exporter=exporter.__name__):
                path = self.root / exporter.__name__
                path.write_bytes(b"previous output")
                with self.assertRaises(FileExistsError):
                    exporter([], path)
                self.assertEqual(path.read_bytes(), b"previous output")

    def test_csv_parent_must_already_exist(self):
        for exporter in (export_filtered_images, export_excluded_images):
            with self.assertRaises(FileNotFoundError):
                exporter([], self.root / "missing" / "index.csv")

    def test_json_creates_parents_and_preserves_unicode(self):
        path = self.root / "nested" / "summary.json"
        report = {"description": "qualità", "count": 3}
        write_json(report, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
        self.assertIn("qualità", path.read_text(encoding="utf-8"))
        write_json({"count": 4}, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"count": 4})


if __name__ == "__main__":
    unittest.main()
