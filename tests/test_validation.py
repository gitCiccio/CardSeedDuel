"""Validate F2.3 on temporary CSVs and images, never on the original dataset.

Run from the repository root after installing the package:
    python -m unittest discover -s tests -p "test_validation.py" -v
"""
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from card_suit_duel.data.mapping.definitions import REQUIRED_FIELDS, ValidationStatus
from card_suit_duel.data.validation.validation import parse_card_row, validate_card_file
from card_suit_duel.data.validation.validation_result import (
    RowValidationResult, ValidationIssue,
)
from card_suit_duel.data.validation.validation_suite import validate_dataset, export_validation
from card_suit_duel.data.validation.sub_validation.serialization import serialize_row
from card_suit_duel.exceptions import DatasetStructureError, ImageValidationError, MetadataError


def sample_row(path="train/ace of hearts/a.jpg", label="ace of hearts",
               rank="ace", index="0", split="train"):
    """Build explicit original-format metadata for a test case."""
    return dict(zip(REQUIRED_FIELDS, (index, path, label, rank, split)))


class ValidationTests(unittest.TestCase):
    """Exercise public row/file/pipeline APIs and exported artifacts."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "data/raw"
        for split in ("train", "valid", "test"):
            (self.root / split).mkdir(parents=True)
        self.metadata = self.root / "cards.csv"

    def image(self, relative="train/ace of hearts/a.jpg"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (8, 8)) as image:
            image.save(path, format="JPEG")
        return path

    def run_rows(self, rows):
        with self.metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return validate_dataset(self.root, self.metadata)

    def test_parse_normal_card_diamonds_and_joker(self):
        for suit in ("hearts", "clubs", "spades", "diamonds"):
            with self.subTest(suit=suit):
                label = f"ace of {suit}"
                card = parse_card_row(sample_row(path=f"train/{label}/a.jpg", label=label))
                self.assertEqual(card.suit.value, suit)
        joker = parse_card_row(sample_row(path="train/joker/a.jpg", label="joker", rank="xxx", index="20"))
        self.assertIsNone(joker.suit)

    def test_invalid_metadata_is_rejected(self):
        cases = (
            {"class index": "-1"}, {"class index": "1.5"},
            {"data set": "training"}, {"labels": ""},
            {"card type": "unknown"}, {"card type": "king"},
            {"labels": "ace hearts"}, {"labels": "ace of unknown"},
            {"filepaths": "train/ace of clubs/a.jpg"},
            {"filepaths": "../outside.jpg"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(MetadataError):
                    parse_card_row({**sample_row(), **changes})

    def test_valid_file_is_returned_without_modification(self):
        path = self.image()
        before = path.read_bytes()
        self.assertEqual(validate_card_file(parse_card_row(sample_row()), self.root), path.resolve())
        self.assertEqual(before, path.read_bytes())

    def test_missing_and_corrupt_file_have_distinct_reasons(self):
        card = parse_card_row(sample_row())
        with self.assertRaises(ImageValidationError) as missing:
            validate_card_file(card, self.root)
        self.assertEqual(missing.exception.reason, "missing_file")
        path = self.root / card.path
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a JPEG")
        with self.assertRaises(ImageValidationError) as corrupt:
            validate_card_file(card, self.root)
        self.assertEqual(corrupt.exception.reason, "unreadable_image")
        self.assertIsNotNone(corrupt.exception.__cause__)

    def test_resolved_target_must_remain_inside_dataset(self):
        # Model resolution of an escaping symbolic link without requiring
        # Windows privileges to create a real filesystem link.
        with patch.object(Path, "resolve", side_effect=[self.root, self.project / "outside.jpg"]):
            with self.assertRaises(ImageValidationError) as error:
                validate_card_file(parse_card_row(sample_row()), self.root)
        self.assertEqual(error.exception.reason, "outside_dataset")

    def test_missing_root_or_split_stops_pipeline(self):
        with self.assertRaises(DatasetStructureError):
            validate_dataset(self.root / "absent", self.metadata)
        (self.root / "valid").rmdir()
        with self.assertRaises(DatasetStructureError):
            self.run_rows([])

    def test_missing_or_duplicate_csv_headers_stop_pipeline(self):
        for header in ("labels\n", "labels,labels\n", ""):
            with self.subTest(header=header):
                self.metadata.write_text(header, encoding="utf-8")
                with self.assertRaises(MetadataError):
                    validate_dataset(self.root, self.metadata)

    def test_one_result_per_row_with_original_record_numbers(self):
        self.image()
        result = self.run_rows([sample_row(), sample_row(path="train/ace of hearts/missing.jpg"), {**sample_row(path="train/ace of hearts/invalid.jpg"), "labels": ""}])
        self.assertEqual([row.row_number for row in result.rows], [2, 3, 4])
        self.assertEqual([row.status for row in result.rows], [ValidationStatus.ACCEPTED, ValidationStatus.REJECTED, ValidationStatus.REJECTED])
        self.assertEqual(result.rows[1].issues[0].code, "missing_file")
        self.assertEqual(result.rows[2].issues[0].code, "invalid_metadata")
        self.assertEqual(result.source_sha256, hashlib.sha256(self.metadata.read_bytes()).hexdigest())

    def test_extra_csv_fields_reject_only_the_bad_record(self):
        self.image()
        self.metadata.write_text(
            ",".join(REQUIRED_FIELDS) + "\n0,train/ace of hearts/a.jpg,ace of hearts,ace,train,EXTRA\n",
            encoding="utf-8",
        )
        result = validate_dataset(self.root, self.metadata)
        self.assertEqual(result.rows[0].status, ValidationStatus.REJECTED)

    def test_identical_references_keep_one_and_decode_once(self):
        self.image()
        with patch("card_suit_duel.data.validation.sub_validation.validation_images.validate_card_file", wraps=validate_card_file) as decoder:
            result = self.run_rows([sample_row(), sample_row()])
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(result.rows[0].status, ValidationStatus.ACCEPTED)
        self.assertEqual(result.rows[1].issues[0].code, "duplicate_reference")

    def test_same_path_with_conflicting_metadata_requires_review(self):
        result = self.run_rows([sample_row(), sample_row(index="1")])
        self.assertTrue(all(row.status is ValidationStatus.REVIEW for row in result.rows))
        self.assertTrue(all("path_metadata_conflict" in {issue.code for issue in row.issues} for row in result.rows))

    def test_one_index_with_two_labels_requires_review(self):
        other = sample_row(path="train/ace of clubs/b.jpg", label="ace of clubs")
        result = self.run_rows([sample_row(), other])
        self.assertTrue(all(row.status is ValidationStatus.REVIEW for row in result.rows))
        self.assertTrue(all(row.issues[0].code == "class_index_conflict" for row in result.rows))

    def test_one_label_with_two_indices_requires_review(self):
        other = sample_row(path="train/ace of hearts/b.jpg", index="1")
        result = self.run_rows([sample_row(), other])
        self.assertTrue(all(row.status is ValidationStatus.REVIEW for row in result.rows))

    def test_invalid_conflicting_reference_does_not_allow_silent_acceptance(self):
        result = self.run_rows([sample_row(), {**sample_row(), "labels": "wrong"}])
        self.assertEqual(result.rows[0].status, ValidationStatus.REVIEW)
        self.assertEqual(result.rows[1].status, ValidationStatus.REJECTED)

    def test_unreferenced_image_is_reported_not_added(self):
        self.image("test/ace of hearts/unlisted.jpg")
        result = self.run_rows([])
        self.assertEqual(result.rows, ())
        self.assertEqual(result.unreferenced_images, ("test/ace of hearts/unlisted.jpg",))

    def test_result_objects_require_consistent_status_and_issues(self):
        with self.assertRaises(ValueError):
            RowValidationResult(2, None, ValidationStatus.ACCEPTED)
        with self.assertRaises(ValueError):
            RowValidationResult(2, None, ValidationStatus.REJECTED)
        with self.assertRaises(ValueError):
            ValidationIssue("", "message")

    def test_serialized_rejections_hide_absolute_paths(self):
        result = RowValidationResult(2, "C:/Users/person/private.jpg", ValidationStatus.REJECTED,
                                     issues=(ValidationIssue("invalid_metadata", "Invalid metadata."),))
        record = serialize_row(result)
        self.assertEqual(record["original_path"], "<non-relative path omitted>")
        self.assertNotIn("person", json.dumps(record))

    def test_export_writes_index_reports_and_rejects_overwrite(self):
        image = self.image()
        original = image.read_bytes()
        result = self.run_rows([sample_row(), sample_row(path="train/ace of hearts/output")])
        output = self.project / "results/run1"
        summary = export_validation(result, output)
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)
        self.assertTrue(summary["ready_for_next_phase"])
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertEqual(json.loads((output / "review_rows.json").read_text()), [])
        rejected = json.loads((output / "rejected_rows.json").read_text())
        self.assertEqual(rejected[0]["issues"][0]["code"], "missing_file")
        with (output / "validated_images.csv").open(newline="", encoding="utf-8") as stream:
            cards = list(csv.DictReader(stream))
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["suit"], "hearts")
        self.assertEqual(original, image.read_bytes())
        with self.assertRaises(FileExistsError):
            export_validation(result, output)

    def test_review_and_empty_indices_are_not_ready(self):
        for name, rows in (("empty", []), ("review", [sample_row(), sample_row(index="1")])):
            with self.subTest(name=name):
                result = self.run_rows(rows)
                summary = export_validation(result, self.project / name)
                self.assertFalse(summary["ready_for_next_phase"])

    def test_joker_export_keeps_empty_suit_and_original_rank(self):
        self.image("train/joker/a.jpg")
        row = sample_row(path="train/joker/a.jpg", label="joker", rank="xxx", index="20")
        output = self.project / "joker-run"
        export_validation(self.run_rows([row]), output)
        with (output / "validated_images.csv").open(newline="", encoding="utf-8") as stream:
            record = next(csv.DictReader(stream))
        self.assertEqual(record["suit"], "")
        self.assertEqual(record["rank"], "xxx")


if __name__ == "__main__":
    unittest.main()
