"""Integration tests for filtering outputs, provenance and CLI behavior."""

import csv
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.filter.filtering_pipeline import run_filtering
from card_suit_duel.data.mapping.definitions import INDEX_FIELDS
from card_suit_duel.data.serialization import build_filtering_summary
from card_suit_duel.exceptions import MetadataError


class FilteringPipelineTests(unittest.TestCase):
    """Run on tiny temporary indices without reading original dataset images."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / "validated_images.csv"
        self.output = self.root / "results"
        self.write_index([
            ("hearts", "train"), ("diamonds", "valid"),
            ("clubs", "valid"), ("", "test"), ("spades", "test"),
        ])

    def write_index(self, entries):
        with self.index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(INDEX_FIELDS)
            for suit, split in entries:
                label = f"ace of {suit}" if suit else "joker"
                path = f"{split}/{label}/001.jpg"
                writer.writerow((path, path, split, label, "ace" if suit else "xxx", suit))

    def read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_complete_run_preserves_input_order_splits_and_hash(self):
        original = self.index.read_bytes()
        output = run_filtering(self.index, self.output)
        self.assertEqual(output.parent, self.output)
        self.assertEqual({p.name for p in output.iterdir()}, {
            "filtered_images.csv", "excluded_images.csv", "summary.json",
        })
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_id"], output.name)
        self.assertEqual(summary["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual((summary["input_count"], summary["included_count"], summary["excluded_count"]), (5, 3, 2))
        self.assertEqual(summary["included_by_split"], {"train": 1, "valid": 1, "test": 1})
        self.assertEqual(summary["excluded_by_split"], {"train": 0, "valid": 1, "test": 1})
        self.assertEqual(summary["included_by_suit"], {"hearts": 1, "clubs": 1, "spades": 1})
        self.assertEqual(summary["excluded_by_suit"], {"diamonds": 1, "joker": 1})
        included = self.read_csv(output / summary["included_index"])
        excluded = self.read_csv(output / summary["excluded_index"])
        self.assertEqual([r["suit"] for r in included], ["hearts", "clubs", "spades"])
        self.assertEqual([r["split"] for r in included], ["train", "valid", "test"])
        self.assertEqual([r["exclusion_reason"] for r in excluded], ["excluded_suit", "joker"])
        input_ids = {c.image_id for c in load_validate_index(self.index)}
        included_ids = {r["image_id"] for r in included}
        excluded_ids = {r["image_id"] for r in excluded}
        self.assertFalse(included_ids & excluded_ids)
        self.assertEqual(input_ids, included_ids | excluded_ids)
        self.assertEqual(self.index.read_bytes(), original)

    def test_repeated_runs_keep_previous_outputs(self):
        first = run_filtering(self.index, self.output)
        original = {p.name: p.read_bytes() for p in first.iterdir()}
        second = run_filtering(self.index, self.output)
        self.assertNotEqual(first, second)
        self.assertEqual(original, {p.name: p.read_bytes() for p in first.iterdir()})

    def test_all_included_and_all_excluded_inputs(self):
        for entries, expected in (([("hearts", "train")], (1, 0)), ([("", "test")], (0, 1))):
            with self.subTest(entries=entries):
                self.write_index(entries)
                output = run_filtering(self.index, self.output)
                self.assertEqual(len(self.read_csv(output / "filtered_images.csv")), expected[0])
                self.assertEqual(len(self.read_csv(output / "excluded_images.csv")), expected[1])

    def test_invalid_index_creates_no_output(self):
        self.index.write_text("wrong,header\n1,2\n", encoding="utf-8")
        with self.assertRaises(MetadataError):
            run_filtering(self.index, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_source_creates_no_output(self):
        self.index.unlink()
        with self.assertRaises(OSError):
            run_filtering(self.index, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_source_is_rejected_before_export(self):
        target = "card_suit_duel.data.filter.filtering_pipeline._calculate_source_hash"
        with patch(target, side_effect=["before", "after"]):
            with self.assertRaisesRegex(MetadataError, "changed during loading"):
                run_filtering(self.index, self.output)
        self.assertFalse(self.output.exists())

    def test_inconsistent_partition_count_is_rejected(self):
        target = "card_suit_duel.data.filter.filtering_pipeline.filter_game_cards"
        with patch(target, return_value=([], [])):
            with self.assertRaisesRegex(MetadataError, "counts do not match"):
                run_filtering(self.index, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_run_directory_is_not_overwritten(self):
        existing = self.output / "existing_run"
        existing.mkdir(parents=True)
        sentinel = existing / "summary.json"
        sentinel.write_bytes(b"previous run")
        with patch("card_suit_duel.data.filter.filtering_pipeline._create_run_id", return_value="existing_run"):
            with self.assertRaises(FileExistsError):
                run_filtering(self.index, self.output)
        self.assertEqual(sentinel.read_bytes(), b"previous run")

    def test_export_failure_does_not_produce_success_summary(self):
        target = "card_suit_duel.data.filter.filtering_pipeline.export_excluded_images"
        with patch(target, side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                run_filtering(self.index, self.output)
        self.assertEqual(list(self.output.rglob("summary.json")), [])

    def test_empty_summary_has_zero_split_counts(self):
        summary = build_filtering_summary([], [], run_id="test", source_sha256="a" * 64)
        self.assertEqual(summary["input_count"], 0)
        for key in ("included_by_split", "excluded_by_split"):
            self.assertEqual(summary[key], {"train": 0, "valid": 0, "test": 0})
        self.assertEqual(summary["policy"]["allowed_suits"], ["hearts", "clubs", "spades"])

    def run_cli(self):
        project = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project / "src")
        return subprocess.run(
            [sys.executable, "-B", str(project / "scripts/filter_dataset.py"),
             "--index", str(self.index), "--output-root", str(self.output)],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=30,
        )

    def test_cli_writes_portable_completion_log(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        logs = list(self.output.glob("*/execution.log"))
        self.assertEqual(len(logs), 1)
        content = logs[0].read_text(encoding="utf-8")
        self.assertIn("completed successfully", content)
        self.assertNotIn(str(self.root), content)
        self.assertNotIn("\x1b[", content)

    def test_cli_invalid_input_returns_failure_without_traceback(self):
        self.index.unlink()
        result = self.run_cli()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Filtering failed", result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
