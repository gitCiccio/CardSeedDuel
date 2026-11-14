"""End-to-end tests of the deduplication CLI with temporary RGB images."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image, PngImagePlugin

from card_suit_duel.data.mapping.definitions import INDEX_FIELDS


class DeduplicationScriptTests(unittest.TestCase):
    """Exercise the real entry point without touching the original dataset."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.root / "filtered.csv"
        self.output = self.root / "results"

    def prepare(self, entries):
        with self.index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(INDEX_FIELDS)
            for number, (split, suit, color) in enumerate(entries):
                label = f"ace of {suit}"
                relative = f"{split}/{label}/{number}.png"
                path = self.raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                info = PngImagePlugin.PngInfo()
                info.add_text("copy", str(number))
                with Image.new("RGB", (8, 8), color) as image:
                    image.save(path, pnginfo=info)
                writer.writerow((relative, relative, split, label, "ace", suit))

    def run_script(self):
        project = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project / "src")
        return subprocess.run(
            [sys.executable, "-B", str(project / "scripts/deduplication_dataset.py"),
             "--index", str(self.index), "--dataset-root", str(self.raw),
             "--output-root", str(self.output)],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=30,
        )

    def read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def completed_run(self, code=0):
        process = self.run_script()
        self.assertEqual(process.returncode, code, process.stdout + process.stderr)
        self.assertNotIn("Traceback", process.stdout + process.stderr)
        directory = next(self.output.iterdir())
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        return directory, summary

    def test_cross_split_pixel_duplicates_export_consistent_decisions(self):
        self.prepare([("train", "hearts", "red"), ("test", "hearts", "red"), ("valid", "clubs", "blue")])
        original = self.index.read_bytes()
        image_bytes = {p: p.read_bytes() for p in self.raw.rglob("*.png")}
        directory, summary = self.completed_run()
        self.assertEqual({p.name for p in directory.iterdir()}, {
            "deduplicated_images.csv", "removed_images.csv", "duplicate_groups.json",
            "review_groups.json", "summary.json", "execution.log",
        })
        self.assertEqual((summary["input_count"], summary["retained_count"], summary["removed_count"], summary["review_count"]), (3, 2, 1, 0))
        self.assertTrue(summary["ready_for_next_phase"])
        self.assertEqual(summary["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(summary["run_id"], directory.name)
        self.assertEqual(summary["cross_split_group_count"], 1)
        retained = self.read_csv(directory / "deduplicated_images.csv")
        removed = self.read_csv(directory / "removed_images.csv")
        self.assertEqual([r["split"] for r in retained], ["test", "valid"])
        self.assertEqual(removed[0]["representative_id"], retained[0]["image_id"])
        self.assertEqual(removed[0]["removal_reason"], "exact_pixel_duplicate")
        groups = json.loads((directory / "duplicate_groups.json").read_text())
        self.assertEqual({m["decision"] for m in groups[0]["members"]}, {"retained", "removed"})
        self.assertEqual(len({m["file_sha256"] for m in groups[0]["members"]}), 2)
        self.assertEqual(self.index.read_bytes(), original)
        self.assertEqual(image_bytes, {p: p.read_bytes() for p in self.raw.rglob("*.png")})
        log = (directory / "execution.log").read_text(encoding="utf-8")
        self.assertIn("completed successfully", log)
        self.assertNotIn(str(self.root), log)
        self.assertNotIn("\x1b[", log)

    def test_conflicting_labels_return_review_code_and_provisional_index(self):
        self.prepare([("train", "hearts", "red"), ("test", "clubs", "red")])
        directory, summary = self.completed_run(code=2)
        self.assertEqual((summary["retained_count"], summary["removed_count"], summary["review_count"]), (0, 0, 2))
        self.assertFalse(summary["ready_for_next_phase"])
        self.assertEqual(self.read_csv(directory / "deduplicated_images.csv"), [])
        self.assertEqual(self.read_csv(directory / "removed_images.csv"), [])
        review = json.loads((directory / "review_groups.json").read_text())
        self.assertEqual(len(review), 1)
        self.assertIsNone(review[0]["representative_id"])
        self.assertTrue(all(m["decision"] == "review" for m in review[0]["members"]))

    def test_no_duplicates_keeps_all_cards_and_empty_reports(self):
        self.prepare([("train", "hearts", "red"), ("valid", "clubs", "blue")])
        directory, summary = self.completed_run()
        self.assertEqual(summary["retained_count"], 2)
        self.assertEqual(summary["duplicate_group_count"], 0)
        self.assertEqual(summary["retained_by_split"], {"train": 1, "valid": 1, "test": 0})
        self.assertEqual(json.loads((directory / "duplicate_groups.json").read_text()), [])
        self.assertEqual(json.loads((directory / "review_groups.json").read_text()), [])
        self.assertEqual(self.read_csv(directory / "removed_images.csv"), [])

    def test_multiple_runs_preserve_previous_artifacts(self):
        self.prepare([("train", "hearts", "red")])
        first, _ = self.completed_run()
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        process = self.run_script()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})

    def test_missing_index_returns_error_without_output(self):
        process = self.run_script()
        self.assertEqual(process.returncode, 1)
        self.assertIn("Deduplication failed", process.stdout + process.stderr)
        self.assertNotIn("Traceback", process.stdout + process.stderr)
        self.assertFalse(self.output.exists())

    def test_disallowed_suit_is_rejected(self):
        self.prepare([("train", "diamonds", "red")])
        process = self.run_script()
        self.assertEqual(process.returncode, 1)
        self.assertIn("outside the game suits", process.stdout + process.stderr)
        self.assertFalse(self.output.exists())

    def test_corrupt_image_stops_without_exporting_partial_dataset(self):
        self.prepare([("train", "hearts", "red")])
        next(self.raw.rglob("*.png")).write_bytes(b"broken image")
        process = self.run_script()
        self.assertEqual(process.returncode, 1)
        self.assertIn("Deduplication failed", process.stdout + process.stderr)
        self.assertNotIn("Traceback", process.stdout + process.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
