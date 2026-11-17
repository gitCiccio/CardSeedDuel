"""Integration tests for pair CSV generation and source/run provenance."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from card_suit_duel.data.mapping.definitions import DatasetSplit, INDEX_FIELDS, PAIR_FIELDS
from card_suit_duel.data.pairs.pairs_pipeline import run_pair_generation
from card_suit_duel.data.serialization import build_pair_split_summary, export_pairs
from card_suit_duel.exceptions import MetadataError
from card_suit_duel.game.oracle import determine_outcome
from card_suit_duel.data.mapping.definitions import CardSuit


class PairPipelineTests(unittest.TestCase):
    """Use approved synthetic indices; original image files are not required."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "approved"
        self.source.mkdir()
        self.index = self.source / "deduplicated_images.csv"
        self.output = self.root / "pair_runs"
        self.counts = {s: 6 for s in DatasetSplit}
        self.seeds = {s: 40 + i for i, s in enumerate(DatasetSplit)}
        self.cards = {}
        with self.index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            for split in DatasetSplit:
                for suit in ("hearts", "clubs", "spades"):
                    for i in range(4):
                        label = f"ace of {suit}"
                        path = f"{split.value}/{label}/{i}.jpg"
                        row = dict(zip(INDEX_FIELDS, (path, path, split.value, label, "ace", suit)))
                        writer.writerow(row)
                        self.cards[path] = row
        self.summary = {
            "pipeline": "exact_image_deduplication", "run_id": "approved_run",
            "ready_for_next_phase": True, "retained_count": 36,
            "retained_index": self.index.name,
        }
        self.write_source_summary(self.summary)

    def write_source_summary(self, value):
        (self.source / "summary.json").write_text(json.dumps(value), encoding="utf-8")

    def run_pipeline(self, *, counts=None, seeds=None):
        return run_pair_generation(
            self.index, self.output,
            pairs_per_outcome=self.counts if counts is None else counts,
            seeds=self.seeds if seeds is None else seeds,
        )

    def read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_complete_run_exports_correct_labels_counts_and_provenance(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        output = self.run_pipeline()
        self.assertEqual({p.name for p in output.iterdir()}, {"train_pairs.csv", "valid_pairs.csv", "test_pairs.csv", "summary.json"})
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["run_id"], output.name)
        self.assertEqual(summary["source_run_id"], "approved_run")
        self.assertEqual(summary["source_sha256"], hashlib.sha256(before[self.index.name]).hexdigest())
        self.assertEqual(summary["source_image_count"], 36)
        image_sets = []
        for split in DatasetSplit:
            fields, rows = self.read_csv(output / f"{split.value}_pairs.csv")
            self.assertEqual(fields, list(PAIR_FIELDS))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({r["pair_id"] for r in rows}), 18)
            used = set()
            for row in rows:
                a, b = self.cards[row["first_image_id"]], self.cards[row["second_image_id"]]
                self.assertEqual(a["split"], split.value)
                self.assertEqual(b["split"], split.value)
                self.assertEqual(row["split"], split.value)
                self.assertNotEqual(a["image_id"], b["image_id"])
                self.assertEqual(row["outcome"], determine_outcome(CardSuit(a["suit"]), CardSuit(b["suit"])).value)
                used.update((a["image_id"], b["image_id"]))
            details = summary["splits"][split.value]
            self.assertEqual(details["pairs_by_outcome"], {"p1_win": 6, "p2_win": 6, "draw": 6})
            self.assertEqual(details["unique_pair_count"], 18)
            self.assertEqual(details["participating_image_count"], len(used))
            self.assertEqual(len(details["pairs_by_suit_combination"]), 9)
            self.assertEqual(sum(r["pair_count"] for r in details["pairs_by_suit_combination"]), 18)
            self.assertEqual(details["seed"], self.seeds[split])
            image_sets.append(used)
        self.assertEqual(len(set.union(*image_sets)), sum(map(len, image_sets)))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_repeated_runs_have_distinct_directories_and_identical_csvs(self):
        first = self.run_pipeline()
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        second = self.run_pipeline()
        self.assertNotEqual(first, second)
        for split in DatasetSplit:
            filename = f"{split.value}_pairs.csv"
            self.assertEqual(before[filename], (second / filename).read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})

    def test_changing_train_request_preserves_validation_and_test_csvs(self):
        first = self.run_pipeline()
        counts = {**self.counts, DatasetSplit.TRAIN: 9}
        second = self.run_pipeline(counts=counts)
        for split in ("valid", "test"):
            filename = f"{split}_pairs.csv"
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_unapproved_wrong_pipeline_wrong_index_and_wrong_count_are_rejected(self):
        for change in (
            {"ready_for_next_phase": False}, {"pipeline": "dataset_validation"},
            {"retained_index": "another.csv"}, {"retained_count": 35},
        ):
            with self.subTest(change=change):
                self.write_source_summary({**self.summary, **change})
                with self.assertRaises(MetadataError):
                    self.run_pipeline()
                self.assertFalse(self.output.exists())

    def test_missing_malformed_and_non_object_summaries_are_rejected(self):
        path = self.source / "summary.json"
        for content in (None, "not json", "[]"):
            with self.subTest(content=content):
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(content, encoding="utf-8")
                with self.assertRaises(MetadataError):
                    self.run_pipeline()
                self.assertFalse(self.output.exists())

    def test_missing_configuration_split_is_rejected(self):
        with self.assertRaises(MetadataError):
            self.run_pipeline(counts={DatasetSplit.TRAIN: 6})
        with self.assertRaises(MetadataError):
            self.run_pipeline(seeds={DatasetSplit.TRAIN: 42})
        self.assertFalse(self.output.exists())

    def test_invalid_seed_and_count_do_not_create_outputs(self):
        with self.assertRaises(MetadataError):
            self.run_pipeline(counts={**self.counts, DatasetSplit.TEST: 0})
        with self.assertRaises(MetadataError):
            self.run_pipeline(seeds={**self.seeds, DatasetSplit.TEST: True})
        self.assertFalse(self.output.exists())

    def test_insufficient_capacity_aborts_before_output_creation(self):
        with self.assertRaisesRegex(MetadataError, "Insufficient pair capacity"):
            self.run_pipeline(counts={s: 1000 for s in DatasetSplit})
        self.assertFalse(self.output.exists())

    def test_source_changed_during_loading_is_rejected(self):
        target = "card_suit_duel.data.pairs.pairs_pipeline.load_validate_index"
        from card_suit_duel.data.filter.filtering import load_validate_index
        def modify_after_loading(path):
            cards = load_validate_index(path)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            return cards
        with patch(target, side_effect=modify_after_loading):
            with self.assertRaisesRegex(MetadataError, "changed during loading"):
                self.run_pipeline()
        self.assertFalse(self.output.exists())

    def test_repeated_pairs_from_generator_are_rejected(self):
        target = "card_suit_duel.data.pairs.pairs_pipeline.generate_pairs_for_split"
        from card_suit_duel.data.pairs.pairs_generator import generate_pairs_for_split
        def duplicate(*args, **kwargs):
            pairs = generate_pairs_for_split(*args, **kwargs)
            return (*pairs, pairs[0])
        with patch(target, side_effect=duplicate):
            with self.assertRaisesRegex(MetadataError, "Repeated ordered pairs"):
                self.run_pipeline()
        self.assertFalse(self.output.exists())

    def test_shared_images_from_generator_are_rejected(self):
        from card_suit_duel.data.filter.filtering import load_validate_index
        from card_suit_duel.data.pairs.pairs_generator import generate_pairs_for_split
        pairs = generate_pairs_for_split(load_validate_index(self.index), split=DatasetSplit.TRAIN, pairs_per_outcome=6, seed=42)
        target = "card_suit_duel.data.pairs.pairs_pipeline.generate_pairs_for_split"
        with patch(target, return_value=pairs):
            with self.assertRaisesRegex(MetadataError, "share image identifiers"):
                self.run_pipeline()
        self.assertFalse(self.output.exists())

    def test_export_error_does_not_write_success_summary(self):
        target = "card_suit_duel.data.pairs.pairs_pipeline.export_pairs"
        with patch(target, side_effect=OSError("simulated export failure")):
            with self.assertRaises(OSError):
                self.run_pipeline()
        self.assertEqual(list(self.output.rglob("summary.json")), [])

    def test_empty_pair_summary_contains_all_zero_categories(self):
        summary = build_pair_split_summary([])
        self.assertEqual(summary["pair_count"], 0)
        self.assertEqual(summary["participating_image_count"], 0)
        self.assertEqual(summary["pairs_by_outcome"], {"p1_win": 0, "p2_win": 0, "draw": 0})
        self.assertEqual(len(summary["pairs_by_suit_combination"]), 9)
        self.assertTrue(all(row["pair_count"] == 0 for row in summary["pairs_by_suit_combination"]))

    def test_pair_export_preserves_existing_files(self):
        path = self.root / "previous.csv"
        path.write_bytes(b"previous")
        with self.assertRaises(FileExistsError):
            export_pairs([], path)
        self.assertEqual(path.read_bytes(), b"previous")


if __name__ == "__main__":
    unittest.main()
