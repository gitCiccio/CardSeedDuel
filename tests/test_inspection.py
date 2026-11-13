"""Exercise reusable inspection on temporary data, without touching raw data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from card_suit_duel.data.inspection import (
    compare_metadata_and_files, inspect_images, read_metadata,
    run_inspection, save_report, scan_dataset_files,
)
from card_suit_duel.exceptions import DatasetStructureError, MetadataError


class InspectionTests(unittest.TestCase):
    """Cover anomalies, explicit paths and stable report output."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / 'data/raw'
        for split in ('train', 'valid', 'test'):
            (self.root / split).mkdir(parents=True)

    def image(self, relative, mode='RGB', size=(8, 6), format='JPEG'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size).save(path, format=format)
        return path

    def metadata(self, paths):
        path = self.root / 'cards.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['class index', 'filepaths', 'labels', 'card type', 'data set'])
            for relative in paths:
                writer.writerow([0, relative, 'ace of hearts', 'ace', 'train'])
        return path

    def test_pipeline_reports_missing_reference_and_unlisted_image(self):
        image = self.image('train/ace of hearts/a.jpg')
        self.image('test/ace of hearts/unlisted.jpg')
        metadata = self.metadata(['train/ace of hearts/a.jpg', 'train/ace of hearts/output'])
        original_image = image.read_bytes()
        original_csv = metadata.read_bytes()
        report = run_inspection(self.root, metadata, self.project)
        self.assertEqual(report['metadata']['row_count'], 2)
        self.assertEqual(report['filesystem']['image_file_count'], 2)
        self.assertEqual(report['images']['readable_image_count'], 2)
        self.assertEqual(report['anomalies']['csv_paths_without_scanned_image'],
                         ['train/ace of hearts/output'])
        self.assertEqual(report['anomalies']['scanned_images_without_csv_reference'],
                         ['test/ace of hearts/unlisted.jpg'])
        self.assertEqual(image.read_bytes(), original_image)
        self.assertEqual(metadata.read_bytes(), original_csv)
        self.assertFalse((self.project / 'results').exists())
        output = self.project / 'results/report.json'
        save_report(report, output)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8')), report)
        self.assertEqual(report, run_inspection(self.root, metadata, self.project))

    def test_corrupt_image_is_reported_while_other_images_continue(self):
        self.image('train/ace of hearts/a.jpg')
        bad = self.root / 'train/broken.jpg'
        bad.write_bytes(b'not an image')
        self.image('valid/card.png', mode='L', size=(4, 5), format='PNG')
        summary = inspect_images(scan_dataset_files(self.root), self.root)
        self.assertEqual(summary['inspected_file_count'], 3)
        self.assertEqual(summary['readable_image_count'], 2)
        self.assertEqual(summary['unreadable_image_count'], 1)
        self.assertEqual(summary['unreadable_images'][0]['path'], 'train/broken.jpg')
        self.assertEqual(summary['color_modes'], {'L': 1, 'RGB': 1})
        self.assertEqual(summary['formats'], {'JPEG': 1, 'PNG': 1})
        self.assertEqual(summary['dimensions'], {'4x5': 1, '8x6': 1})

    def test_missing_split_stops_scan(self):
        (self.root / 'valid').rmdir()
        with self.assertRaises(DatasetStructureError):
            scan_dataset_files(self.root)

    def test_metadata_schema_failure_is_explicit(self):
        path = self.root / 'cards.csv'
        path.write_text('labels\nace of hearts\n', encoding='utf-8')
        with self.assertRaises(MetadataError):
            read_metadata(path)

    def test_comparison_normalizes_separators_and_counts_empty_paths(self):
        image = self.image('train/ace of hearts/a.jpg')
        result = compare_metadata_and_files(
            [{'filepaths': 'train\\ace of hearts\\a.jpg'}, {'filepaths': ''}],
            {'train': [image]}, self.root,
        )
        self.assertEqual(result, {
            'csv_paths_without_scanned_image': [],
            'scanned_images_without_csv_reference': [],
            'rows_without_path': 1,
        })

    def test_distinct_dataset_roots_do_not_share_state(self):
        self.image('train/ace of hearts/a.jpg')
        first = scan_dataset_files(self.root)
        other = self.project / 'other'
        for split in ('train', 'valid', 'test'):
            (other / split).mkdir(parents=True)
        second = scan_dataset_files(other)
        self.assertEqual(len(first['train']), 1)
        self.assertEqual(second, {'train': [], 'valid': [], 'test': []})


if __name__ == '__main__':
    unittest.main()
