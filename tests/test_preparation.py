"""Exercise orchestration and review replay using a synthetic raw dataset."""
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from card_suit_duel.data.preparation import run_preparation
from card_suit_duel.data.deduplication.deduplication import deduplicate_cards
from card_suit_duel.data.deduplication.fingerprints import ImageFingerprint
from card_suit_duel.data.deduplication.manual_review import apply_manual_decisions
from card_suit_duel.data.mapping.definitions import CardSuit, DatasetSplit, REQUIRED_FIELDS
from card_suit_duel.data.mapping.schema import CardImage
from card_suit_duel.exceptions import MetadataError


class ManualReviewReplayTests(unittest.TestCase):
    def setUp(self):
        self.cards = []
        for suit in (CardSuit.CLUBS, CardSuit.SPADES):
            label = f"four of {suit.value}"; path = f"train/{label}/a.png"
            self.cards.append(CardImage(path,path,DatasetSplit.TRAIN,label,"four",suit))
        self.hash = "a" * 64
        self.result = deduplicate_cards(self.cards,[ImageFingerprint(c.image_id,"b"*64,self.hash) for c in self.cards])
        self.decision = {"pixels_sha256":self.hash,"representative_id":self.cards[1].image_id,"removed_id":self.cards[0].image_id,"reason":"visually verified spades"}

    def test_replay_keeps_correct_reference_and_resolves_group(self):
        result = apply_manual_decisions(self.cards,self.result,[self.decision])
        self.assertEqual(result.retained,(self.cards[1],))
        self.assertEqual(result.removed,(self.cards[0],))
        self.assertFalse(result.review)
        self.assertFalse(result.groups[0].requires_review)
        self.assertEqual(len(self.result.review),2)

    def test_unknown_repeated_or_wrong_member_decisions_are_rejected(self):
        for decisions in (
            [{**self.decision,"pixels_sha256":"c"*64}],
            [self.decision,self.decision],
            [{**self.decision,"representative_id":"unknown"}],
            [{**self.decision,"reason":""}],
            [{**self.decision,"removed_id":"unknown"}],
        ):
            with self.subTest(decisions=decisions),self.assertRaises(MetadataError):
                apply_manual_decisions(self.cards,self.result,decisions)

    def test_no_decision_preserves_unresolved_review(self):
        self.assertEqual(apply_manual_decisions(self.cards,self.result,[]),self.result)


class PreparationTests(unittest.TestCase):
    """Fresh runs must reproduce CSVs while preserving source files."""
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);raw=self.root/'data/raw';raw.mkdir(parents=True)
        (self.root/'configs').mkdir()
        (self.root/'configs/review.json').write_text('[]',encoding='utf-8')
        rows=[];number=0
        for split in DatasetSplit:
            for class_id,suit in enumerate(("hearts","clubs","spades")):
                label=f'ace of {suit}'
                for i in range(4):
                    relative=f'{split.value}/{label}/{i}.png';path=raw/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    with Image.new('RGB',(8,8),(number,number+50,number+100)) as image:image.save(path)
                    rows.append(dict(zip(REQUIRED_FIELDS,(str(class_id),relative,label,'ace',split.value))));number+=1
        with (raw/'cards.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=REQUIRED_FIELDS);writer.writeheader();writer.writerows(rows)
        self.config={'schema_version':1,'dataset_root':'data/raw','metadata':'data/raw/cards.csv','output_root':'results/preparation','manual_decisions':'configs/review.json','pairs_per_outcome':{'train':30,'valid':3,'test':3},'pair_seeds':{'train':42,'valid':43,'test':44},'subset_seed':123}
        self.path=self.root/'configs/preparation.json'
        self.write_config()

    def write_config(self):self.path.write_text(json.dumps(self.config),encoding='utf-8')

    def test_end_to_end_reproduces_content_and_preserves_raw(self):
        raw=self.root/'data/raw'
        before={p.relative_to(raw).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()}
        first=run_preparation(self.root,self.path);second=run_preparation(self.root,self.path)
        self.assertNotEqual(first,second)
        self.assertEqual((first/'reproducibility.json').read_bytes(),(second/'reproducibility.json').read_bytes())
        summary=json.loads((first/'summary.json').read_text())
        self.assertTrue(summary['data_preparation_complete'])
        self.assertFalse(summary['model_integration_verified'])
        self.assertFalse(summary['companion_clone_verified'])
        for key,path in summary['indices'].items():
            self.assertTrue((first/path).is_file(),key)
        self.assertEqual(before,{p.relative_to(raw).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()})

    def test_output_inside_raw_is_rejected(self):
        self.config['output_root']='data/raw/output';self.write_config()
        with self.assertRaises(MetadataError):run_preparation(self.root,self.path)

    def test_paths_outside_project_are_rejected(self):
        self.config['metadata']='../outside.csv';self.write_config()
        with self.assertRaises(MetadataError):run_preparation(self.root,self.path)

    def test_invalid_config_version_is_rejected(self):
        self.config['schema_version']=99;self.write_config()
        with self.assertRaises(MetadataError):run_preparation(self.root,self.path)


if __name__=='__main__':unittest.main()
