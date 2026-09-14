import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_necessity_answer_model_shard as runner
from literature_method_runner import LiteratureMethodCatalog


class LiteratureConditionsTest(unittest.TestCase):
    def setUp(self):
        self.sid = "geo_bench__all__1"
        self.manifest = [{"sample_id": self.sid, "target_document_index": 1}]
        self.source = {"benchmark": "geo_bench", "domain": "all", "instance_id": "1", "documents": ["other", "original", "last"]}
        self.rows = [{"sample_id": self.sid, "method_id": m, "target_document_index": 1, "source_text": "original", "rewritten_text": "  rewritten " + m + "\n"} for m in LiteratureMethodCatalog().method_ids]

    def build(self):
        with patch.object(runner, "read_jsonl", side_effect=[self.manifest, self.rows]), patch.object(runner, "load_geo", return_value=[self.source]), patch.object(runner, "load_cseo", return_value=[]):
            return runner.build_literature_conditions(Path("manifest"), Path("rewrites"), Path("geo"), Path("cseo"))

    def test_only_target_changes_and_text_is_byte_preserved(self):
        conditions, diagnostic = self.build()
        self.assertEqual(len(conditions), 8)
        self.assertEqual(conditions[0][2], self.source["documents"])
        for condition, row in zip(conditions[1:], self.rows):
            self.assertEqual(condition[1], row["method_id"])
            self.assertEqual(condition[2], ["other", row["rewritten_text"], "last"])
        self.assertEqual(diagnostic["rewrite_count"], 7)

    def test_duplicate_rejected(self):
        self.rows.append(self.rows[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()

    def test_missing_method_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "set mismatch"):
            self.build()

    def test_declared_failure_skips_only_missing_method(self):
        removed=self.rows.pop()
        with patch.object(runner,'read_jsonl',side_effect=[self.manifest,self.rows,[removed]]), patch.object(runner,'load_geo',return_value=[self.source]), patch.object(runner,'load_cseo',return_value=[]):
            conditions, diagnostic=runner.build_literature_conditions(Path('manifest'),Path('rewrites'),Path('geo'),Path('cseo'),missing_rewrites_path=Path('missing'))
        self.assertEqual(len(conditions),7)
        self.assertEqual(conditions[0][2],self.source['documents'])
        self.assertEqual(diagnostic['declared_missing_rewrite_count'],1)

    def test_declared_failure_cannot_overlap_success(self):
        with patch.object(runner,'read_jsonl',side_effect=[self.manifest,self.rows,[self.rows[0]]]), patch.object(runner,'load_geo',return_value=[self.source]), patch.object(runner,'load_cseo',return_value=[]):
            with self.assertRaisesRegex(ValueError,'conflicting'):
                runner.build_literature_conditions(Path('manifest'),Path('rewrites'),Path('geo'),Path('cseo'),missing_rewrites_path=Path('missing'))

    def test_source_mismatch_rejected(self):
        self.rows[0]["source_text"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source or target mismatch"):
            self.build()

    def test_unexpected_sample_rejected(self):
        self.rows[0]["sample_id"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.build()


if __name__ == "__main__":
    unittest.main()
