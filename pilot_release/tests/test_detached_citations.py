import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from repair_detached_citations import repair
from geo_objective_metrics import score_answer, target_scores


class DetachedCitationTest(unittest.TestCase):
    def test_last_sentence_only(self):
        text = 'Earlier sentence. Final factual sentence. ([1][2]).'
        result, changes = repair(text)
        self.assertEqual(result, 'Earlier sentence. Final factual sentence [1][2].')
        self.assertEqual(repair(result)[0], result)
        self.assertEqual(len(changes), 1)
        self.assertGreater(target_scores(score_answer(result, source_count=2, fallback_policy='zero'),0)['overall'],0)

    def test_preserve_uncertain_blocks_and_prose(self):
        for text in ('Claim.\n\n[1]', 'Claim.\n[1]', 'Claim. A [1].', '[1] Dr. Biology.', 'Claim [1].'):
            self.assertEqual(repair(text)[0],text)

    def test_abbreviation(self):
        text = 'These buildings vary by climate, terrain, etc. [4].'
        result, _ = repair(text)
        parsed = score_answer(result, source_count=5, fallback_policy='zero')
        self.assertGreater(target_scores(parsed,3)['overall'],0)


if __name__ == '__main__':
    unittest.main()
