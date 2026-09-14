import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from summarize_literature_overall import aggregate, summarize, ORIGINAL


class SummaryTests(unittest.TestCase):
    def test_seed_means(self):
        obj = {('m','s',ORIGINAL,s): ('b',s/2) for s in (0,1,2)}
        sub = {k:(b,v*5) for k,(b,v) in obj.items()}
        self.assertEqual(aggregate(obj,sub)[('m','b','s',ORIGINAL)],(0.5,2.5))

    def test_missing_seeds_rejected(self):
        obj = {('m','s',ORIGINAL,0):('b',0)}
        with self.assertRaises(ValueError): aggregate(obj,obj)

    def test_missing_keys_rejected(self):
        with self.assertRaises(ValueError): aggregate({('m','s',ORIGINAL,0):('b',0)}, {})

    def test_paired_subset_not_full_original(self):
        units = {('m','b','s1',ORIGINAL):(0,0), ('m','b','s2',ORIGINAL):(1,5),
                 ('m','b','s1','method'):(0.2,1)}
        r = next(r for r in summarize(units) if r['method']=='method')
        self.assertEqual(r['objective_before'],0)
        self.assertEqual(r['subjective_delta'],1)
        self.assertEqual(r['objective_up'],1)


if __name__ == '__main__': unittest.main()
