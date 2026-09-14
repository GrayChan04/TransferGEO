import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from normalized_improvement_pilots import improvement, compute, summarize, ORIGINAL


class NormalizedTests(unittest.TestCase):
    def test_zero_baseline(self):
        self.assertEqual(improvement(0,0),0)
        self.assertEqual(improvement(0.2,0),2)

    def test_negative_and_identity(self):
        self.assertEqual(improvement(0,1),-5)
        self.assertEqual(improvement(3,3),0)

    def test_seed_before_averaging(self):
        obj={}
        for s,(before,after) in enumerate([(0,1),(1,0),(1,1)]):
            obj[('m','s',ORIGINAL,s)]=('b',before)
            obj[('m','s','method',s)]=('b',after)
        paired,units=compute(obj,obj)
        r=next(r for r in summarize(units) if r['method']=='method')
        self.assertAlmostEqual(r['objective'],5/3)
        self.assertNotAlmostEqual(r['objective'],improvement(2/3,2/3))
        self.assertEqual(len(paired),6)

    def test_pair_mismatch(self):
        with self.assertRaises(ValueError): compute({('m','s',ORIGINAL,0):('b',0)}, {})

    def test_incomplete_seeds(self):
        d={('m','s',ORIGINAL,0):('b',0)}
        with self.assertRaises(ValueError): compute(d,d)


if __name__=='__main__': unittest.main()
