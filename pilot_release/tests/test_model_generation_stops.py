import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from model_loader import generation_eos_override


class GenerationStopsTest(unittest.TestCase):
    def make(self, model_eos, tokenizer_eos):
        return SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos)),tokenizer=SimpleNamespace(eos_token_id=tokenizer_eos))

    def test_preserves_model_list(self):
        loaded=self.make([151329,151336,151338],151336)
        self.assertEqual(generation_eos_override(loaded),{})
        self.assertEqual(loaded.model.generation_config.eos_token_id,[151329,151336,151338])

    def test_preserves_model_single(self):
        self.assertEqual(generation_eos_override(self.make(42,99)),{})

    def test_fallback(self):
        self.assertEqual(generation_eos_override(self.make(None,99)),{'eos_token_id':99})
        self.assertEqual(generation_eos_override(self.make(None,None)),{})


if __name__=='__main__':unittest.main()
