import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from citation_canonicalizer import canonicalize_citations, _strip_trailing_reference_section


class SuccessiveTailsTest(unittest.TestCase):
    def test_named_tails_are_removed_in_one_canonicalization(self):
        result=canonicalize_citations('Body text.\n\n(Source 1)\n\n(Source 7)',source_count=7)
        self.assertEqual(result['answer'],'Body text.')
        self.assertEqual(canonicalize_citations(result['answer'],source_count=7)['answer'],result['answer'])

    def test_body_citation_not_removed(self):
        text='Supported body [1].\n\n[2]\n\n[3]'
        cleaned,_=_strip_trailing_reference_section(text)
        self.assertEqual(cleaned,'Supported body [1].')
        self.assertEqual(_strip_trailing_reference_section(cleaned),(cleaned,None))


if __name__=='__main__':unittest.main()
