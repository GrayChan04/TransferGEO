"""Self-contained checks; no model, benchmark text, or network required."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from answer_prompt import build_answer_messages, build_geo_messages, build_cseo_messages


class PublicPackageTests(unittest.TestCase):
    def test_v7_exact_rendering(self):
        for benchmark in ('geo_bench', 'cseo_bench'):
            messages = build_answer_messages(protocol='unified_answer_v7', benchmark=benchmark,
                domain='web', query='Example?', sources=['First.', 'Second.'])
            p = ROOT / 'prompts/answer_generation/unified_v7'
            self.assertEqual(messages[0]['content'], (p/'system_prompt.txt').read_text().strip())
            self.assertEqual(messages[1]['content'], (p/'user_prompt.txt').read_text().strip().format(
                query='Example?', sources='Source 1:\nFirst.\n\nSource 2:\nSecond.',
                valid_citation_identifiers='[1], [2]'))

    def test_omitted_official_templates_fail(self):
        with self.assertRaisesRegex(ValueError, 'not included'):
            build_geo_messages('Q', ['A'])
        with self.assertRaisesRegex(ValueError, 'not included'):
            build_cseo_messages('web', 'Q', ['A'])

    def test_code_syntax(self):
        for path in (ROOT/'code').glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main()
