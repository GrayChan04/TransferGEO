import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from json_format_repair import parse_json_format_only


class JsonFormatRepairTest(unittest.TestCase):
    def test_inner_quotes(self):
        payload, audit = parse_json_format_only('{\n  "suggestion": "Use "one-two-many" examples."\n}')
        self.assertEqual(payload,{'suggestion':'Use "one-two-many" examples.'})
        self.assertEqual(json.loads(audit['after']),payload)

    def test_existing_escapes_unchanged(self):
        text=json.dumps({'excerpt':'A "quoted" word.', 'necessity':75})
        payload,audit=parse_json_format_only(text)
        self.assertEqual(audit['after'],text)
        self.assertEqual(payload['necessity'],75)

    def test_fence_and_terminal_marker(self):
        payload,_=parse_json_format_only('```json\n{"a":1}\n```<|user|>')
        self.assertEqual(payload,{'a':1})

    def test_reject_incomplete_or_prose(self):
        for text in ('{"a":', 'Here is JSON: {"a":1}', '{\n"a": "x", "b": broken\n}'):
            with self.assertRaises((ValueError,json.JSONDecodeError)):
                parse_json_format_only(text)


if __name__=='__main__':unittest.main()
