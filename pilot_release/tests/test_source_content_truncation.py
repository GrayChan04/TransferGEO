import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from source_content_truncation import input_hash, apply_source_truncations


class TruncationTest(unittest.TestCase):
    def setUp(self):
        self.item=dict(sample_id='s',benchmark='b',domain='d',query='q')
        self.docs=['first source','second source']
        self.row=dict(sample_id='s',prompt_id='p',protocol='v7',
            original_input_sha256=input_hash(self.item,self.docs),
            documents=['first','second'],
            truncated_input_sha256=input_hash(self.item,['first','second']))

    def apply(self,row):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.jsonl';p.write_text(json.dumps(row)+'\n')
            return apply_source_truncations([(self.item,'p',self.docs)],p,'v7')

    def test_preserves_query_source_count_and_prefixes(self):
        result,_=self.apply(self.row)
        self.assertEqual(result[0],(self.item,'p',['first','second']))
        self.assertEqual(self.docs,['first source','second source'])

    def test_query_change_rejected(self):
        self.item['query']='different'
        with self.assertRaisesRegex(ValueError,'mismatch'):self.apply(self.row)

    def test_rephrasing_rejected(self):
        self.row['documents'][0]='new content'
        with self.assertRaisesRegex(ValueError,'prefix'):self.apply(self.row)

    def test_no_override_preserves_original(self):
        conditions=[(self.item,'p',self.docs)]
        result,_=apply_source_truncations(conditions,None,'v7')
        self.assertIs(result,conditions)


if __name__=='__main__':unittest.main()
