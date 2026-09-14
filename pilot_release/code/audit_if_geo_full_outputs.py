"""Read-only input audit; writes separate diagnostics, never repairs outputs."""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

from json_format_repair import parse_json_format_only


def repetition(text):
    paragraphs = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
    # Count separately: a one-line paragraph must not be counted twice.
    pcs = collections.Counter(x for x in paragraphs if len(x) >= 80)
    lcs = collections.Counter(x.strip() for x in text.splitlines() if len(x.strip()) >= 80)
    candidates = [(v, k) for c in (pcs, lcs) for k, v in c.items() if v >= 2]
    count, example = max(candidates, default=(0, ''))
    return dict(paragraph_count=len(paragraphs),
                duplicate_long_paragraph_occurrences=sum(v-1 for v in pcs.values()),
                max_identical_long_unit_occurrences=count,
                repeated_example=example[:600])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--experiment-id', required=True)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    capped, failures = [], []
    seen = collections.Counter()
    expected = set()
    for shard in sorted(args.input_root.glob('shard*')):
        m = json.loads((shard/'metrics.json').read_text())
        expected.update((s, k) for s in m['selected_sample_ids'] for k in m['method_ids'])
        for line in (shard/'realized_documents.jsonl').open():
            d = json.loads(line)
            seen[d['sample_id'], d['method_id']] += 1
            if d['method_id'] != 'if_geo.full_pipeline':
                continue
            stage = d['method_run']['stage_records'][-1]
            if stage['output_tokens'] < stage['max_new_tokens']:
                continue
            a = d['rewritten_text']
            stops = ['<|user|>', '<|endoftext|>', '<|observation|>']
            capped.append(dict(sample_id=d['sample_id'], benchmark=d['benchmark'],
                input_file=str(shard/'realized_documents.jsonl'),
                output_tokens=stage['output_tokens'], max_new_tokens=stage['max_new_tokens'],
                source_tokens=d['method_run']['source_document_tokens'],
                source_chars=len(d['source_text']), output_chars=len(a),
                ends_with_native_stop=any(stage['raw_output'].rstrip().endswith(s) for s in stops),
                ends_with_sentence_punctuation=bool(re.search(r'[.!?。！？][\s\"\u201d\u2019\)*_]*$', a)),
                ending=a[-400:], **repetition(a)))
        for line in (shard/'errors.jsonl').open():
            d=json.loads(line); seen[d['sample_id'], d['method_id']] += 1
            s=d['last_completed_stage_record']; a=s['output']
            item=dict(sample_id=d['sample_id'], stage=s['stage_id'], error=d['error'],
                input_file=str(shard/'errors.jsonl'), output_tokens=s['output_tokens'],
                max_new_tokens=s['max_new_tokens'], output_at_cap=s['output_tokens']>=s['max_new_tokens'],
                output=a, empty_excerpt_count=0, missing_excerpt_count=0, empty_suggestion_count=0)
            if 'require excerpt and suggestion' in d['error']:
                payload, _ = parse_json_format_only(a)
                suggestions=payload['suggestions']
                item['empty_excerpt_count']=sum('excerpt' in x and not str(x['excerpt']).strip() for x in suggestions)
                item['missing_excerpt_count']=sum('excerpt' not in x for x in suggestions)
                item['empty_suggestion_count']=sum(not str(x.get('suggestion','')).strip() for x in suggestions)
                item['category']='empty_excerpt'
            elif 'Invalid \\escape' in d['error']:
                item['category']='invalid_escape'
            elif 'Ambiguous quote' in d['error']:
                item['category']='ambiguous_quotes'
            elif s['stage_id']=='if_geo.blueprint_construction' and "Expecting ','" in d['error']:
                item['category']='blueprint_missing_comma'
            else:
                item['category']='other_json_syntax'
            failures.append(item)
    assert len(expected)==2800 and set(seen)==expected and all(v==1 for v in seen.values())
    metrics=dict(experiment_id=args.experiment_id, accounted_conditions=len(seen),
        failure_count=len(failures), failure_categories=dict(collections.Counter(x['category'] for x in failures)),
        failed_output_at_cap=sum(x['output_at_cap'] for x in failures),
        if_geo_capped_count=len(capped),
        capped_with_exact_long_repetition=sum(x['max_identical_long_unit_occurrences']>=2 for x in capped),
        capped_with_native_stop=sum(x['ends_with_native_stop'] for x in capped),
        capped_without_sentence_punctuation=sum(not x['ends_with_sentence_punctuation'] for x in capped),
        diagnostics_only=True, gpu_calls=0, input_modified=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    for name, rows in [('capped_outputs',capped),('failed_outputs',failures)]:
        with (args.output_dir/(name+'.jsonl')).open('w') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (args.output_dir/'capped_outputs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(capped[0]));w.writeheader();w.writerows(capped)
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':
    main()
