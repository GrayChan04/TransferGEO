"""Repair only the 18 inspected saved JSON failures, with reversible diffs.

Not a replacement for the formal runner's parser; no model calls.
"""
import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from literature_method_runner import LiteratureMethodExecutor


def repair(row):
    original = row['output']
    text = original.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    sid = row['sample_id']
    if sid == 'cseo_bench__news__207':
        old = ' // Assuming this is the end of the current analysis section'
        assert text.count(old) == 1
        text = text.replace(old, '')
    if sid == 'cseo_bench__web__111':
        old = '"id": "suggest_24",\n    {'
        assert text.count(old) == 1
        text = text.replace(old, '"id": "suggest_24",')
        old = '\n    }\n  }\n]'
        assert text.endswith(old)
        text = text[:-len(old)] + '\n    }\n]'
    if sid == 'geo_bench__all__296':
        old = 'languages around us."\','
        assert text.count(old) == 1
        text = text.replace(old, 'languages around us.",')
    # These five inspected files contain one-property-per-line quoted excerpts.
    # Preserve already-valid escape pairs; escape internal quotes. For the
    # three literal-path/math cases, preserve literal backslashes in excerpts.
    quote_ids = {'geo_bench__all__296', 'geo_bench__all__904'}
    slash_ids = {'cseo_bench__retail__93241', 'geo_bench__all__365', 'cseo_bench__web__238'}
    if sid in quote_ids | slash_ids:
        fixed = []
        for line in text.splitlines(keepends=True):
            m = re.fullmatch(r'(\s*"excerpt"\s*:\s*")(.*)("\s*,?\s*)', line)
            if not m:
                fixed.append(line)
                continue
            body = m[2]
            if sid in slash_ids:
                # Literal slashes in paths/LaTeX, not JSON control sequences.
                body = re.sub(r'\\(?!["\\])', r'\\\\', body)
            body = re.sub(r'(?<!\\)"', r'\\"', body)
            fixed.append(m[1] + body + m[3])
        text = ''.join(fixed)
    # Only insert a comma at an inspected missing-comma boundary. Reject any
    # other parse failure rather than completing missing text.
    for _ in range(30):
        try:
            payload = json.loads(text)
            break
        except json.JSONDecodeError as exc:
            if row['category'] != 'blueprint_missing_comma' or exc.msg != "Expecting ',' delimiter":
                raise
            previous = text[:exc.pos].rstrip()
            assert text[exc.pos] == '"' and previous.endswith('"')
            text = previous + ',' + text[len(previous):]
    else:
        raise ValueError('Too many comma repairs')
    if row['stage'].startswith('if_geo.query_request_'):
        LiteratureMethodExecutor._validate_suggestions(payload, maximum=5)
    elif row['stage'] == 'if_geo.blueprint_construction':
        assert isinstance(payload, dict) and isinstance(payload.get('revision_blueprint'), list)
    elif row['stage'] == 'if_geo.prioritization_deduplication':
        assert isinstance(payload, list)
    else:
        raise ValueError(row['stage'])
    # Apart from the explicit JSON comment, all non-syntax text is unchanged.
    before = original.removeprefix('```json\n').removesuffix('\n```')
    if sid == 'cseo_bench__news__207':
        before = before.replace(' // Assuming this is the end of the current analysis section', '')
    content = lambda x: re.sub(r'[\s\\"\'{}\[\],:]', '', x)
    assert content(before) == content(text), 'Unexpected content change'
    return dict(sample_id=sid, stage=row['stage'], input_file=row['input_file'],
        original_output=original, repaired_output=text, parsed_payload=payload,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        repaired_sha256=hashlib.sha256(text.encode()).hexdigest(),
        diff=''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                        fromfile='original', tofile='repaired')),
        current_stage_schema_passed=True, content_added=False,
        requires_remaining_model_stages=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--validate-only', action='store_true')
    args=ap.parse_args()
    rows=[json.loads(l) for l in args.input.open()]
    selected=[x for x in rows if x['category']!='empty_excerpt']
    assert len(rows)==28 and len(selected)==18
    fixed=[repair(x) for x in selected]
    metrics=dict(experiment_id=args.experiment_id, json_repaired=18,
                 current_stage_schema_passed=18, empty_excerpt_unchanged=10,
                 gpu_calls=0, final_rewrites_generated=0, capped_outputs_modified=0)
    if not args.validate_only:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        with (args.output_dir/'repaired_stage_outputs.jsonl').open('w') as f:
            for x in fixed: f.write(json.dumps(x,ensure_ascii=False)+'\n')
        (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':
    main()
