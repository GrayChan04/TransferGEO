"""Remove reference tails only; keep prior answers/scores immutable."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from citation_canonicalizer import _strip_trailing_reference_section


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--experiment-id', required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open() as src, (args.output_dir/'canonicalized_answers.jsonl').open('w') as dst, (args.output_dir/'removed_reference_lists.jsonl').open('w') as audit:
        for line in src:
            row = json.loads(line)
            original = row['answer']
            cleaned, removed = _strip_trailing_reference_section(original)
            if _strip_trailing_reference_section(cleaned)[0] != cleaned:
                raise ValueError('Cleanup is not idempotent')
            counts['answers'] += 1
            if removed:
                assert original.startswith(cleaned)
                counts['changed_answers'] += 1
                counts['changed_'+row['model']] += 1
                audit.write(json.dumps({k:row[k] for k in ('model','sample_id','prompt_id','seed')} | {'before_sha256': hashlib.sha256(original.encode()).hexdigest(), 'removal':removed}, ensure_ascii=False)+'\n')
                row['answer'] = cleaned
                row['reference_tail_cleanup'] = {'experiment_id':args.experiment_id, 'previous_answer_sha256':hashlib.sha256(original.encode()).hexdigest(), 'old_scores_require_review':True}
                row['canonicalized_answer_sha256'] = hashlib.sha256(cleaned.encode()).hexdigest()
            dst.write(json.dumps(row, ensure_ascii=False)+'\n')
    assert before == hashlib.sha256(args.input.read_bytes()).hexdigest()
    metrics = {'experiment_id':args.experiment_id, **counts, 'input_sha256':before, 'input_unchanged':True, 'old_scores_updated':False, 'scope':'reference tail removal only; no sentence or citation relocation'}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__ == '__main__':
    main()
