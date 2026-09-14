"""Repair citation-only sentence fragments in immutable scoring copies."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from geo_objective_metrics import parse_answer_official

LABELS = re.compile(r'\[\d+\]')
FRAGMENT = re.compile(r'''\s*\(?\s*(?:\[\d+\]\s*)+\)?[.!?。！？]?\s*["”’]?\s*''')
END = re.compile(r'''([.!?。！？])(["'”’）)]*)$''')


def repair(answer):
    """Attach only pure citation fragments within the same paragraph.

    Never merge prose (including 'A [3].'), cross blank lines, or add labels.
    """
    changes = []
    paragraphs = re.split(r'(\n\s*\n)', answer)
    for i in range(0, len(paragraphs), 2):
        paragraph = paragraphs[i]
        sentences = parse_answer_official(paragraph)
        spans = []
        cursor = 0
        for sentence in sentences:
            start = paragraph.index(sentence.text, cursor)
            end = start + len(sentence.text)
            spans.append((start, end, sentence.text))
            cursor = end
        for j in range(len(spans)-1, 0, -1):
            start, end, fragment = spans[j]
            prev_start, prev_end, previous = spans[j-1]
            if not FRAGMENT.fullmatch(fragment) or FRAGMENT.fullmatch(previous):
                continue
            # Line-separated citation blocks are ambiguous; do not backfill them.
            if '\n' in paragraph[prev_end:start]:
                continue
            punctuation = END.search(previous)
            if punctuation is None:
                continue
            labels = ''.join(LABELS.findall(fragment))
            pos = prev_start + punctuation.start()
            replacement = ' ' + labels + paragraph[pos:prev_end]
            changes.append({'before':paragraph[prev_start:end],
                            'after':paragraph[prev_start:pos]+replacement})
            paragraph = paragraph[:pos] + replacement + paragraph[end:]
        paragraphs[i] = paragraph
    result = ''.join(paragraphs)
    assert Counter(LABELS.findall(answer)) == Counter(LABELS.findall(result))
    return result, changes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.input.read_bytes()).hexdigest()
    counts = Counter()
    with args.input.open() as source, (args.output_dir/'canonicalized_answers.jsonl').open('w') as output, (args.output_dir/'citation_repairs.jsonl').open('w') as audit:
        for line in source:
            row = json.loads(line)
            old = row['answer']
            new, changes = repair(old)
            assert repair(new)[0] == new, 'Non-idempotent repair'
            counts['answers'] += 1
            if changes:
                counts['changed_answers'] += 1
                counts['changed_'+row['model']] += 1
                counts['repaired_fragments'] += len(changes)
                row['answer'] = new
                row['canonicalized_answer_sha256'] = hashlib.sha256(new.encode()).hexdigest()
                row['detached_citation_repair'] = {'experiment_id':args.experiment_id, 'previous_answer_sha256':hashlib.sha256(old.encode()).hexdigest(), 'old_scores_require_review':True}
                audit.write(json.dumps({k:row[k] for k in ('model','sample_id','prompt_id','seed')} | {'changes':changes},ensure_ascii=False)+'\n')
            output.write(json.dumps(row,ensure_ascii=False)+'\n')
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == fingerprint
    metrics = {'experiment_id':args.experiment_id, **counts, 'input_sha256':fingerprint, 'input_unchanged':True, 'old_scores_updated':False}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
