"""Plan subjective score reuse against the actual previously scored text."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from audit_subjective_score_alignment import classify_alignment


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def key(row):
    return (row['model'], row['sample_id'], row['prompt_id'],
            int(row.get('answer_seed', row.get('seed'))))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cited(text, target):
    return target in {int(n) for n in re.findall(r'\[(\d+)\]', text)}


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('old-answers', 'new-answers', 'old-scores', 'manifest', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--experiment-id', required=True)
    p.add_argument('--expected-count', type=int, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    targets = {r['sample_id']:r for r in rows(args.manifest)}
    old = {}
    for r in rows(args.old_answers):
        k = key(r)
        assert k not in old
        old[k] = (digest(r['answer']), cited(r['answer'],targets[k[1]]['target_document_index']+1))
    verified = set()
    trial_count = 0
    for r in rows(args.old_scores):
        k = key(r)
        assert k in old and k not in verified
        verified.add(k)
        assert r['canonicalized_answer_sha256'] == old[k][0]
        assert r['target_source_cited'] == old[k][1]
        target = targets[k[1]]
        for field in ('target_document_index','candidate_document_count'):
            assert r[field] == target[field]
        assert r['target_citation_index'] == target['target_document_index']+1
        assert len(r['dimensions']) == 7
        means = []
        for dimension, value in r['dimensions'].items():
            trials = value['trials']
            assert sorted(t['judge_seed'] for t in trials) == [0,1,2]
            assert all(key(t) == k and t['dimension'] == dimension for t in trials)
            scores = [t['score'] for t in trials]
            assert all(type(s) is int and 0 <= s <= 5 for s in scores)
            mean = sum(scores)/3
            assert math.isclose(mean,value['mean_score'],abs_tol=1e-12)
            means.append(mean)
            trial_count += 3
        assert math.isclose(sum(means)/7,r['subjective_average'],abs_tol=1e-12)
        if not old[k][1]:
            assert r['subjective_average'] == 0
    assert len(old) == args.expected_count and verified == old.keys()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    counts, transitions, groups = Counter(), Counter(), Counter()
    seen = set()
    with (args.output_dir/'answer_alignment.jsonl').open('w') as alignment, (args.output_dir/'rerun_glm_judge_answers.jsonl').open('w') as rerun:
        for r in rows(args.new_answers):
            k = key(r)
            assert k in old and k not in seen
            seen.add(k)
            target = targets[k[1]]['target_document_index']+1
            before_hash, before_cited = old[k]
            after_hash = digest(r['answer'])
            assert r['canonicalized_answer_sha256'] == after_hash
            after_cited = cited(r['answer'],target)
            changed = before_hash != after_hash
            action = classify_alignment(text_changed=changed,target_cited_after=after_cited)
            counts[action] += 1
            counts['changed_answers'] += changed
            transitions[f'{before_cited}_to_{after_cited}'] += 1
            groups[f'{k[0]}/{action}'] += 1
            record = dict(model=k[0],sample_id=k[1],prompt_id=k[2],answer_seed=k[3],
                          target_citation_index=target, previous_scored_answer_sha256=before_hash,
                          canonicalized_answer_sha256=after_hash, text_changed=changed,
                          previous_target_cited=before_cited,target_source_cited=after_cited,
                          subjective_score_action=action)
            alignment.write(json.dumps(record,ensure_ascii=False)+'\n')
            if action == 'rerun_glm_judge':
                rerun.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert seen == old.keys()
    metrics = dict(experiment_id=args.experiment_id,answer_count=len(seen),old_trials_verified=trial_count,
                   counts=counts,transitions=transitions,model_actions=groups,
                   required_glm_trials=counts['rerun_glm_judge']*21,
                   integrity_passed=True, gpu_used=False, old_scores_modified=False,
                   status='plan_only_no_judge_calls',
                   inputs={name:str(getattr(args,name)) for name in ('old_answers','new_answers','old_scores','manifest')})
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__ == '__main__':
    main()
