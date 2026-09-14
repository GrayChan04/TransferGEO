"""Verify existing scoring artifacts without regenerating or rescoring."""
import argparse
import json
import hashlib
from pathlib import Path
from score_literature_answers_full import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--experiment-id',required=True)
    a=ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    key=lambda r:(r['model'],r['sample_id'],r['prompt_id'],r.get('seed',r.get('answer_seed')))
    raw=read(a.input_dir/'raw_answers_merged.jsonl'); final=read(a.input_dir/'03_detached_citation_repair/canonicalized_answers.jsonl')
    scores=read(a.input_dir/'04_objective_scores/objective_scores.jsonl')
    keys={key(r) for r in raw}
    assert len(raw)==len(final)==len(scores)==len(keys)==28701
    assert {key(r) for r in final}=={key(r) for r in scores}==keys
    old={key(r):r for r in raw}
    for r in final:
        assert hashlib.sha256(r['answer'].encode()).hexdigest()==r['canonicalized_answer_sha256']
        for f in ['effective_source_input_sha256','source_truncated','source_truncation_file']:
            assert r[f]==old[key(r)][f]
    def overall(r):return r['zero_fallback_sensitivity']['overall']
    assert all(0<=overall(r)<=1 for r in scores)
    metrics=dict(experiment_id=a.experiment_id,answer_count=28701,score_count=28701,
        exact_key_alignment=True,scoring_answer_hashes_verified=True,truncation_metadata_preserved=True,
        answer_text_changed=sum(r['answer']!=old[key(r)]['answer'] for r in final),
        objective_zero_count=sum(overall(r)==0 for r in scores),
        primary_score_field='zero_fallback_sensitivity.overall',gpu_calls=0,rescoring_calls=0,
        scoring_answers=str(a.input_dir/'03_detached_citation_repair/canonicalized_answers.jsonl'),
        objective_scores=str(a.input_dir/'04_objective_scores/objective_scores.jsonl'))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':main()
