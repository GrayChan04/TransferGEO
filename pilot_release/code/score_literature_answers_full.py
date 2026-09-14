"""Assemble verified full answers and run the existing scoring-copy pipeline."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def read(path):
    with path.open() as f:return [json.loads(l) for l in f if l.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--answers-root',type=Path,required=True)
    ap.add_argument('--rewrites',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    manifest=read(args.manifest)
    conditions={(r['sample_id'],r['method_id']) for r in read(args.rewrites)}
    conditions|={(r['sample_id'],'original.no_rewrite') for r in manifest}
    expected={(m,s,p,k) for m in ['qwen','llama','mistral'] for s,p in conditions for k in [0,1,2]}
    answers=[]
    for tag in ['qwen_a','qwen_b','llama','mistral']:
        p=args.answers_root/tag
        assert json.loads((p/'metrics.json').read_text())['execution_success']
        assert not read(p/'errors.jsonl')
        answers.extend(read(p/'answers.jsonl'))
    key=lambda r:(r['model'],r['sample_id'],r['prompt_id'],r['seed'])
    assert len(answers)==len(expected)==28701 and {key(r) for r in answers}==expected
    assert all(r['answer'].strip() and r['answer_prompt_protocol']=='unified_answer_v7' for r in answers)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    merged=args.output_dir/'raw_answers_merged.jsonl'
    with merged.open('w') as f:
        for r in answers:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    errors=args.output_dir/'generation_errors.jsonl';errors.touch()
    def run(script,*argv):
        subprocess.run([sys.executable,'code/'+script,'--experiment-id',args.experiment_id,*map(str,argv)],check=True)
    canonical=args.output_dir/'01_citation_normalization'
    tails=args.output_dir/'02_reference_tail_cleanup'
    detached=args.output_dir/'03_detached_citation_repair'
    scores=args.output_dir/'04_objective_scores'
    run('audit_citation_canonicalization.py','--answers',merged,'--manifest',args.manifest,
        '--output-dir',canonical,'--answer-prompt-protocol','unified_answer_v7',
        '--method-version','deterministic_citation_canonicalization_v3',
        '--expected-answer-count','28701','--models','qwen','llama','mistral')
    run('clean_remaining_reference_lists.py','--input',canonical/'canonicalized_answers.jsonl','--output-dir',tails)
    run('repair_detached_citations.py','--input',tails/'canonicalized_answers.jsonl','--output-dir',detached)
    final=detached/'canonicalized_answers.jsonl'
    run('evaluate_geo_objective.py','--answers',final,'--errors',errors,'--manifest',args.manifest,
        '--output-dir',scores,'--answer-prompt-protocol','unified_answer_v7',
        '--expected-answer-count','28701','--expected-error-count','0')
    scored=read(scores/'objective_scores.jsonl')
    assert len(scored)==28701 and {(r['model'],r['sample_id'],r['prompt_id'],r['answer_seed']) for r in scored}==expected
    finalrows=read(final);assert len(finalrows)==28701 and {key(r) for r in finalrows}==expected
    old={key(r):r for r in answers}
    for r in finalrows:
        for field in ['effective_source_input_sha256','source_truncated','source_truncation_file']:
            assert r[field]==old[key(r)][field]
    metrics=dict(experiment_id=args.experiment_id,answers=28701,objective_scores=len(scored),
        primary_score_field='zero_fallback_sensitivity.overall',
        scoring_answers=str(final),objective_scores_path=str(scores/'objective_scores.jsonl'),
        answer_text_changed=sum(r['answer']!=old[key(r)]['answer'] for r in finalrows),
        truncation_metadata_preserved=True,raw_answers_modified=False,gpu_calls=0,
        subjective_judge_started=False)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':main()
