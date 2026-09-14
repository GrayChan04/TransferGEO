"""Prepare identical source prefixes for all three answer models, CPU only."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from answer_prompt import build_answer_messages
from audit_benchmark_lengths import count_chat_tokens
from model_loader import MODEL_SPECS
from run_necessity_answer_model_shard import build_literature_conditions
from source_content_truncation import input_hash


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rewrites',type=Path,required=True)
    ap.add_argument('--missing-rewrites',type=Path,required=True)
    ap.add_argument('--overflows',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--experiment-id',required=True)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    conditions,_=build_literature_conditions(Path('data/processed/necessity_pilot_v1/sample_manifest.jsonl'),args.rewrites,
        Path('data/raw/benchmarks/geo_bench/test.jsonl'),Path('data/raw/benchmarks/cseo_bench/data'),missing_rewrites_path=args.missing_rewrites)
    affected={(r['sample_id'],r['prompt_id']) for r in map(json.loads,args.overflows.open())}
    toks={m:AutoTokenizer.from_pretrained(MODEL_SPECS[m].path(Path('data/raw/models')),
          local_files_only=True,trust_remote_code=False,use_fast=True) for m in ['qwen','llama','mistral']}
    out=[]
    for item,prompt,docs in conditions:
        if (item['sample_id'],prompt) not in affected:continue
        def lengths(texts):
            messages=build_answer_messages(protocol='unified_answer_v7',benchmark=item['benchmark'],
                domain=item['domain'],query=item['query'],sources=texts)
            return {m:count_chat_tokens(t,messages,disable_thinking=MODEL_SPECS[m].disable_thinking) for m,t in toks.items()}
        original=lengths(docs)
        assert max(original.values())>20480
        low=0.; high=1.;best=[s[:1] for s in docs];best_counts=lengths(best)
        if max(best_counts.values())>20480:raise ValueError('Fixed prompt/query exceeds budget')
        # Keep the same fraction of each source by Unicode characters. Only
        # prefix deletion; no rephrasing, token decoding, or citation renumbering.
        for _ in range(14):
            ratio=(low+high)/2
            candidate=[s[:max(1,int(len(s)*ratio))] for s in docs]
            counts=lengths(candidate)
            if max(counts.values())<=20480:low=ratio;best=candidate;best_counts=counts
            else:high=ratio
        out.append(dict(sample_id=item['sample_id'],prompt_id=prompt,protocol='unified_answer_v7',
            original_input_sha256=input_hash(item,docs),truncated_input_sha256=input_hash(item,best),
            documents=best,keep_character_fraction=low,original_chars=[len(s) for s in docs],
            truncated_chars=[len(s) for s in best],before_tokens=original,after_tokens=best_counts))
        print(item['sample_id'],prompt,original,'->',best_counts,flush=True)
    assert len(out)==len(affected)==16
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'source_truncations.jsonl').open('w') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    metrics=dict(experiment_id=args.experiment_id,changed_conditions=len(out),
        checked_model_conditions=len(out)*3,remaining_overflow_count=sum(v>20480 for r in out for v in r['after_tokens'].values()),
        source_count_preserved=True,query_and_prompt_unchanged=True,same_text_across_models=True,
        input_budget=20480,gpu_calls=0)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':main()
