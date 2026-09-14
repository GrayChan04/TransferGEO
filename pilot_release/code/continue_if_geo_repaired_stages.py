"""Continue inspected IF-GEO JSON repairs, replaying saved calls by hash."""
import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from literature_method_runner import (BackendGeneration, LiteratureMethodCatalog,
    LiteratureMethodExecutor, StageBudgets, canonical_json, sha256_text)
from run_literature_method_rewrite_stage import GlmChatBackend, build_condition_error_payload


class ReachedNewStage(Exception):
    pass


class ReplayBackend:
    def __init__(self, saved, source, source_tokens, live=None):
        self.saved, self.source, self.source_tokens = saved, source, source_tokens
        self.live, self.index, self.new_calls = live, 0, 0

    def count_text_tokens(self, text):
        assert text == self.source
        if self.live:
            assert self.live.count_text_tokens(text) == self.source_tokens
        return self.source_tokens

    def generate(self, messages, *, stage_id, max_new_tokens, seed):
        assert seed == 0
        if self.index < len(self.saved):
            s = self.saved[self.index]
            assert s['stage_id'] == stage_id
            assert s['max_new_tokens'] == max_new_tokens
            assert s['messages_sha256'] == sha256_text(canonical_json(messages)), stage_id
            assert s['output_sha256'] == sha256_text(s['output'])
            self.index += 1
            return BackendGeneration(answer=s['output'], raw_answer=s['raw_output'],
                input_tokens=s['input_tokens'], output_tokens=s['output_tokens'], model_id=s['model_id'])
        if self.live is None:
            raise ReachedNewStage(stage_id)
        self.new_calls += 1
        return self.live.generate(messages, stage_id=stage_id, max_new_tokens=max_new_tokens, seed=seed)


class RepairedExecutor(LiteratureMethodExecutor):
    def __init__(self, repair, **kwargs):
        super().__init__(**kwargs)
        self.repair = repair

    def _parse_json_stage(self, text, *, stage_id):
        if stage_id == self.repair['stage']:
            assert text == self.repair['original_output']
            assert sha256_text(text) == self.repair['original_sha256']
            fixed = self.repair['repaired_output']
            assert sha256_text(fixed) == self.repair['repaired_sha256']
            assert self._stage_records[-1].stage_id == stage_id
            self._stage_records[-1] = replace(self._stage_records[-1],
                serialization_normalized=True,
                format_repair={'policy':'inspected_saved_json_syntax_repair_v1',
                    'before':text, 'after':fixed, 'diff':self.repair['diff'],
                    'content_completion':False})
            return json.loads(fixed)
        return super()._parse_json_stage(text, stage_id=stage_id)


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repairs', type=Path, required=True)
    ap.add_argument('--shard-index', type=int, required=True)
    ap.add_argument('--num-shards', type=int, default=4)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--model-root', type=Path, default=Path('data/raw/models'))
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    rows=[json.loads(l) for l in args.repairs.open()]
    assert len(rows)==18 and len({x['sample_id'] for x in rows})==18
    selected=rows[args.shard_index::args.num_shards]
    catalog=LiteratureMethodCatalog()
    budgets=StageBudgets(intermediate_max_new_tokens=2048, if_geo_aggregation_max_new_tokens=8192,
                         rewrite_growth_ratio=2.0, rewrite_growth_slack_tokens=256)
    jobs=[]
    for r in selected:
        errors=[json.loads(l) for l in Path(r['input_file']).open()]
        old=next(x for x in errors if x['sample_id']==r['sample_id'] and x['method_id']=='if_geo.full_pipeline')
        saved=old['completed_stage_records']
        assert saved[-1]['stage_id']==r['stage'] and saved[-1]['output']==r['original_output']
        with Path(r['input_file']).with_name('realized_documents.jsonl').open() as f:
            source=next(json.loads(l) for l in f if json.loads(l)['sample_id']==r['sample_id'])
        jobs.append((r,saved,source))
    if not args.validate_only:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True,exist_ok=True)
    context=nullcontext(None) if args.validate_only else GlmChatBackend(
        model_root=args.model_root,device='cuda:0',dtype='bfloat16',attn_implementation='sdpa',
        device_map_strategy='balanced',do_sample=False)
    completed=errors=attempts=reused=0
    with context as live:
        for r,saved,source in jobs:
            backend=ReplayBackend(saved,source['source_text'],source['method_run']['source_document_tokens'],live)
            executor=RepairedExecutor(r,catalog=catalog,backend=backend,budgets=budgets,json_format_repair=True)
            try:
                result=executor.run(source['source_text'],method_id='if_geo.full_pipeline',seed=0)
                payload={k:source[k] for k in ['sample_id','benchmark','domain','instance_id','target_document_index','source_text']}
                payload.update(experiment_id=args.experiment_id,method_id='if_geo.full_pipeline',
                    rewritten_text=result.realized_document,method_run=result.to_dict(),
                    reused_stage_count=backend.index,new_model_calls=backend.new_calls,
                    parent_error_file=r['input_file'],repair_file=str(args.repairs))
                append(args.output_dir/'realized_documents.jsonl',payload)
                completed+=1
            except ReachedNewStage as exc:
                assert args.validate_only and backend.index==len(saved)
                print(r['sample_id'], 'saved_requests_verified', backend.index,
                      'remaining_calls',10-len(saved),'next_stage',str(exc),flush=True)
            except Exception as exc:
                if args.validate_only:
                    raise
                errors+=1
                payload=build_condition_error_payload(experiment_id=args.experiment_id,
                    sample_id=r['sample_id'],method_id='if_geo.full_pipeline',executor=executor,exc=exc)
                payload.update(reused_stage_count=backend.index,new_model_calls=backend.new_calls)
                append(args.output_dir/'errors.jsonl',payload)
                if 'out of memory' in str(exc).lower() or 'cuda' in str(exc).lower():
                    raise
            attempts+=backend.new_calls
            reused+=backend.index
            if not args.validate_only:
                print(f'processed={completed+errors}/{len(jobs)} completed={completed} errors={errors}',flush=True)
    metrics=dict(experiment_id=args.experiment_id,expected_conditions=len(jobs),
        completed_conditions=completed,error_count=errors,new_model_call_attempts=attempts,
        reused_stage_count=reused,planned_max_new_calls=sum(10-len(s) for _,s,_ in jobs),
        do_sample=False,seed=0,capped_outputs_modified=0,empty_excerpt_modified=0)
    if not args.validate_only:
        (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))
    return int(errors>0)


if __name__=='__main__':
    raise SystemExit(main())
