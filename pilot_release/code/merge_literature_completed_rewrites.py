"""Merge disjoint completed rewrites and retain unresolved failures explicitly."""
import argparse
import json
from pathlib import Path
from literature_method_runner import LiteratureMethodCatalog, sha256_text


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original-root',type=Path,required=True)
    ap.add_argument('--continuation-root',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--experiment-id',required=True)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    goods={}; failures={}
    for root in [args.original_root,args.continuation_root]:
        for shard in sorted(root.glob('shard*')):
            for filename, target in [('realized_documents.jsonl',goods),('errors.jsonl',failures)]:
                p=shard/filename
                if not p.exists():continue
                for line in p.open():
                    d=json.loads(line);key=(d['sample_id'],d['method_id'])
                    if target is goods:
                        assert key not in goods
                        assert d['rewritten_text'] and sha256_text(d['rewritten_text'])==d['method_run']['realized_document_sha256']
                        assert sha256_text(d['source_text'])==d['method_run']['source_document_sha256']
                    target[key]=dict(d,provenance_file=str(p))
    ids=[json.loads(l)['sample_id'] for l in args.manifest.open()]
    methods=LiteratureMethodCatalog().method_ids
    expected={(s,m) for s in ids for m in methods}
    missing=expected-set(goods)
    assert len(goods)==2789 and len(missing)==11
    assert set(goods)<=expected and missing<=set(failures)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'realized_documents.jsonl').open('w') as f:
        for key in sorted(goods):f.write(json.dumps(goods[key],ensure_ascii=False)+'\n')
    with (args.output_dir/'missing_rewrites.jsonl').open('w') as f:
        for key in sorted(missing):f.write(json.dumps(failures[key],ensure_ascii=False)+'\n')
    metrics=dict(experiment_id=args.experiment_id,expected_rewrites=len(expected),
        completed_rewrites=len(goods),missing_rewrites=len(missing),
        original_conditions=len(ids),available_answer_conditions=len(goods)+len(ids),
        per_method={m:sum(k[1]==m for k in goods) for m in methods},
        text_hashes_verified=True,original_inputs_modified=False)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':main()
