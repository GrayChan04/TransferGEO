"""One paired-seed normalized-improvement implementation for both pilots."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean

from summarize_literature_overall import read_scores, ORIGINAL, FIXED, PIPELINES

NEUTRAL = 'neutral_rewrite_control.neutral_rewrite_control'


def improvement(after, before):
    return 10 * (after - before) / (before + 1)


def compute(obj, sub):
    if obj.keys() != sub.keys():
        raise ValueError('Objective/subjective key mismatch')
    paired, grouped = [], defaultdict(list)
    for key, (benchmark, objective) in sorted(obj.items()):
        model, sample, method, seed = key
        base = (model, sample, ORIGINAL, seed)
        if sub[key][0] != benchmark or obj[base][0] != benchmark or sub[base][0] != benchmark:
            raise ValueError('Benchmark mismatch')
        row = dict(model=model, sample_id=sample, benchmark=benchmark, method=method, answer_seed=seed)
        for metric, after, before in [('objective', objective, obj[base][1]),
                                      ('subjective', sub[key][1], sub[base][1])]:
            row[metric+'_before'] = before
            row[metric+'_after'] = after
            row[metric+'_improvement'] = improvement(after, before)
        paired.append(row)
        grouped[(model, benchmark, sample, method)].append(row)
    units = []
    for (model, benchmark, sample, method), records in sorted(grouped.items()):
        if {r['answer_seed'] for r in records} != {0,1,2} or len(records) != 3:
            raise ValueError('Incomplete seeds')
        r = dict(model=model, benchmark=benchmark, sample_id=sample, method=method)
        for metric in ('objective','subjective'):
            r[metric] = fmean(x[metric+'_improvement'] for x in records)
        units.append(r)
    return paired, units


def summarize(units):
    groups = defaultdict(list)
    for r in units:
        groups[(r['model'],r['benchmark'],r['method'])].append(r)
    rows = []
    for (model, benchmark, method), records in sorted(groups.items()):
        rows.append(dict(model=model, benchmark=benchmark, method=method, n=len(records),
                         objective=fmean(r['objective'] for r in records),
                         subjective=fmean(r['subjective'] for r in records)))
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    specs=json.loads(args.spec.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metrics={'formula':'10*(after-before)/(before+1)', 'order':'paired answer seed -> sample mean -> benchmark mean',
             'metric_name_cn':'归一化提升分数','gpu_calls':0,'parts':{}}
    all_rows, winners, reversals = [], [], []
    for part, spec in specs.items():
        obj,sub=read_scores(Path(spec['objective'])),read_scores(Path(spec['subjective']),True)
        if len(obj)!=spec['expected_count']: raise ValueError('Unexpected score count')
        paired,units=compute(obj,sub)
        methods={r['method'] for r in units}
        if len(methods)!=spec['expected_methods']: raise ValueError('Unexpected methods')
        members=defaultdict(set)
        for r in units: members[r['sample_id']].add((r['model'],r['method']))
        if spec['selection']=='common_complete':
            selected={s for s,v in members.items() if len(v)==3*len(methods)}
            if len(selected)!=397: raise ValueError('Part1 common sample count changed')
            # Verify exact historical sample membership, not just the count.
            with Path(spec['previous_units']).open() as f:
                previous={r['sample_id'] for r in csv.DictReader(f) if r['is_common_complete']=='True'}
            if selected != previous: raise ValueError('Historical selection mismatch')
        else:
            selected=set(members)
            if len(selected)!=400: raise ValueError('Part2 sample count changed')
        units=[r for r in units if r['sample_id'] in selected]
        paired=[r for r in paired if r['sample_id'] in selected]
        for method in methods:
            sets=[{r['sample_id'] for r in units if r['model']==m and r['method']==method}
                  for m in ('qwen','llama','mistral')]
            if any(s!=sets[0] for s in sets): raise ValueError('Cross-model coverage mismatch')
        summary=summarize(units)
        directory=args.output_dir/part; directory.mkdir()
        write_csv(directory/'逐答案配对归一化提升分数.csv',paired)
        write_csv(directory/'逐样本归一化提升分数.csv',units)
        write_csv(directory/'各模型各benchmark各方法归一化提升分数.csv',summary)
        eligible=methods-{ORIGINAL,NEUTRAL} if part=='part1' else set(FIXED)
        for benchmark in sorted({r['benchmark'] for r in summary}):
            for model in ('qwen','llama','mistral'):
                candidates=[r for r in summary if r['benchmark']==benchmark and r['model']==model and r['method'] in eligible]
                for metric in ('objective','subjective'):
                    best=max(r[metric] for r in candidates)
                    for r in candidates:
                        if abs(r[metric]-best)<1e-12:
                            winners.append(dict(part=part,benchmark=benchmark,model=model,metric=metric,method=r['method'],score=best,n=r['n']))
            lookup={(r['model'],r['method']):r for r in summary if r['benchmark']==benchmark}
            for a,b in combinations(sorted(eligible),2):
                for metric in ('objective','subjective'):
                    differences={m:lookup[(m,a)][metric]-lookup[(m,b)][metric] for m in ('qwen','llama','mistral')}
                    if min(differences.values()) < -1e-12 and max(differences.values()) > 1e-12:
                        reversals.append(dict(part=part,benchmark=benchmark,metric=metric,method_a=a,method_b=b,**differences))
        all_rows += [dict(part=part,**r) for r in summary]
        metrics['parts'][part]=dict(input_answers=len(obj),selected_answers=len(paired),samples=len(selected),
                                   sample_model_method_units=len(units),summary_rows=len(summary),
                                   input_files={k:dict(path=spec[k],sha256=hashlib.file_digest(Path(spec[k]).open('rb'),'sha256').hexdigest()) for k in ('objective','subjective')})
    write_csv(args.output_dir/'两部分最高归一化提升方法.csv',winners)
    write_csv(args.output_dir/'两部分方法排序反转.csv',reversals)
    write_csv(args.output_dir/'两部分归一化提升分数汇总.csv',all_rows)
    metrics['winners']=winners
    metrics['rank_reversal_counts']={f'{p}/{b}/{m}':sum(r['part']==p and r['benchmark']==b and r['metric']==m for r in reversals)
       for p in specs for b in ('geo_bench','cseo_bench') for m in ('objective','subjective')}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__': main()
