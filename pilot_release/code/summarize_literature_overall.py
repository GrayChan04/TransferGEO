"""Paired descriptive overall tables; no inference, rescoring, or GPU calls."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ORIGINAL = 'original.no_rewrite'
FIXED = ['geo.statistics_addition', 'geo.cite_sources', 'geo.quotation_addition',
         'cseo.content_improvement', 'cseo.llm_guidance']
PIPELINES = ['raid_gseo.full_pipeline', 'if_geo.full_pipeline']


def read_scores(path, subjective=False):
    out = {}
    for line in path.open():
        r = json.loads(line)
        key = tuple(r[k] for k in ('model', 'sample_id', 'prompt_id', 'answer_seed'))
        if key in out:
            raise ValueError(f'Duplicate: {key}')
        score = r['subjective_average'] if subjective else r['zero_fallback_sensitivity']['overall']
        if not math.isfinite(score) or not 0 <= score <= (5 if subjective else 1):
            raise ValueError(f'Invalid score: {key}')
        if subjective and not r['all_dimensions_scorable']:
            raise ValueError(f'Incomplete score: {key}')
        out[key] = (r['benchmark'], score)
    return out


def aggregate(obj, subj):
    if obj.keys() != subj.keys():
        raise ValueError('Objective/subjective key mismatch')
    groups = defaultdict(dict)
    for k, (bench, score) in obj.items():
        if bench != subj[k][0]:
            raise ValueError('Benchmark mismatch')
        groups[(k[0], bench, k[1], k[2])][k[3]] = (score, subj[k][1])
    units = {}
    for k, seeds in groups.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError(f'Incomplete seeds: {k}')
        units[k] = tuple(fmean(seeds[s][i] for s in (0, 1, 2)) for i in (0, 1))
    return units


def summarize(units):
    groups = defaultdict(list)
    for (model, bench, sample, method), scores in units.items():
        baseline = units[(model, bench, sample, ORIGINAL)]
        groups[(model, bench, method)].append((scores, baseline))
    rows = []
    for (model, bench, method), pairs in sorted(groups.items()):
        row = dict(model=model, benchmark=bench, method=method, n=len(pairs))
        for i, metric in enumerate(('objective', 'subjective')):
            after = fmean(p[0][i] for p in pairs)
            before = fmean(p[1][i] for p in pairs)
            row.update({f'{metric}_before': before, f'{metric}_after': after,
                        f'{metric}_delta': after-before})
            for label, predicate in [('up', lambda d: d > 1e-12),
                                     ('down', lambda d: d < -1e-12),
                                     ('unchanged', lambda d: abs(d) <= 1e-12)]:
                row[f'{metric}_{label}'] = sum(predicate(p[0][i]-p[1][i]) for p in pairs)
            row[f'{metric}_both_zero'] = sum(p[0][i] == 0 and p[1][i] == 0 for p in pairs)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objective', type=Path, required=True)
    p.add_argument('--subjective', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    obj, subj = read_scores(args.objective), read_scores(args.subjective, True)
    units = aggregate(obj, subj)
    if len(obj) != 28701 or len(units) != 9567:
        raise ValueError('Unexpected full literature run size')
    methods = {k[3] for k in units}
    if methods != {ORIGINAL, *FIXED, *PIPELINES}:
        raise ValueError(methods)
    # Require identical sample membership across models for each method.
    models = sorted({k[0] for k in units})
    if set(models) != {'qwen', 'llama', 'mistral'}:
        raise ValueError(models)
    for method in methods:
        sets = [{(k[1], k[2]) for k in units if k[0] == m and k[3] == method} for m in models]
        if any(s != sets[0] for s in sets) or len(sets[0]) != (389 if method == 'if_geo.full_pipeline' else 400):
            raise ValueError(f'Unexpected sample coverage: {method}')
    rows = summarize(units)
    winners = []
    for bench in sorted({r['benchmark'] for r in rows}):
        for model in models:
            eligible = [r for r in rows if r['model'] == model and r['benchmark'] == bench and r['method'] in FIXED]
            for metric in ('objective', 'subjective'):
                best = max(r[f'{metric}_after'] for r in eligible)
                winners.append(dict(benchmark=bench, model=model, metric=metric,
                                    methods=[r['method'] for r in eligible if abs(r[f'{metric}_after']-best)<1e-12], score=best))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'每模型每benchmark每方法_主客观overall及原文变化.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir/'逐样本三次答案均分.jsonl').open('w') as f:
        for k, scores in sorted(units.items()):
            f.write(json.dumps(dict(model=k[0], benchmark=k[1], sample_id=k[2], method=k[3], objective=scores[0], subjective=scores[1]))+'\n')
    lines = ['# 文献方法全量：主客观overall与对应原文比较', '',
             '客观分范围0–1；主观分范围0–5。先将每个样本的3次答案评分取平均，再对样本平均。',
             '每个单元格是“原文 → 优化后（绝对变化）”；不是相对提升百分比。两个指标同等重要，不合并成一个分数。',
             'IF-GEO只使用成功的389个改写及对应原文，缺失11个不填0。各行n是样本数，不是答案次数。',
             '五种固定方法使用相同样本，可以直接比较最优者；两种多步方法预算和成功样本不同，单列且不混合排名。',
             '16个条件做过来源前缀截断，不同方法可能截断不同比例的竞争来源；表中含这些结果，不能全归因于文本优化策略。', '']
    for bench in sorted({r['benchmark'] for r in rows}):
        for model in models:
            lines += [f'## {bench} / {model}', '']
            for label, selected in [('原文及五种固定方法', [ORIGINAL]+FIXED), ('两种多步方法（单独展示）', PIPELINES)]:
                lines += [f'### {label}', '', '| 方法 | n | 客观overall | 主观overall |', '|---|---:|---:|---:|']
                lookup = {r['method']: r for r in rows if r['model']==model and r['benchmark']==bench}
                for method in selected:
                    r = lookup[method]
                    cells = [f"{r[m+'_before']:.4f} → {r[m+'_after']:.4f} ({r[m+'_delta']:+.4f})" for m in ('objective','subjective')]
                    lines.append(f"| {method} | {r['n']} | {' | '.join(cells)} |")
                lines.append('')
    lines += ['## 五种固定方法中的最高均分（不代表显著更好）', '', '| Benchmark | 模型 | 指标 | 方法 | 均分 |', '|---|---|---|---|---:|']
    for r in winners:
        lines.append(f"| {r['benchmark']} | {r['model']} | {r['metric']} | {', '.join(r['methods'])} | {r['score']:.4f} |")
    lines += ['', '这里只展示描述性均值和同样本变化，没有显著性检验，也没有证明迁移方法有效。完整CSV另含各方法得分上升、下降、不变及前后都为0的样本数。']
    (args.output_dir/'文献方法_各模型各benchmark主客观得分对照.md').write_text('\n'.join(lines)+'\n')
    metrics = dict(answer_count=len(obj), sample_model_method_count=len(units), summary_rows=len(rows),
                   paired_baseline=True, exact_key_alignment=True, answer_seeds=[0,1,2],
                   fixed_method_winners=winners, gpu_calls=0)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
