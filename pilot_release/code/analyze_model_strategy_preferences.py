#!/usr/bin/env python3
"""Exploratory model-specific GEO strategy preference analysis.

This script intentionally performs descriptive analysis only.  It averages the
three answer-generation seeds within each benchmark sample/model/prompt cell,
pairs every prompt with the matching Original score, and reports rankings and
rank reversals without turning p-values or multiplicity corrections into a
pilot gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ORIGINAL_PROMPT_ID = "original.no_rewrite"
NEUTRAL_PROMPT_ID = "neutral_rewrite_control.neutral_rewrite_control"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ci(
    values: list[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty list")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def answer_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["answer_seed"]),
    )


def load_score_map(
    path: Path,
    *,
    score_getter: Any,
    label: str,
) -> tuple[dict[tuple[str, str, str, int], float], int]:
    scores: dict[tuple[str, str, str, int], float] = {}
    row_count = 0
    for row in read_jsonl(path):
        row_count += 1
        key = answer_key(row)
        if key in scores:
            raise ValueError(f"Duplicate {label} key: {key}")
        value = float(score_getter(row))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {label} score for {key}: {value}")
        scores[key] = value
    return scores, row_count


def prompt_family(prompt_id: str) -> str:
    return prompt_id.split(".", 1)[0]


def build_units(
    objective: dict[tuple[str, str, str, int], float],
    subjective: dict[tuple[str, str, str, int], float],
    manifest: dict[str, dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    prompt_ids: tuple[str, ...],
    original_prompt_id: str = ORIGINAL_PROMPT_ID,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    if set(objective) != set(subjective):
        only_objective = len(set(objective) - set(subjective))
        only_subjective = len(set(subjective) - set(objective))
        raise ValueError(
            "Objective/subjective keys differ: "
            f"objective_only={only_objective}, subjective_only={only_subjective}"
        )

    expected_seed_set = set(seeds)
    grouped: defaultdict[tuple[str, str, str], dict[int, tuple[float, float]]] = (
        defaultdict(dict)
    )
    for key, objective_value in objective.items():
        model, sample_id, prompt_id, seed = key
        if model not in models:
            raise ValueError(f"Unexpected model in scores: {model}")
        if sample_id not in manifest:
            raise ValueError(f"Score sample missing from manifest: {sample_id}")
        if prompt_id not in prompt_ids:
            raise ValueError(f"Unexpected prompt in scores: {prompt_id}")
        grouped[(model, sample_id, prompt_id)][seed] = (
            objective_value,
            subjective[key],
        )

    partial_seed_groups: list[tuple[str, str, str]] = []
    units: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model, sample_id, prompt_id = key
        seed_values = grouped[key]
        if set(seed_values) != expected_seed_set:
            partial_seed_groups.append(key)
            continue
        item = manifest[sample_id]
        units.append(
            {
                "model": model,
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "prompt_family": prompt_family(prompt_id),
                "benchmark": str(item["benchmark"]),
                "domain": str(item.get("domain", "")),
                "objective_raw": statistics.fmean(
                    seed_values[seed][0] for seed in seeds
                ),
                "subjective_raw": statistics.fmean(
                    seed_values[seed][1] for seed in seeds
                ),
            }
        )
    if partial_seed_groups:
        raise ValueError(
            f"Found {len(partial_seed_groups)} cells with incomplete seed sets; "
            f"first={partial_seed_groups[0]}"
        )

    baseline = {
        (row["model"], row["sample_id"]): row
        for row in units
        if row["prompt_id"] == original_prompt_id
    }
    for row in units:
        base = baseline.get((row["model"], row["sample_id"]))
        if base is None:
            raise ValueError(
                "Missing Original baseline for "
                f"{row['model']} / {row['sample_id']}"
            )
        row["objective_delta"] = row["objective_raw"] - base["objective_raw"]
        row["subjective_delta"] = row["subjective_raw"] - base["subjective_raw"]

    complete_samples: set[str] = set()
    for sample_id in manifest:
        required = {
            (model, sample_id, prompt_id)
            for model in models
            for prompt_id in prompt_ids
        }
        if required.issubset(grouped):
            complete_samples.add(sample_id)
    for row in units:
        row["is_common_complete"] = row["sample_id"] in complete_samples

    audit = {
        "seed_averaged_unit_count": len(units),
        "partial_seed_group_count": len(partial_seed_groups),
        "common_complete_sample_count": len(complete_samples),
        "excluded_from_common_complete_sample_count": len(manifest)
        - len(complete_samples),
        "excluded_from_common_complete_sample_ids": sorted(
            set(manifest) - complete_samples
        ),
    }
    return units, complete_samples, audit


def summarize_units(
    units: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    original_prompt_id: str,
    neutral_prompt_id: str,
) -> list[dict[str, Any]]:
    analysis_sets = {
        "common_complete": [row for row in units if row["is_common_complete"]],
        "full_available": units,
    }
    summaries: list[dict[str, Any]] = []
    for analysis_set, selected in analysis_sets.items():
        groups: defaultdict[
            tuple[str, str, str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in selected:
            for outcome in ("objective", "subjective"):
                groups[
                    (
                        row["benchmark"],
                        outcome,
                        row["model"],
                        row["prompt_id"],
                    )
                ].append(row)
        for key in sorted(groups):
            benchmark, outcome, model, prompt_id = key
            rows = groups[key]
            raw_values = [float(row[f"{outcome}_raw"]) for row in rows]
            delta_values = [float(row[f"{outcome}_delta"]) for row in rows]
            raw_low, raw_high = bootstrap_ci(
                raw_values,
                resamples=bootstrap_resamples,
                seed=stable_seed(bootstrap_seed, f"raw:{analysis_set}:{key}"),
            )
            delta_low, delta_high = bootstrap_ci(
                delta_values,
                resamples=bootstrap_resamples,
                seed=stable_seed(bootstrap_seed, f"delta:{analysis_set}:{key}"),
            )
            summaries.append(
                {
                    "analysis_set": analysis_set,
                    "benchmark": benchmark,
                    "outcome": outcome,
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt_family": prompt_family(prompt_id),
                    "is_control": prompt_id
                    in {original_prompt_id, neutral_prompt_id},
                    "n_samples": len(rows),
                    "mean_raw": statistics.fmean(raw_values),
                    "raw_ci_low": raw_low,
                    "raw_ci_high": raw_high,
                    "mean_delta_vs_original": statistics.fmean(delta_values),
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                }
            )
    return summaries


def build_rankings(
    summaries: list[dict[str, Any]],
    *,
    control_prompt_ids: set[str],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in summaries:
        if row["prompt_id"] in control_prompt_ids:
            continue
        groups[
            (
                row["analysis_set"],
                row["benchmark"],
                row["outcome"],
                row["model"],
            )
        ].append(row)
    rankings: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda row: (-row["mean_delta_vs_original"], row["prompt_id"]),
        )
        for rank, row in enumerate(ordered, start=1):
            rankings.append({**row, "rank": rank, "is_top_strategy": rank == 1})
    return rankings


def find_rank_reversals(
    summaries: list[dict[str, Any]],
    *,
    control_prompt_ids: set[str],
) -> list[dict[str, Any]]:
    index = {
        (
            row["analysis_set"],
            row["benchmark"],
            row["outcome"],
            row["model"],
            row["prompt_id"],
        ): row
        for row in summaries
        if row["prompt_id"] not in control_prompt_ids
    }
    prefixes = sorted({key[:3] for key in index})
    reversals: list[dict[str, Any]] = []
    for analysis_set, benchmark, outcome in prefixes:
        models = sorted(
            {
                key[3]
                for key in index
                if key[:3] == (analysis_set, benchmark, outcome)
            }
        )
        prompts = sorted(
            {
                key[4]
                for key in index
                if key[:3] == (analysis_set, benchmark, outcome)
            }
        )
        for model_a, model_b in combinations(models, 2):
            for prompt_a, prompt_b in combinations(prompts, 2):
                keys = [
                    (analysis_set, benchmark, outcome, model_a, prompt_a),
                    (analysis_set, benchmark, outcome, model_a, prompt_b),
                    (analysis_set, benchmark, outcome, model_b, prompt_a),
                    (analysis_set, benchmark, outcome, model_b, prompt_b),
                ]
                if any(key not in index for key in keys):
                    continue
                preference_a = (
                    index[keys[0]]["mean_delta_vs_original"]
                    - index[keys[1]]["mean_delta_vs_original"]
                )
                preference_b = (
                    index[keys[2]]["mean_delta_vs_original"]
                    - index[keys[3]]["mean_delta_vs_original"]
                )
                if preference_a * preference_b >= 0:
                    continue
                reversals.append(
                    {
                        "analysis_set": analysis_set,
                        "benchmark": benchmark,
                        "outcome": outcome,
                        "model_a": model_a,
                        "model_b": model_b,
                        "prompt_a": prompt_a,
                        "prompt_b": prompt_b,
                        "model_a_prompt_a_minus_b": preference_a,
                        "model_b_prompt_a_minus_b": preference_b,
                        "reversal_strength_min_abs": min(
                            abs(preference_a), abs(preference_b)
                        ),
                        "difference_of_differences_abs": abs(
                            preference_a - preference_b
                        ),
                    }
                )
    return sorted(
        reversals,
        key=lambda row: (
            row["analysis_set"],
            row["benchmark"],
            row["outcome"],
            -row["reversal_strength_min_abs"],
            row["model_a"],
            row["model_b"],
            row["prompt_a"],
            row["prompt_b"],
        ),
    )


def heat_color(value: float, maximum: float) -> str:
    if maximum <= 0:
        return "rgba(128,128,128,0.08)"
    intensity = min(abs(value) / maximum, 1.0)
    alpha = 0.12 + 0.58 * intensity
    if value >= 0:
        return f"rgba(25,135,84,{alpha:.3f})"
    return f"rgba(220,53,69,{alpha:.3f})"


def render_heatmaps(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    control_prompt_ids: set[str],
) -> None:
    selected = [
        row for row in summaries if row["prompt_id"] not in control_prompt_ids
    ]
    sections: list[str] = []
    for analysis_set in ("common_complete", "full_available"):
        for benchmark in sorted({row["benchmark"] for row in selected}):
            for outcome in ("objective", "subjective"):
                rows = [
                    row
                    for row in selected
                    if row["analysis_set"] == analysis_set
                    and row["benchmark"] == benchmark
                    and row["outcome"] == outcome
                ]
                if not rows:
                    continue
                models = sorted({row["model"] for row in rows})
                prompts = sorted({row["prompt_id"] for row in rows})
                lookup = {
                    (row["model"], row["prompt_id"]): row for row in rows
                }
                maximum = max(
                    abs(row["mean_delta_vs_original"]) for row in rows
                )
                header = "".join(f"<th>{html.escape(model)}</th>" for model in models)
                body_rows = []
                for prompt_id in prompts:
                    cells = []
                    for model in models:
                        row = lookup[(model, prompt_id)]
                        value = row["mean_delta_vs_original"]
                        title = (
                            f"95% bootstrap interval: "
                            f"[{row['delta_ci_low']:.5f}, {row['delta_ci_high']:.5f}]"
                        )
                        cells.append(
                            f'<td style="background:{heat_color(value, maximum)}" '
                            f'title="{html.escape(title)}">{value:+.5f}</td>'
                        )
                    body_rows.append(
                        f"<tr><th>{html.escape(prompt_id)}</th>{''.join(cells)}</tr>"
                    )
                sections.append(
                    f"<h2>{html.escape(analysis_set)} · {html.escape(benchmark)} · "
                    f"{html.escape(outcome)}</h2>"
                    f"<table><thead><tr><th>Prompt variant</th>{header}</tr></thead>"
                    f"<tbody>{''.join(body_rows)}</tbody></table>"
                )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>TransferGEO 模型策略偏好热力图</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#202124}}
table{{border-collapse:collapse;margin:12px 0 34px;width:100%}}
th,td{{border:1px solid #d9dce1;padding:8px 10px;text-align:right}}
th:first-child{{text-align:left;max-width:520px;word-break:break-word}}
h1{{margin-bottom:8px}} .note{{color:#5f6368;margin-bottom:28px}}
</style></head><body>
<h1>模型策略偏好热力图</h1>
<p class="note">单元格为三个 answer seeds 先聚合后，相对同模型同样本 Original 的平均绝对变化。绿色为提升，红色为下降；悬停查看95%样本bootstrap区间。本图仅作探索性分析。</p>
{''.join(sections)}
</body></html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--objective-scores", type=Path, required=True)
    parser.add_argument("--subjective-scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen", "llama", "mistral"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--objective-policy", default="zero_fallback_sensitivity")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--expected-score-count", type=int)
    parser.add_argument("--expected-manifest-count", type=int)
    parser.add_argument("--expected-prompt-count", type=int)
    parser.add_argument("--expected-common-sample-count", type=int)
    args = parser.parse_args()

    manifest_rows = list(read_jsonl(args.manifest))
    manifest: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        if sample_id in manifest:
            raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
        manifest[sample_id] = row
    if (
        args.expected_manifest_count is not None
        and len(manifest) != args.expected_manifest_count
    ):
        raise ValueError(
            f"Expected {args.expected_manifest_count} manifest rows, found {len(manifest)}"
        )

    objective, objective_count = load_score_map(
        args.objective_scores,
        score_getter=lambda row: row[args.objective_policy]["overall"],
        label="objective",
    )
    subjective, subjective_count = load_score_map(
        args.subjective_scores,
        score_getter=lambda row: row["subjective_average"],
        label="subjective",
    )
    if args.expected_score_count is not None:
        if objective_count != args.expected_score_count:
            raise ValueError(
                f"Expected {args.expected_score_count} objective rows, found {objective_count}"
            )
        if subjective_count != args.expected_score_count:
            raise ValueError(
                f"Expected {args.expected_score_count} subjective rows, found {subjective_count}"
            )

    for key, value in objective.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Objective score outside [0,1] for {key}: {value}")
    for key, value in subjective.items():
        if not 0.0 <= value <= 5.0:
            raise ValueError(f"Subjective score outside [0,5] for {key}: {value}")

    prompt_ids = tuple(sorted({key[2] for key in objective}))
    if args.expected_prompt_count is not None and len(prompt_ids) != args.expected_prompt_count:
        raise ValueError(
            f"Expected {args.expected_prompt_count} prompts, found {len(prompt_ids)}"
        )
    if ORIGINAL_PROMPT_ID not in prompt_ids or NEUTRAL_PROMPT_ID not in prompt_ids:
        raise ValueError("Original or Neutral Rewrite control is missing")

    units, complete_samples, unit_audit = build_units(
        objective,
        subjective,
        manifest,
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        prompt_ids=prompt_ids,
    )
    if (
        args.expected_common_sample_count is not None
        and len(complete_samples) != args.expected_common_sample_count
    ):
        raise ValueError(
            f"Expected {args.expected_common_sample_count} common-complete samples, "
            f"found {len(complete_samples)}"
        )

    summaries = summarize_units(
        units,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        original_prompt_id=ORIGINAL_PROMPT_ID,
        neutral_prompt_id=NEUTRAL_PROMPT_ID,
    )
    controls = {ORIGINAL_PROMPT_ID, NEUTRAL_PROMPT_ID}
    rankings = build_rankings(summaries, control_prompt_ids=controls)
    top_strategies = [row for row in rankings if row["is_top_strategy"]]
    reversals = find_rank_reversals(summaries, control_prompt_ids=controls)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "analysis_set",
        "benchmark",
        "outcome",
        "model",
        "prompt_id",
        "prompt_family",
        "is_control",
        "n_samples",
        "mean_raw",
        "raw_ci_low",
        "raw_ci_high",
        "mean_delta_vs_original",
        "delta_ci_low",
        "delta_ci_high",
    ]
    ranking_fields = summary_fields + ["rank", "is_top_strategy"]
    reversal_fields = [
        "analysis_set",
        "benchmark",
        "outcome",
        "model_a",
        "model_b",
        "prompt_a",
        "prompt_b",
        "model_a_prompt_a_minus_b",
        "model_b_prompt_a_minus_b",
        "reversal_strength_min_abs",
        "difference_of_differences_abs",
    ]
    unit_fields = [
        "model",
        "sample_id",
        "benchmark",
        "domain",
        "prompt_id",
        "prompt_family",
        "is_common_complete",
        "objective_raw",
        "objective_delta",
        "subjective_raw",
        "subjective_delta",
    ]
    write_csv(args.output_dir / "seed_averaged_scores.csv", units, unit_fields)
    write_csv(args.output_dir / "strategy_score_summary.csv", summaries, summary_fields)
    write_csv(args.output_dir / "strategy_rankings.csv", rankings, ranking_fields)
    write_csv(args.output_dir / "top_strategies.csv", top_strategies, ranking_fields)
    write_csv(args.output_dir / "rank_reversals.csv", reversals, reversal_fields)
    render_heatmaps(
        args.output_dir / "preference_heatmaps.html",
        summaries,
        control_prompt_ids=controls,
    )

    top_payload = [
        {
            "analysis_set": row["analysis_set"],
            "benchmark": row["benchmark"],
            "outcome": row["outcome"],
            "model": row["model"],
            "prompt_id": row["prompt_id"],
            "mean_delta_vs_original": row["mean_delta_vs_original"],
            "delta_ci_low": row["delta_ci_low"],
            "delta_ci_high": row["delta_ci_high"],
        }
        for row in top_strategies
    ]
    reversal_counts: defaultdict[str, int] = defaultdict(int)
    for row in reversals:
        reversal_counts[
            f"{row['analysis_set']}::{row['benchmark']}::{row['outcome']}"
        ] += 1
    metrics = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "analysis_protocol": "descriptive_model_strategy_preference_v1",
        "exploratory_only": True,
        "objective_score": f"{args.objective_policy}.overall",
        "subjective_score": "subjective_average",
        "baseline": ORIGINAL_PROMPT_ID,
        "neutral_control": NEUTRAL_PROMPT_ID,
        "models": args.models,
        "answer_seeds": args.seeds,
        "bootstrap": {
            "unit": "benchmark_sample",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
            "role": "uncertainty_display_not_significance_gate",
        },
        "input_counts": {
            "manifest": len(manifest),
            "objective": objective_count,
            "subjective": subjective_count,
        },
        "prompt_ids": list(prompt_ids),
        "target_strategy_prompt_count": len(prompt_ids) - len(controls),
        "unit_audit": unit_audit,
        "summary_row_count": len(summaries),
        "ranking_row_count": len(rankings),
        "top_strategy_row_count": len(top_strategies),
        "rank_reversal_row_count": len(reversals),
        "rank_reversal_counts": dict(sorted(reversal_counts.items())),
        "top_strategies": top_payload,
        "outputs": [
            "seed_averaged_scores.csv",
            "strategy_score_summary.csv",
            "strategy_rankings.csv",
            "top_strategies.csv",
            "rank_reversals.csv",
            "preference_heatmaps.html",
            "report.md",
            "metrics.json",
        ],
        "interpretation_boundary_cn": (
            "仅用于探索不同答案模型是否呈现不同GEO策略排名和排序反转；"
            "不使用显著性、Holm、多单元硬门或H2迁移regret自动判定论文claim。"
        ),
    }
    report_lines = [
        "# 模型策略偏好探索性前置分析",
        "",
        "本报告只展示不同答案模型的策略得分、排名和排序反转，不执行复杂的确认性通过门槛。",
        "",
        f"- Objective输入：{objective_count:,}条，使用`{args.objective_policy}.overall`。",
        f"- Subjective输入：{subjective_count:,}条，使用`subjective_average`。",
        f"- 三seed聚合后的模型—样本—Prompt单元：{len(units):,}个。",
        f"- 三模型共同完整样本：{len(complete_samples):,}/{len(manifest):,}。",
        f"- 被共同完整主表排除的样本：{', '.join(unit_audit['excluded_from_common_complete_sample_ids']) or '无'}。",
        f"- 主表策略排序记录：{sum(row['analysis_set'] == 'common_complete' for row in rankings):,}条。",
        f"- 主表观察到的策略对排序反转：{sum(row['analysis_set'] == 'common_complete' for row in reversals):,}条。",
        "",
        "## 主表Top-1策略",
        "",
        "| Benchmark | Outcome | Model | Top-1 Prompt | 相对Original均值 | 95%区间 |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in top_strategies:
        if row["analysis_set"] != "common_complete":
            continue
        report_lines.append(
            f"| {row['benchmark']} | {row['outcome']} | {row['model']} | "
            f"`{row['prompt_id']}` | {row['mean_delta_vs_original']:+.5f} | "
            f"[{row['delta_ci_low']:+.5f}, {row['delta_ci_high']:+.5f}] |"
        )
    report_lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "是否值得进入迁移方法阶段，需要结合Top-1是否跨模型变化、整体排名是否不同、排序反转强度及误差区间人工判断。这里不自动给出显著性结论，也不证明迁移方法有效。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
