"""Aggregate and audit checkpointed RAID judge trial shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_raid_judge_shard import (
    DIMENSIONS,
    answer_key,
    load_target_map,
    parse_score,
    parse_score_with_method,
    read_jsonl,
    select_shard_answers,
    target_source_is_cited,
    trial_key,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--shard-dir-pattern", default="shard_{index}",
                        choices=["shard_{index}", "shard{index}"],
                        help="Existing shard folder naming; does not move files.")
    parser.add_argument("--single-shard-dir", type=Path, default=None,
                        help="Explicit existing shard directory when num-shards is one.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--limit-answers-per-shard",
        type=int,
        default=None,
        help="Diagnostic-only selection matching each shard runner exactly.",
    )
    parser.add_argument(
        "--reparse-scores-from-answer",
        action="store_true",
        help=(
            "Recompute integer scores from each saved answer while preserving the "
            "original stored score in stored_score."
        ),
    )
    parser.add_argument(
        "--require-all-scores-parsed",
        action="store_true",
        help="Return failure unless every saved Judge answer becomes a score.",
    )
    parser.add_argument(
        "--zero-when-target-not-cited",
        action="store_true",
        help=(
            "Assign score 0 to every dimension/seed when the generated answer "
            "does not contain the exact target citation [index]."
        ),
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if args.single_shard_dir is not None and args.num_shards != 1:
        raise ValueError("--single-shard-dir requires --num-shards 1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Judge seeds must be unique: {args.seeds}")
    if args.limit_answers_per_shard is not None and args.limit_answers_per_shard <= 0:
        raise ValueError("limit-answers-per-shard must be positive")

    all_answers = read_jsonl(args.input)
    answers: list[dict[str, Any]] = []
    selected_answer_counts_per_shard: list[int] = []
    for shard_index in range(args.num_shards):
        shard_answers = select_shard_answers(
            all_answers,
            shard_index=shard_index,
            num_shards=args.num_shards,
            limit_answers=args.limit_answers_per_shard,
        )
        selected_answer_counts_per_shard.append(len(shard_answers))
        answers.extend(shard_answers)
    target_map = load_target_map(args.manifest)
    answer_keys = [answer_key(row) for row in answers]
    if len(answer_keys) != len(set(answer_keys)):
        raise ValueError("Duplicate Answer keys in judge input")
    target_cited_by_answer_key = {
        answer_key(row): target_source_is_cited(
            row, target_map[str(row["sample_id"])]
        )
        for row in answers
    }
    expected = {
        (*key, dimension, judge_seed)
        for key in answer_keys
        for dimension in DIMENSIONS
        for judge_seed in args.seeds
    }
    trials: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    shard_metrics: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard_dir = args.single_shard_dir or args.shards_root / args.shard_dir_pattern.format(index=shard_index)
        trials.extend(read_jsonl(shard_dir / "trials.jsonl"))
        errors.extend(read_jsonl(shard_dir / "errors.jsonl"))
        metrics_path = shard_dir / "metrics.json"
        if metrics_path.exists():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    stored_parsed_score_count = sum(
        row.get("score") is not None for row in trials
    )
    if args.reparse_scores_from_answer:
        normalized_trials = []
        for row in trials:
            if row.get("score_origin") == "deterministic_target_absence":
                score = 0
                parse_method = "deterministic_target_absence"
            else:
                answer = row.get("answer")
                raw = answer if isinstance(answer, str) else ""
                score, parse_method = parse_score_with_method(raw)
            normalized_trials.append(
                {
                    **row,
                    "stored_score": row.get("score"),
                    "score": score,
                    "score_parse_method": parse_method,
                    "score_normalization": "store extracted score as integer",
                }
            )
        trials = normalized_trials

    absent_trials_with_nonzero_pre_rule_score = 0
    normalized_trials = []
    for row in trials:
        key = (
            str(row["model"]),
            str(row["sample_id"]),
            str(row["prompt_id"]),
            int(row["answer_seed"]),
        )
        if key not in target_cited_by_answer_key:
            raise ValueError(f"Trial answer key is missing from input: {key}")
        target_source_cited = target_cited_by_answer_key[key]
        normalized = {
            **row,
            "target_source_cited": target_source_cited,
            "score_origin": row.get("score_origin", "glm_judge"),
        }
        if args.zero_when_target_not_cited and not target_source_cited:
            pre_rule_score = normalized.get("score")
            if pre_rule_score not in (None, 0):
                absent_trials_with_nonzero_pre_rule_score += 1
            normalized.update(
                {
                    "pre_absence_rule_score": pre_rule_score,
                    "score": 0,
                    "score_parse_method": "deterministic_target_absence",
                    "score_origin": "deterministic_target_absence",
                    "score_normalization": "store extracted score as integer",
                }
            )
        normalized_trials.append(normalized)
    trials = normalized_trials
    score_parse_method_counts: dict[str, int] = {}
    for row in trials:
        method = str(row.get("score_parse_method", "missing"))
        score_parse_method_counts[method] = (
            score_parse_method_counts.get(method, 0) + 1
        )

    trial_keys = [trial_key(row) for row in trials]
    error_keys = [trial_key(row) for row in errors]
    trial_set = set(trial_keys)
    error_set = set(error_keys)
    accounted = trial_set | error_set
    duplicate_count = (len(trial_keys) - len(trial_set)) + (
        len(error_keys) - len(error_set)
    )
    overlap = trial_set & error_set
    missing = sorted(expected - accounted)
    unexpected = sorted(accounted - expected)
    target_alignment_errors = []
    for row in trials + errors:
        sample_metadata = target_map.get(str(row["sample_id"]))
        if sample_metadata is None:
            target_alignment_errors.append(
                {"key": trial_key(row), "reason": "sample_missing_from_manifest"}
            )
            continue
        expected_target = {
            "target_document_index": sample_metadata["target_document_index"],
            "target_citation_index": sample_metadata["target_citation_index"],
            "candidate_document_count": sample_metadata["candidate_document_count"],
        }
        observed = {
            "target_document_index": int(row["target_document_index"]),
            "target_citation_index": int(row["target_citation_index"]),
            "candidate_document_count": int(row["candidate_document_count"]),
        }
        if observed != expected_target:
            target_alignment_errors.append(
                {
                    "key": trial_key(row),
                    "expected": expected_target,
                    "observed": observed,
                }
            )
    complete_accounting = (
        len(accounted) == len(expected)
        and not missing
        and not unexpected
        and not overlap
        and duplicate_count == 0
        and not target_alignment_errors
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.sort(key=trial_key)
    errors.sort(key=trial_key)
    (args.output_dir / "judge_trials.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trials),
        encoding="utf-8",
    )
    (args.output_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )

    grouped: dict[
        tuple[str, str, str, int], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in trials:
        grouped[
            (
                str(row["model"]),
                str(row["sample_id"]),
                str(row["prompt_id"]),
                int(row["answer_seed"]),
            )
        ][str(row["dimension"])].append(row)

    scores: list[dict[str, Any]] = []
    scorable_dimensions = 0
    for answer in answers:
        key = answer_key(answer)
        dimensions: dict[str, Any] = {}
        means: list[float] = []
        for dimension in DIMENSIONS:
            dimension_trials = sorted(
                grouped[key].get(dimension, []), key=lambda row: row["judge_seed"]
            )
            valid_scores = [
                int(row["score"])
                for row in dimension_trials
                if row.get("score") is not None
            ]
            fully_scorable = (
                len(dimension_trials) == len(args.seeds)
                and len(valid_scores) == len(args.seeds)
            )
            mean_score = (
                sum(valid_scores) / len(valid_scores) if valid_scores else None
            )
            if fully_scorable:
                scorable_dimensions += 1
            if mean_score is not None:
                means.append(mean_score)
            dimensions[dimension] = {
                "trials": dimension_trials,
                "valid_scores": valid_scores,
                "mean_score": mean_score,
                "scorable": fully_scorable,
            }
        all_dimensions_scorable = all(
            dimensions[dimension]["scorable"] for dimension in DIMENSIONS
        )
        scores.append(
            {
                "model": key[0],
                "sample_id": key[1],
                "prompt_id": key[2],
                "answer_seed": key[3],
                "benchmark": answer["benchmark"],
                "domain": answer["domain"],
                **target_map[str(answer["sample_id"])],
                "target_source_cited": target_cited_by_answer_key[key],
                "dimensions": dimensions,
                "subjective_average": sum(means) / len(means) if means else None,
                "all_dimensions_scorable": all_dimensions_scorable,
            }
        )
    scores.sort(
        key=lambda row: (
            row["model"],
            row["sample_id"],
            row["prompt_id"],
            row["answer_seed"],
        )
    )
    (args.output_dir / "judge_scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scores),
        encoding="utf-8",
    )
    parsed_trials = sum(row.get("score") is not None for row in trials)
    all_scores_parsed = parsed_trials == len(trials)
    integrity_passed = complete_accounting and (
        all_scores_parsed or not args.require_all_scores_parsed
    )
    total_dimension_scores = len(answers) * len(DIMENSIONS)
    fully_scorable_answers = sum(
        bool(row["all_dimensions_scorable"]) for row in scores
    )
    target_cited_answer_count = sum(target_cited_by_answer_key.values())
    target_absent_answer_count = len(answers) - target_cited_answer_count
    glm_judge_trial_count = sum(
        row.get("score_origin") == "glm_judge" for row in trials
    )
    deterministic_zero_trial_count = sum(
        row.get("score_origin") == "deterministic_target_absence"
        for row in trials
    )
    metrics = {
        "experiment_id": args.experiment_id,
        "input_answer_count": len(all_answers),
        "answer_count": len(answers),
        "selected_answer_counts_per_shard": selected_answer_counts_per_shard,
        "limit_answers_per_shard": args.limit_answers_per_shard,
        "dimension_count": len(DIMENSIONS),
        "seeds": args.seeds,
        "expected_trials": len(expected),
        "completed_trials": len(trials),
        "generation_error_count": len(errors),
        "parsed_score_count": parsed_trials,
        "unparsed_score_count": len(trials) - parsed_trials,
        "all_scores_parsed": all_scores_parsed,
        "require_all_scores_parsed": args.require_all_scores_parsed,
        "integrity_passed": integrity_passed,
        "total_dimension_scores": total_dimension_scores,
        "scorable_dimension_scores": scorable_dimensions,
        "unscorable_dimension_scores": total_dimension_scores
        - scorable_dimensions,
        "fully_scorable_answer_count": fully_scorable_answers,
        "duplicate_count": duplicate_count,
        "trial_error_overlap_count": len(overlap),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "target_alignment_error_count": len(target_alignment_errors),
        "target_cited_answer_count": target_cited_answer_count,
        "target_absent_answer_count": target_absent_answer_count,
        "zero_when_target_not_cited": args.zero_when_target_not_cited,
        "target_absence_policy": (
            "if exact [target citation index] is absent, assign 0 to all "
            "dimensions/seeds without calling GLM"
            if args.zero_when_target_not_cited
            else "disabled"
        ),
        "glm_judge_trial_count": glm_judge_trial_count,
        "deterministic_zero_trial_count": deterministic_zero_trial_count,
        "saved_glm_call_count": deterministic_zero_trial_count,
        "absent_trials_with_nonzero_pre_rule_score": (
            absent_trials_with_nonzero_pre_rule_score
        ),
        "complete_accounting": complete_accounting,
        "checkpointed_parallel_shards": True,
        "reparse_scores_from_answer": args.reparse_scores_from_answer,
        "stored_parsed_score_count": stored_parsed_score_count,
        "score_parse_method_counts": score_parse_method_counts,
        "score_output_protocol": "extract one unambiguous integer score from 0-5",
        "score_normalization": "store extracted score as integer",
        "judge_model": "ZhipuAI/GLM-4-9B-0414",
        "shard_metrics": shard_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "missing_first_100": missing[:100],
                "unexpected_first_100": unexpected[:100],
                "trial_error_overlap_first_100": sorted(overlap)[:100],
                "target_alignment_errors_first_100": target_alignment_errors[:100],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if integrity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
