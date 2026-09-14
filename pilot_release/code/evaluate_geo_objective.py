"""Score necessity-pilot answers with official-code-compatible GEO metrics.

The evaluator reads immutable Answer-stage outputs, computes target-source
Word, Position, and Overall/PAWC scores, and pairs every condition with the
same ``model × sample × answer_seed`` Original answer. It writes no repaired
answers and performs no LLM or GPU calls.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geo_objective_metrics import (
    DEFAULT_NLTK_DATA_DIR,
    METRIC_NAMES,
    implementation_provenance,
    paired_improvement,
    parse_answer_official,
    score_parsed_answer,
    sha256_file,
    target_scores,
)


ORIGINAL_PROMPT_ID = "original.no_rewrite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nltk-data-dir", type=Path, default=DEFAULT_NLTK_DATA_DIR)
    parser.add_argument("--answer-prompt-protocol", default="unified_answer_v1")
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--expected-answer-count", type=int, default=None)
    parser.add_argument("--expected-error-count", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def answer_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["seed"]),
    )


def baseline_key(row: dict[str, Any]) -> tuple[str, str, int]:
    seed = row["answer_seed"] if "answer_seed" in row else row["seed"]
    return (str(row["model"]), str(row["sample_id"]), int(seed))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def target_payload(score: dict[str, Any], target_document_index: int) -> dict[str, Any]:
    payload = target_scores(score, target_document_index)
    payload["fallback_used"] = dict(score["fallback_used"])
    payload["raw_target"] = {
        name: float(score["raw"][name][target_document_index])
        for name in METRIC_NAMES
    }
    payload["raw_sum"] = {
        name: float(score["raw_sums"][name]) for name in METRIC_NAMES
    }
    return payload


def main() -> int:
    args = parse_args()
    for path in (args.answers, args.errors, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    manifest_rows = read_jsonl(args.manifest)
    manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Duplicate sample_id values in manifest")
    manifest = {str(row["sample_id"]): row for row in manifest_rows}

    selected_ids = set(args.sample_ids or manifest)
    unknown_selected = sorted(selected_ids - set(manifest))
    if unknown_selected:
        raise ValueError(f"Unknown --sample-ids: {unknown_selected}")
    answers = [
        row
        for row in read_jsonl(args.answers)
        if str(row.get("sample_id")) in selected_ids
    ]
    errors = [
        row
        for row in read_jsonl(args.errors)
        if str(row.get("sample_id")) in selected_ids
    ]
    if args.expected_answer_count is not None and len(answers) != args.expected_answer_count:
        raise ValueError(
            f"Expected {args.expected_answer_count} answers, found {len(answers)}"
        )
    if args.expected_error_count is not None and len(errors) != args.expected_error_count:
        raise ValueError(f"Expected {args.expected_error_count} errors, found {len(errors)}")

    keys = [answer_key(row) for row in answers]
    duplicate_count = len(keys) - len(set(keys))
    protocol_mismatches: list[dict[str, Any]] = []
    metadata_mismatches: list[dict[str, Any]] = []
    scoring_errors: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    group_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for input_row in answers:
        key = answer_key(input_row)
        item = manifest.get(key[1])
        if item is None:
            metadata_mismatches.append({"key": key, "reason": "sample_missing_from_manifest"})
            continue
        if str(input_row.get("answer_prompt_protocol")) != args.answer_prompt_protocol:
            protocol_mismatches.append(
                {
                    "key": key,
                    "observed": input_row.get("answer_prompt_protocol"),
                    "expected": args.answer_prompt_protocol,
                }
            )
        expected_metadata = {
            "benchmark": str(item["benchmark"]),
            "domain": str(item["domain"]),
            "query": str(item["query"]),
        }
        observed_metadata = {
            field: str(input_row.get(field)) for field in expected_metadata
        }
        if observed_metadata != expected_metadata:
            metadata_mismatches.append(
                {"key": key, "expected": expected_metadata, "observed": observed_metadata}
            )
        source_count = int(item["candidate_document_count"])
        target_document_index = int(item["target_document_index"])
        try:
            parsed = parse_answer_official(
                str(input_row.get("answer") or ""),
                nltk_data_dir=args.nltk_data_dir,
            )
            official_score = score_parsed_answer(
                parsed,
                source_count=source_count,
                fallback_policy="official_uniform",
            )
            sensitivity_score = score_parsed_answer(
                parsed,
                source_count=source_count,
                fallback_policy="zero",
            )
            official_target = target_payload(official_score, target_document_index)
            sensitivity_target = target_payload(
                sensitivity_score, target_document_index
            )
            for metric_name in METRIC_NAMES:
                official_sum = sum(official_score["normalized"][metric_name])
                sensitivity_sum = sum(
                    sensitivity_score["normalized"][metric_name]
                )
                if not math.isclose(official_sum, 1.0, abs_tol=1e-12):
                    raise AssertionError(
                        f"Official normalized {metric_name} sum={official_sum}"
                    )
                expected_sensitivity_sum = (
                    0.0
                    if sensitivity_score["fallback_used"][metric_name]
                    else 1.0
                )
                if not math.isclose(
                    sensitivity_sum, expected_sensitivity_sum, abs_tol=1e-12
                ):
                    raise AssertionError(
                        f"Sensitivity normalized {metric_name} sum={sensitivity_sum}"
                    )
            diagnostics = {
                name: official_score[name]
                for name in (
                    "sentence_count",
                    "eligible_word_count",
                    "citation_occurrence_count",
                    "in_range_citation_occurrence_count",
                    "out_of_range_citation_occurrence_count",
                    "zero_citation_occurrence_count",
                    "has_in_range_citation",
                )
            }
            score_rows.append(
                {
                    "experiment_id": args.experiment_id,
                    "metric_protocol": "geo_official_code_compatible_v1",
                    "model": key[0],
                    "sample_id": key[1],
                    "prompt_id": key[2],
                    "answer_seed": key[3],
                    "benchmark": str(item["benchmark"]),
                    "domain": str(item["domain"]),
                    "candidate_document_count": source_count,
                    "target_document_index": target_document_index,
                    "target_citation_index": target_document_index + 1,
                    "answer_cap_hit": bool(input_row.get("cap_hit")),
                    "citation_diagnostics": diagnostics,
                    "official_uniform": official_target,
                    "zero_fallback_sensitivity": sensitivity_target,
                }
            )
        except Exception as exc:
            scoring_errors.append(
                {
                    "key": key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    baselines: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_baseline_keys: list[tuple[str, str, int]] = []
    for row in score_rows:
        if row["prompt_id"] != ORIGINAL_PROMPT_ID:
            continue
        key = baseline_key(row)
        if key in baselines:
            duplicate_baseline_keys.append(key)
        baselines[key] = row

    missing_baselines: list[tuple[str, str, int]] = []
    for row in score_rows:
        base = baselines.get(baseline_key(row))
        if base is None:
            missing_baselines.append(baseline_key(row))
            continue
        row["paired_improvement"] = {}
        for policy_name in ("official_uniform", "zero_fallback_sensitivity"):
            row["paired_improvement"][policy_name] = {
                metric_name: paired_improvement(
                    base[policy_name][metric_name], row[policy_name][metric_name]
                )
                for metric_name in METRIC_NAMES
            }

        group_keys = (
            "overall",
            f"model::{row['model']}",
            f"benchmark::{row['benchmark']}",
            f"model_benchmark::{row['model']}::{row['benchmark']}",
        )
        flags = {
            "score_count": True,
            "cap_hit_count": bool(row["answer_cap_hit"]),
            "no_in_range_citation_count": not bool(
                row["citation_diagnostics"]["has_in_range_citation"]
            ),
            "out_of_range_citation_answer_count": int(
                row["citation_diagnostics"][
                    "out_of_range_citation_occurrence_count"
                ]
            )
            > 0,
            "zero_citation_answer_count": int(
                row["citation_diagnostics"]["zero_citation_occurrence_count"]
            )
            > 0,
        }
        for metric_name in METRIC_NAMES:
            flags[f"official_{metric_name}_fallback_count"] = bool(
                row["official_uniform"]["fallback_used"][metric_name]
            )
            flags[f"official_target_{metric_name}_zero_count"] = (
                row["official_uniform"][metric_name] == 0
            )
            if row["prompt_id"] != ORIGINAL_PROMPT_ID:
                flags[f"{metric_name}_improvement_condition_count"] = True
                flags[f"official_{metric_name}_relative_improvement_undefined_count"] = (
                    row["paired_improvement"]["official_uniform"][metric_name][
                        "relative_percent"
                    ]
                    is None
                )
        for group_key in group_keys:
            for flag_name, enabled in flags.items():
                if enabled:
                    group_counts[group_key][flag_name] += 1

    integrity_passed = not any(
        (
            duplicate_count,
            protocol_mismatches,
            metadata_mismatches,
            scoring_errors,
            duplicate_baseline_keys,
            missing_baselines,
            len(score_rows) != len(answers),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows.sort(
        key=lambda row: (
            row["model"],
            row["sample_id"],
            row["prompt_id"],
            row["answer_seed"],
        )
    )
    (args.output_dir / "objective_scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows),
        encoding="utf-8",
    )

    metrics = {
        "experiment_id": args.experiment_id,
        "phase": "objective_evaluation",
        "method": "geo_official_code_compatible_v1",
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "selected_sample_count": len(selected_ids),
        "selected_sample_ids": sorted(selected_ids) if args.sample_ids else None,
        "input_answer_count": len(answers),
        "input_protocol_exclusion_count": len(errors),
        "objective_score_count": len(score_rows),
        "duplicate_answer_key_count": duplicate_count,
        "protocol_mismatch_count": len(protocol_mismatches),
        "metadata_mismatch_count": len(metadata_mismatches),
        "scoring_error_count": len(scoring_errors),
        "duplicate_baseline_key_count": len(duplicate_baseline_keys),
        "missing_baseline_count": len(set(missing_baselines)),
        "integrity_passed": integrity_passed,
        "gpu_required": False,
        "llm_required": False,
        "raw_answers_modified": False,
        "primary_metric_policy": "official_uniform",
        "sensitivity_metric_policy": "zero_fallback_sensitivity",
        "relative_improvement_zero_baseline_policy": "null_without_epsilon",
        "group_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(group_counts.items())
        },
        "implementation_provenance": implementation_provenance(
            args.nltk_data_dir
        ),
        "input_provenance": {
            "answers": {"path": str(args.answers), "sha256": sha256_file(args.answers)},
            "errors": {"path": str(args.errors), "sha256": sha256_file(args.errors)},
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256_file(args.manifest),
            },
        },
        "result_paths": {
            "objective_scores": str(args.output_dir / "objective_scores.jsonl"),
            "metrics": str(args.output_dir / "metrics.json"),
            "audit": str(args.output_dir / "audit.json"),
        },
        "next_gate": (
            "validate_smoke_then_confirm_full_objective_scoring"
            if args.sample_ids
            else "audit_full_objective_scores_then_freeze_h1_h2_statistics"
        ),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "audit.json",
        {
            "protocol_mismatches_first_100": protocol_mismatches[:100],
            "metadata_mismatches_first_100": metadata_mismatches[:100],
            "scoring_errors_first_100": scoring_errors[:100],
            "duplicate_baseline_keys_first_100": duplicate_baseline_keys[:100],
            "missing_baselines_first_100": sorted(set(missing_baselines))[:100],
            "policy": (
                "Official uniform fallback is primary; zero fallback is a labelled "
                "sensitivity view; zero-baseline relative improvement is null; no "
                "answer is modified, retried, repaired, or filtered."
            ),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if integrity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
