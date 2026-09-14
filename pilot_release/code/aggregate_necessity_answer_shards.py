"""Aggregate and audit checkpointed per-model necessity Answer shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_prompt import (
    ANSWER_PROMPT_PROTOCOLS,
    BENCHMARK_OFFICIAL_V1,
    UNIFIED_ANSWER_V3,
    UNIFIED_ANSWER_V4,
    UNIFIED_ANSWER_V5,
    UNIFIED_ANSWER_V6,
    UNIFIED_ANSWER_V7,
    UNIFIED_ANSWER_V8,
    UNIFIED_ANSWER_V9,
)
from model_loader import MODEL_SPECS
from run_necessity_rewrite_stage import PROMPT_ORDER


ORIGINAL_PROMPT_ID = "original.no_rewrite"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["seed"]),
    )


def summarize_citation_diagnostics(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize stored diagnostics without rescoring or editing any answer."""

    basic = [
        row.get("citation_diagnostics")
        for row in rows
        if isinstance(row.get("citation_diagnostics"), dict)
    ]
    strict = [
        row.get("v2_citation_contract")
        for row in rows
        if isinstance(row.get("v2_citation_contract"), dict)
    ]
    optional = [
        row.get("v4_citation_contract")
        for row in rows
        if isinstance(row.get("v4_citation_contract"), dict)
    ]
    v6_strict = [
        row.get("v6_citation_contract")
        for row in rows
        if isinstance(row.get("v6_citation_contract"), dict)
    ]
    return {
        "answer_count": len(rows),
        "stored_basic_diagnostic_count": len(basic),
        "stored_strict_diagnostic_count": len(strict),
        "stored_optional_citation_diagnostic_count": len(optional),
        "stored_v6_strict_citation_diagnostic_count": len(v6_strict),
        "has_numeric_citation_count": sum(
            bool(item.get("has_numeric_citation")) for item in basic
        ),
        "citation_format_scorable_count": sum(
            bool(item.get("citation_format_scorable")) for item in basic
        ),
        "invalid_numeric_citation_answer_count": sum(
            bool(item.get("invalid_numeric_citations")) for item in basic
        ),
        "citationless_sentence_answer_count": sum(
            int(item.get("citationless_sentence_count", 0)) > 0 for item in basic
        ),
        "strict_citation_contract_compliant_count": sum(
            bool(item.get("contract_compliant")) for item in strict
        ),
        "strict_citation_contract_failure_count": sum(
            not bool(item.get("contract_compliant")) for item in strict
        ),
        "optional_citation_contract_compliant_count": sum(
            bool(item.get("contract_compliant")) for item in optional
        ),
        "optional_citation_contract_failure_count": sum(
            not bool(item.get("contract_compliant")) for item in optional
        ),
        "citationless_answer_count": sum(
            bool(item.get("citationless_answer")) for item in optional
        ),
        "malformed_citation_answer_count": sum(
            int(item.get("malformed_citation_sentence_count", 0)) > 0
            for item in optional
        ),
        "separate_reference_section_answer_count": sum(
            bool(item.get("separate_reference_section_detected"))
            for item in optional
        ),
        "v6_strict_citation_contract_compliant_count": sum(
            bool(item.get("contract_compliant")) for item in v6_strict
        ),
        "v6_strict_citation_contract_failure_count": sum(
            not bool(item.get("contract_compliant")) for item in v6_strict
        ),
        "v6_uncited_sentence_answer_count": sum(
            int(item.get("uncited_sentence_count", 0)) > 0
            for item in v6_strict
        ),
    }


def summarize_error_types(
    rows: list[dict[str, Any]],
    *,
    allowed_error_types: list[str],
) -> tuple[dict[str, int], list[str]]:
    """Separate pre-declared protocol exclusions from unexpected errors."""

    counts = Counter(str(row.get("error_type", "unknown")) for row in rows)
    unexpected = sorted(set(counts) - set(allowed_error_types))
    return dict(sorted(counts.items())), unexpected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-id",
        default="necessity_answer_stage_v1",
    )
    parser.add_argument(
        "--answer-prompt-protocol",
        choices=ANSWER_PROMPT_PROTOCOLS,
        default=BENCHMARK_OFFICIAL_V1,
    )
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allowed-error-types", nargs="*", default=[])
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    prompt_ids = (ORIGINAL_PROMPT_ID, *PROMPT_ORDER)
    expected = {
        (model, str(item["sample_id"]), prompt_id, seed)
        for model in MODEL_SPECS
        for item in manifest
        for prompt_id in prompt_ids
        for seed in args.seeds
    }
    answers: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    shard_metrics: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        shard_dir = args.shards_root / model
        answers.extend(read_jsonl(shard_dir / "answers.jsonl"))
        errors.extend(read_jsonl(shard_dir / "errors.jsonl"))
        metrics_path = shard_dir / "metrics.json"
        if metrics_path.exists():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    protocol_mismatches = [
        key(row)
        for row in answers + errors
        if str(row.get("answer_prompt_protocol", BENCHMARK_OFFICIAL_V1))
        != args.answer_prompt_protocol
    ]
    if protocol_mismatches:
        raise ValueError(
            "Shard outputs contain rows from another answer prompt protocol: "
            f"{protocol_mismatches[:10]}"
        )

    answer_keys = [key(row) for row in answers]
    error_keys = [key(row) for row in errors]
    answer_set = set(answer_keys)
    error_set = set(error_keys)
    accounted = answer_set | error_set
    duplicates = (len(answer_keys) - len(answer_set)) + (
        len(error_keys) - len(error_set)
    )
    overlap = answer_set & error_set
    missing = sorted(expected - accounted)
    unexpected = sorted(accounted - expected)
    error_type_counts, unexpected_error_types = summarize_error_types(
        errors,
        allowed_error_types=args.allowed_error_types,
    )
    missing_basic_diagnostics = [
        key(row)
        for row in answers
        if not isinstance(row.get("citation_diagnostics"), dict)
    ]
    missing_strict_diagnostics = [
        key(row)
        for row in answers
        if not isinstance(row.get("v2_citation_contract"), dict)
    ]
    missing_optional_diagnostics = [
        key(row)
        for row in answers
        if not isinstance(row.get("v4_citation_contract"), dict)
    ]
    missing_v6_strict_diagnostics = [
        key(row)
        for row in answers
        if not isinstance(row.get("v6_citation_contract"), dict)
    ]
    diagnostic_storage_complete = (
        not missing_basic_diagnostics
        and (
            args.answer_prompt_protocol != UNIFIED_ANSWER_V3
            or not missing_strict_diagnostics
        )
        and (
            args.answer_prompt_protocol
            not in {
                UNIFIED_ANSWER_V4,
                UNIFIED_ANSWER_V5,
                UNIFIED_ANSWER_V8,
                UNIFIED_ANSWER_V9,
            }
            or not missing_optional_diagnostics
        )
        and (
            args.answer_prompt_protocol
            not in {UNIFIED_ANSWER_V6, UNIFIED_ANSWER_V7}
            or not missing_v6_strict_diagnostics
        )
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers.sort(key=key)
    errors.sort(key=key)
    (args.output_dir / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answers),
        encoding="utf-8",
    )
    (args.output_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )
    complete_accounting = (
        len(accounted) == len(expected)
        and not missing
        and not unexpected
        and not overlap
        and duplicates == 0
    )
    all_successful = complete_accounting and not errors
    execution_gate_passed = (
        complete_accounting
        and not unexpected_error_types
        and diagnostic_storage_complete
    )
    citation_summary = summarize_citation_diagnostics(answers)
    metrics = {
        "experiment_id": args.experiment_id,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "manifest_count": len(manifest),
        "condition_count": len(manifest) * len(prompt_ids),
        "models": list(MODEL_SPECS),
        "seeds": args.seeds,
        "expected_generations": len(expected),
        "completed_generations": len(answers),
        "error_count": len(errors),
        "accounted_generations": len(accounted),
        "duplicate_count": duplicates,
        "success_error_overlap_count": len(overlap),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete_accounting": complete_accounting,
        "all_successful": all_successful,
        "allowed_error_types": sorted(set(args.allowed_error_types)),
        "error_type_counts": error_type_counts,
        "unexpected_error_types": unexpected_error_types,
        "diagnostic_storage_complete": diagnostic_storage_complete,
        "missing_basic_citation_diagnostic_count": len(
            missing_basic_diagnostics
        ),
        "missing_strict_citation_diagnostic_count": len(
            missing_strict_diagnostics
        ),
        "missing_optional_citation_diagnostic_count": len(
            missing_optional_diagnostics
        ),
        "missing_v6_strict_citation_diagnostic_count": len(
            missing_v6_strict_diagnostics
        ),
        "citation_summary": citation_summary,
        "execution_gate_passed": execution_gate_passed,
        "status": (
            "passed"
            if execution_gate_passed and not errors
            else "passed_with_protocol_exclusions"
            if execution_gate_passed
            else "failed"
        ),
        "checkpointed_parallel_models": True,
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
                "success_error_overlap_first_100": sorted(overlap)[:100],
                "unexpected_error_types": unexpected_error_types,
                "missing_basic_citation_diagnostics_first_100": (
                    missing_basic_diagnostics[:100]
                ),
                "missing_strict_citation_diagnostics_first_100": (
                    missing_strict_diagnostics[:100]
                ),
                "missing_optional_citation_diagnostics_first_100": (
                    missing_optional_diagnostics[:100]
                ),
                "missing_v6_strict_citation_diagnostics_first_100": (
                    missing_v6_strict_diagnostics[:100]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if execution_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
