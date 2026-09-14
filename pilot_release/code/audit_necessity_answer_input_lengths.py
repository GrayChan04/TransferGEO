"""Audit all frozen necessity-pilot conditions with local model tokenizers only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoConfig, AutoTokenizer

from answer_prompt import (
    ANSWER_PROMPT_PROTOCOLS,
    UNIFIED_ANSWER_V1,
    build_answer_messages,
    get_answer_prompt_files,
)
from audit_benchmark_lengths import count_chat_tokens, model_context_limit
from model_loader import DEFAULT_MODEL_ROOT, MODEL_SPECS
from run_necessity_answer_model_shard import build_conditions, build_literature_conditions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl"
)
DEFAULT_REWRITES = (
    ROOT / "results/processed/necessity_rewrite_stage_v1/rewrites.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/processed/necessity_answer_v2_length_audit_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-id",
        default="necessity_answer_v2_length_audit_v1",
    )
    parser.add_argument(
        "--answer-prompt-protocol",
        choices=ANSWER_PROMPT_PROTOCOLS,
        default=UNIFIED_ANSWER_V1,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rewrites", type=Path, default=DEFAULT_REWRITES)
    parser.add_argument("--rewrite-profile", choices=("custom_prompts", "literature_methods"), default="custom_prompts")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--missing-rewrites", type=Path,
                        help="Explicit failed rewrite ledger; never fill missing text with Original.")
    parser.add_argument("--source-truncations", type=Path)
    parser.add_argument(
        "--geo-path",
        type=Path,
        default=ROOT / "data/raw/benchmarks/geo_bench/test.jsonl",
    )
    parser.add_argument(
        "--cseo-root",
        type=Path,
        default=ROOT / "data/raw/benchmarks/cseo_bench/data",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--input-budget", type=int, default=20_480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--result-name-prefix",
        default="",
        help=(
            "Optional basename prefix for semantic result filenames; "
            "the empty default preserves historical filenames."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def percentile(sorted_values: Sequence[int], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate percentile for an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def describe_rows(
    rows: Sequence[dict[str, Any]],
    *,
    input_budget: int,
) -> dict[str, int | float]:
    values = sorted(int(row["input_tokens"]) for row in rows)
    overflow_count = sum(value > input_budget for value in values)
    return {
        "count": len(values),
        "min": values[0],
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": values[-1],
        "overflow_count": overflow_count,
        "overflow_rate": round(overflow_count / len(values), 8),
    }


def grouped_descriptions(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    input_budget: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    return {
        "::".join(key): describe_rows(values, input_budget=input_budget)
        for key, values in sorted(grouped.items())
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metrics(
    *,
    experiment_id: str,
    protocol: str,
    seed: int,
    input_budget: int,
    rows: Sequence[dict[str, Any]],
    expected_condition_keys: set[tuple[str, str]],
    models: Sequence[str],
    diagnostics: dict[str, int],
    model_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt_files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in get_answer_prompt_files(protocol).items()
    }
    keys = [
        (str(row["model"]), str(row["sample_id"]), str(row["prompt_id"]))
        for row in rows
    ]
    unique_keys = set(keys)
    expected_keys = {
        (model, sample_id, prompt_id)
        for model in models
        for sample_id, prompt_id in expected_condition_keys
    }
    condition_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        condition_lengths[(str(row["sample_id"]), str(row["prompt_id"]))].append(
            int(row["input_tokens"])
        )
    overflow_rows = [row for row in rows if bool(row["over_budget"])]
    overflow_condition_keys = {
        (str(row["sample_id"]), str(row["prompt_id"])) for row in overflow_rows
    }
    overflow_sample_ids = {str(row["sample_id"]) for row in overflow_rows}
    common_eligible = sum(
        len(values) == len(models) and max(values) <= input_budget
        for values in condition_lengths.values()
    )
    duplicate_count = len(keys) - len(unique_keys)
    missing = expected_keys - unique_keys
    unexpected = unique_keys - expected_keys
    complete_accounting = (
        len(rows) == len(expected_keys)
        and duplicate_count == 0
        and not missing
        and not unexpected
        and len(condition_lengths) == len(expected_condition_keys)
        and all(len(values) == len(models) for values in condition_lengths.values())
    )
    return {
        "experiment_id": experiment_id,
        "phase": "diagnostic",
        "answer_prompt_protocol": protocol,
        "seed": seed,
        "deterministic_tokenization": True,
        "model_weights_loaded": False,
        "gpu_required": False,
        "input_budget": input_budget,
        "manifest_count": diagnostics["manifest_count"],
        "rewrite_count": diagnostics["rewrite_count"],
        "conditions_per_instance": diagnostics["conditions_per_instance"],
        "condition_count": len(expected_condition_keys),
        "models": list(models),
        "expected_record_count": len(expected_keys),
        "record_count": len(rows),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete_accounting": complete_accounting,
        "all_conditions_within_common_budget": not overflow_rows,
        "common_budget_eligible_condition_count": common_eligible,
        "common_budget_overflow_condition_count": len(overflow_condition_keys),
        "overflow_record_count": len(overflow_rows),
        "overflow_sample_count": len(overflow_sample_ids),
        "by_model": grouped_descriptions(
            rows,
            group_fields=("model",),
            input_budget=input_budget,
        ),
        "by_benchmark_model": grouped_descriptions(
            rows,
            group_fields=("benchmark", "model"),
            input_budget=input_budget,
        ),
        "by_prompt_model": grouped_descriptions(
            rows,
            group_fields=("prompt_id", "model"),
            input_budget=input_budget,
        ),
        "model_metadata": model_metadata,
        "prompt_files": prompt_files,
        "integrity_preview": {
            "missing_first_20": sorted(missing)[:20],
            "unexpected_first_20": sorted(unexpected)[:20],
        },
        "next_gate": (
            "minimal_gpu_smoke"
            if complete_accounting and not overflow_rows
            else "review_overflow_conditions_before_gpu_smoke"
        ),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_result_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    """Build contained result paths while preserving historical defaults."""

    if prefix and (Path(prefix).name != prefix or prefix in {".", ".."}):
        raise ValueError(f"result name prefix must be a basename fragment: {prefix!r}")
    if prefix:
        return {
            "lengths": output_dir / f"{prefix}input_token_lengths.jsonl",
            "overflows": output_dir / f"{prefix}input_budget_overflows.jsonl",
            "metrics": output_dir / f"{prefix}input_length_audit_metrics.json",
        }
    return {
        "lengths": output_dir / "lengths.jsonl",
        "overflows": output_dir / "overflows.jsonl",
        "metrics": output_dir / "metrics.json",
    }


def main() -> int:
    args = parse_args()
    if args.input_budget <= 0:
        raise ValueError("input budget must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError(f"models must be unique: {args.models}")
    result_paths = build_result_paths(args.output_dir, args.result_name_prefix)

    if args.rewrite_profile == "literature_methods":
        conditions, diagnostics = build_literature_conditions(
            args.manifest, args.rewrites, args.geo_path, args.cseo_root,
            sample_ids=args.sample_ids,
            missing_rewrites_path=args.missing_rewrites,
        )
    else:
        if args.sample_ids:
            raise ValueError("--sample-ids requires literature_methods")
        conditions, diagnostics = build_conditions(
            args.manifest, args.rewrites, args.geo_path, args.cseo_root,
        )
    from source_content_truncation import apply_source_truncations
    conditions, truncation_info = apply_source_truncations(conditions, args.source_truncations, args.answer_prompt_protocol)
    diagnostics.update(truncation_info)
    expected_condition_keys = {
        (str(item["sample_id"]), prompt_id)
        for item, prompt_id, _documents in conditions
    }
    if len(expected_condition_keys) != len(conditions):
        raise ValueError("condition keys are not unique")
    expected_records = len(conditions) * len(args.models)
    plan = {
        "experiment_id": args.experiment_id,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        **diagnostics,
        "models": args.models,
        "input_budget": args.input_budget,
        "seed": args.seed,
        "expected_record_count": expected_records,
        "result_paths": {name: str(path) for name, path in result_paths.items()},
        "model_weights_loaded": False,
        "gpu_required": False,
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    rows: list[dict[str, Any]] = []
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        model_path = spec.path(args.model_root)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        context_limit = model_context_limit(config, tokenizer)
        model_metadata[model_name] = {
            "model_id": spec.model_id,
            "model_path": str(model_path),
            "model_context_limit": context_limit,
            "tokenizer_model_max_length": int(tokenizer.model_max_length),
            "thinking_disabled": spec.disable_thinking,
        }
        print(
            f"Auditing {model_name}: {len(conditions)} conditions with tokenizer only.",
            flush=True,
        )
        for index, (item, prompt_id, documents) in enumerate(conditions, start=1):
            messages = build_answer_messages(
                protocol=args.answer_prompt_protocol,
                benchmark=str(item["benchmark"]),
                domain=str(item["domain"]),
                query=str(item["query"]),
                sources=documents,
            )
            input_tokens = count_chat_tokens(
                tokenizer,
                messages,
                disable_thinking=spec.disable_thinking,
            )
            rows.append(
                {
                    "experiment_id": args.experiment_id,
                    "answer_prompt_protocol": args.answer_prompt_protocol,
                    "model": model_name,
                    "model_id": spec.model_id,
                    "sample_id": item["sample_id"],
                    "benchmark": item["benchmark"],
                    "domain": item["domain"],
                    "prompt_id": prompt_id,
                    "document_count": len(documents),
                    "document_chars": sum(len(document) for document in documents),
                    "input_tokens": input_tokens,
                    "input_budget": args.input_budget,
                    "over_budget": input_tokens > args.input_budget,
                    "model_context_limit": context_limit,
                }
            )
            if index % 500 == 0 or index == len(conditions):
                print(
                    f"  {model_name}: {index}/{len(conditions)} complete.",
                    flush=True,
                )
        del tokenizer, config
        gc.collect()

    metrics = build_metrics(
        experiment_id=args.experiment_id,
        protocol=args.answer_prompt_protocol,
        seed=args.seed,
        input_budget=args.input_budget,
        rows=rows,
        expected_condition_keys=expected_condition_keys,
        models=args.models,
        diagnostics=diagnostics,
        model_metadata=model_metadata,
    )
    if not metrics["complete_accounting"]:
        raise RuntimeError(
            "Tokenizer audit integrity failure: "
            f"{metrics['integrity_preview']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths_path = result_paths["lengths"]
    overflows_path = result_paths["overflows"]
    metrics_path = result_paths["metrics"]
    write_jsonl(lengths_path, rows)
    write_jsonl(
        overflows_path,
        (row for row in rows if bool(row["over_budget"])),
    )
    metrics["result_paths"] = {
        "lengths": str(lengths_path),
        "overflows": str(overflows_path),
        "metrics": str(metrics_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
