"""Run one checkpointable Answer Model shard for the necessity pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_prompt import (
    ANSWER_PROMPT_PROTOCOLS,
    BENCHMARK_OFFICIAL_V1,
    build_answer_messages,
)
from citation_validator import (
    validate_citations,
    validate_v2_citation_contract,
    validate_v4_optional_citation_contract,
    validate_v6_strict_citation_contract,
)
from create_necessity_manifest import load_cseo, load_geo
from model_loader import (
    MODEL_SPECS,
    generate_answer,
    load_model,
    release_model,
    render_chat,
)
from run_necessity_rewrite_stage import PROMPT_ORDER


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROMPT_ID = "original.no_rewrite"


def build_citation_diagnostics(answer: str, *, source_count: int) -> dict[str, Any]:
    """Return historical, strict, and optional citation views without editing text."""

    return {
        "citation_diagnostics": validate_citations(
            answer,
            source_count=source_count,
        ),
        # Historical field names are retained so later prompt protocols can be
        # compared with the same frozen strict and optional syntax validators.
        "v2_citation_contract": validate_v2_citation_contract(
            answer,
            source_count=source_count,
        ),
        "v4_citation_contract": validate_v4_optional_citation_contract(
            answer,
            source_count=source_count,
        ),
        "v6_citation_contract": validate_v6_strict_citation_contract(
            answer,
            source_count=source_count,
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key_from_row(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["seed"]),
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_conditions(
    manifest_path: Path,
    rewrites_path: Path,
    geo_path: Path,
    cseo_root: Path,
) -> tuple[list[tuple[dict[str, Any], str, list[str]]], dict[str, int]]:
    manifest = read_jsonl(manifest_path)
    source_rows = load_geo(geo_path) + load_cseo(cseo_root)
    source_map = {
        f"{row['benchmark']}__{row['domain']}__{row['instance_id']}": row
        for row in source_rows
    }
    rewrite_rows = read_jsonl(rewrites_path)
    rewrite_map: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_rewrites = 0
    for row in rewrite_rows:
        key = (str(row["sample_id"]), str(row["prompt_id"]))
        if key in rewrite_map:
            duplicate_rewrites += 1
        rewrite_map[key] = row
    if duplicate_rewrites:
        raise ValueError(f"Duplicate rewrite keys: {duplicate_rewrites}")

    expected_prompt_ids = set(PROMPT_ORDER)
    conditions: list[tuple[dict[str, Any], str, list[str]]] = []
    for item in manifest:
        sample_id = str(item["sample_id"])
        if sample_id not in source_map:
            raise KeyError(f"Manifest sample missing from source data: {sample_id}")
        available = {
            prompt_id for candidate_id, prompt_id in rewrite_map if candidate_id == sample_id
        }
        missing = expected_prompt_ids - available
        unexpected = available - expected_prompt_ids
        if missing or unexpected:
            raise ValueError(
                f"Rewrite set mismatch for {sample_id}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        source = source_map[sample_id]
        target_index = int(item["target_document_index"])
        original_documents = list(source["documents"])
        if not 0 <= target_index < len(original_documents):
            raise IndexError(
                f"Target index {target_index} outside {len(original_documents)} documents "
                f"for {sample_id}"
            )
        conditions.append((item, ORIGINAL_PROMPT_ID, original_documents))
        for prompt_id in PROMPT_ORDER:
            documents = list(original_documents)
            documents[target_index] = rewrite_map[(sample_id, prompt_id)][
                "rewritten_text"
            ]
            conditions.append((item, prompt_id, documents))

    diagnostics = {
        "manifest_count": len(manifest),
        "rewrite_count": len(rewrite_rows),
        "conditions_per_instance": 1 + len(PROMPT_ORDER),
        "condition_count": len(conditions),
    }
    return conditions, diagnostics


def build_literature_conditions(manifest_path, rewrites_path, geo_path, cseo_root, *, sample_ids=None, missing_rewrites_path=None):
    """Replace only the target source; keep literature method IDs and text intact."""
    from literature_method_runner import LiteratureMethodCatalog
    from run_literature_method_rewrite_stage import select_manifest_rows

    manifest = select_manifest_rows(read_jsonl(manifest_path), sample_ids)
    ids = [str(item["sample_id"]) for item in manifest]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest must contain unique nonempty samples")
    source_map = {
        f"{r['benchmark']}__{r['domain']}__{r['instance_id']}": r
        for r in load_geo(geo_path) + load_cseo(cseo_root)
    }
    methods = LiteratureMethodCatalog().method_ids
    selected = set(ids)
    rewrite_map = {}
    for row in read_jsonl(rewrites_path):
        if row["sample_id"] not in selected:
            raise ValueError("Rewrite contains samples outside selected manifest")
        key = (row["sample_id"], row["method_id"])
        if key in rewrite_map:
            raise ValueError("Duplicate literature rewrite")
        rewrite_map[key] = row
    expected = {(sid, method) for sid in ids for method in methods}
    missing_rows = read_jsonl(missing_rewrites_path) if missing_rewrites_path else []
    declared_missing = {(r['sample_id'], r['method_id']) for r in missing_rows}
    if len(declared_missing) != len(missing_rows) or declared_missing & set(rewrite_map):
        raise ValueError("Duplicate or conflicting declared missing rewrites")
    if set(rewrite_map) | declared_missing != expected:
        raise ValueError("Literature rewrite set mismatch")
    conditions = []
    for item in manifest:
        sid = item["sample_id"]
        source = source_map[sid]
        original = list(source["documents"])
        target = int(item["target_document_index"])
        if not 0 <= target < len(original):
            raise ValueError("Target index outside source list")
        conditions.append((item, ORIGINAL_PROMPT_ID, original))
        for method in methods:
            if (sid, method) in declared_missing:
                continue
            row = rewrite_map[(sid, method)]
            if int(row["target_document_index"]) != target or row["source_text"] != original[target]:
                raise ValueError("Literature rewrite source or target mismatch")
            if not isinstance(row["rewritten_text"], str) or not row["rewritten_text"].strip():
                raise ValueError("Empty literature rewrite")
            documents = list(original)
            documents[target] = row["rewritten_text"]
            conditions.append((item, method, documents))
    return conditions, {"manifest_count": len(manifest), "rewrite_count": len(rewrite_map),
                        "conditions_per_instance": None if declared_missing else 1 + len(methods),
                        "declared_missing_rewrite_count": len(declared_missing),
                        "condition_count": len(conditions), "rewrite_profile": "literature_methods"}


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl",
    )
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--rewrite-profile", choices=("custom_prompts", "literature_methods"), default="custom_prompts")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--missing-rewrites", type=Path)
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
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--input-budget", type=int, default=20480)
    parser.add_argument("--allowed-error-types", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")
    if args.rewrite_profile == "literature_methods":
        conditions, diagnostics = build_literature_conditions(
            args.manifest, args.rewrites, args.geo_path, args.cseo_root,
            sample_ids=args.sample_ids,
            missing_rewrites_path=args.missing_rewrites,
        )
    else:
        if args.sample_ids:
            raise ValueError("--sample-ids is currently supported only for literature_methods")
        conditions, diagnostics = build_conditions(
            args.manifest, args.rewrites, args.geo_path, args.cseo_root
        )
    from source_content_truncation import apply_source_truncations, input_hash
    conditions, truncation_info = apply_source_truncations(conditions, args.source_truncations, args.answer_prompt_protocol)
    diagnostics.update(truncation_info)
    truncated_keys = {(r['sample_id'], r['prompt_id']) for r in read_jsonl(args.source_truncations)} if args.source_truncations else set()
    expected = len(conditions) * len(args.seeds)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "experiment_id": args.experiment_id,
                    "model": args.model,
                    **diagnostics,
                    "seeds": args.seeds,
                    "expected_generations": expected,
                    "input_budget": args.input_budget,
                    "max_new_tokens": args.max_new_tokens,
                    "answer_prompt_protocol": args.answer_prompt_protocol,
                },
                ensure_ascii=False,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_dir / "answers.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    progress_path = args.output_dir / "progress.json"
    metrics_path = args.output_dir / "metrics.json"
    existing_paths = [
        path
        for path in (answers_path, errors_path, progress_path, metrics_path)
        if path.exists() and path.stat().st_size
    ]
    if existing_paths and not args.resume:
        formatted = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Existing shard outputs require --resume: {formatted}")

    existing_answers = read_jsonl(answers_path) if args.resume else []
    existing_errors = read_jsonl(errors_path) if args.resume else []
    for row in existing_answers + existing_errors:
        if str(row.get("model")) != args.model:
            raise ValueError(
                f"Output shard for {args.model} contains row for {row.get('model')}"
            )
        row_protocol = str(
            row.get("answer_prompt_protocol", BENCHMARK_OFFICIAL_V1)
        )
        if row_protocol != args.answer_prompt_protocol:
            raise ValueError(
                "Refusing to resume output created with answer prompt protocol "
                f"{row_protocol!r} under {args.answer_prompt_protocol!r}"
            )
    answer_keys = {key_from_row(row) for row in existing_answers}
    error_keys = {key_from_row(row) for row in existing_errors}
    overlap = answer_keys & error_keys
    if overlap:
        raise ValueError(f"Keys recorded as both success and error: {len(overlap)}")
    if len(answer_keys) != len(existing_answers):
        raise ValueError("Duplicate answer keys in checkpoint")
    if len(error_keys) != len(existing_errors):
        raise ValueError("Duplicate error keys in checkpoint")
    accounted = answer_keys | error_keys

    def progress_payload() -> dict[str, Any]:
        completed = len(answer_keys)
        failures = len(error_keys)
        finished = completed + failures
        return {
            "experiment_id": args.experiment_id,
            "model": args.model,
            "answer_prompt_protocol": args.answer_prompt_protocol,
            "expected_generations": expected,
            "completed_generations": completed,
            "error_count": failures,
            "accounted_generations": finished,
            "remaining_generations": expected - finished,
            "percent_accounted": round(100.0 * finished / expected, 4),
            "checkpointed": True,
        }

    atomic_json(progress_path, progress_payload())
    loaded = load_model(
        args.model,
        model_root=args.model_root,
        dtype_name="bfloat16",
        attn_implementation="sdpa",
        device_map_strategy="balanced",
    )
    new_successes = 0
    new_errors = 0
    try:
        with answers_path.open("a", encoding="utf-8") as answers_file, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_file:
            for item, prompt_id, documents in conditions:
                messages = build_answer_messages(
                    protocol=args.answer_prompt_protocol,
                    benchmark=str(item["benchmark"]),
                    domain=str(item["domain"]),
                    query=str(item["query"]),
                    sources=documents,
                )
                rendered = render_chat(loaded, messages)
                input_tokens = int(rendered["input_ids"].shape[-1])
                for seed in args.seeds:
                    key = (args.model, str(item["sample_id"]), prompt_id, seed)
                    if key in accounted:
                        continue
                    if input_tokens > args.input_budget:
                        row = {
                            "experiment_id": args.experiment_id,
                            "model": args.model,
                            "sample_id": item["sample_id"],
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "answer_prompt_protocol": args.answer_prompt_protocol,
                            "error_type": "input_budget_overflow",
                            "input_tokens": input_tokens,
                            "input_budget": args.input_budget,
                        }
                        errors_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        errors_file.flush()
                        error_keys.add(key)
                        accounted.add(key)
                        new_errors += 1
                    else:
                        try:
                            generation = generate_answer(
                                loaded,
                                messages,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                seed=seed,
                            )
                            row = {
                                "experiment_id": args.experiment_id,
                                "model": args.model,
                                "sample_id": item["sample_id"],
                                "benchmark": item["benchmark"],
                                "domain": item["domain"],
                                "query": item["query"],
                                "prompt_id": prompt_id,
                                "seed": seed,
                                "answer_prompt_protocol": args.answer_prompt_protocol,
                                "input_tokens": generation.input_tokens,
                                "output_tokens": generation.output_tokens,
                                "max_new_tokens": args.max_new_tokens,
                                "cap_hit": generation.output_tokens
                                >= args.max_new_tokens,
                                "answer": generation.answer,
                                "effective_source_input_sha256": input_hash(item, documents),
                                "source_truncated": (item['sample_id'], prompt_id) in truncated_keys,
                                "source_truncation_file": str(args.source_truncations) if args.source_truncations else None,
                                **build_citation_diagnostics(
                                    generation.answer,
                                    source_count=len(documents),
                                ),
                            }
                            answers_file.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                            answers_file.flush()
                            answer_keys.add(key)
                            accounted.add(key)
                            new_successes += 1
                        except Exception as exc:
                            row = {
                                "experiment_id": args.experiment_id,
                                "model": args.model,
                                "sample_id": item["sample_id"],
                                "prompt_id": prompt_id,
                                "seed": seed,
                                "answer_prompt_protocol": args.answer_prompt_protocol,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            errors_file.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                            errors_file.flush()
                            error_keys.add(key)
                            accounted.add(key)
                            new_errors += 1
                    atomic_json(progress_path, progress_payload())
                    if len(accounted) % 25 == 0 or len(accounted) == expected:
                        print(json.dumps(progress_payload(), ensure_ascii=False), flush=True)
    finally:
        release_model(loaded)

    error_type_counts = Counter(
        str(row.get("error_type", "unknown")) for row in read_jsonl(errors_path)
    )
    unexpected_error_types = sorted(
        set(error_type_counts) - set(args.allowed_error_types)
    )
    complete_accounting = len(accounted) == expected
    metrics = {
        **progress_payload(),
        **diagnostics,
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "input_budget": args.input_budget,
        "new_successes_this_run": new_successes,
        "new_errors_this_run": new_errors,
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "allowed_error_types": sorted(set(args.allowed_error_types)),
        "unexpected_error_types": unexpected_error_types,
        "protocol_exclusion_count": sum(
            count
            for error_type, count in error_type_counts.items()
            if error_type in set(args.allowed_error_types)
        ),
        "complete_accounting": complete_accounting,
        "all_successful": len(answer_keys) == expected and not error_keys,
        "execution_success": complete_accounting and not unexpected_error_types,
    }
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["execution_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
