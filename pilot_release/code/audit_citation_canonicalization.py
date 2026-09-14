"""Create and audit citation-canonicalized scoring copies of v7 answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from citation_canonicalizer import canonicalize_citations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-prompt-protocol", required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--expected-answer-count", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
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
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def answer_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["seed"]),
    )


def main() -> int:
    args = parse_args()
    for path in (args.answers, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    input_sha256_before = sha256_file(args.answers)
    answers = read_jsonl(args.answers)
    manifest_rows = read_jsonl(args.manifest)
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate sample_id in manifest")
    if len(answers) != args.expected_answer_count:
        raise ValueError(
            f"Expected {args.expected_answer_count} answers, found {len(answers)}"
        )
    keys = [answer_key(row) for row in answers]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise ValueError(f"Duplicate answer keys: {duplicate_count}")

    expected_models = set(args.models)
    observed_models = {str(row.get("model")) for row in answers}
    if observed_models != expected_models:
        raise ValueError(
            f"Model mismatch: observed={sorted(observed_models)}, "
            f"expected={sorted(expected_models)}"
        )
    for row in answers:
        if str(row.get("answer_prompt_protocol")) != args.answer_prompt_protocol:
            raise ValueError(f"Protocol mismatch for {answer_key(row)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonicalized_path = args.output_dir / "canonicalized_answers.jsonl"
    actions_path = args.output_dir / "normalization_actions.jsonl"
    unresolved_path = args.output_dir / "unresolved_cases.jsonl"

    overall = Counter()
    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    idempotence_errors: list[dict[str, Any]] = []

    with (
        canonicalized_path.open("w", encoding="utf-8") as canonical_file,
        actions_path.open("w", encoding="utf-8") as actions_file,
        unresolved_path.open("w", encoding="utf-8") as unresolved_file,
    ):
        for row in answers:
            key = answer_key(row)
            item = manifest.get(key[1])
            if item is None:
                raise ValueError(f"Sample missing from manifest: {key[1]}")
            source_count = int(item["candidate_document_count"])
            raw_answer = row.get("answer")
            if not isinstance(raw_answer, str) or not raw_answer.strip():
                raise ValueError(f"Missing answer text: {key}")

            result = canonicalize_citations(raw_answer, source_count=source_count)
            second = canonicalize_citations(
                result["answer"], source_count=source_count
            )
            if second["answer"] != result["answer"]:
                idempotence_errors.append(
                    {"key": key, "first": result["answer"], "second": second["answer"]}
                )

            output_row = dict(row)
            output_row["raw_answer"] = raw_answer
            output_row["answer"] = result["answer"]
            output_row["answer_artifact_role"] = "citation_canonicalized_scoring_copy"
            output_row["source_answer_sha256"] = sha256_text(raw_answer)
            output_row["canonicalized_answer_sha256"] = sha256_text(result["answer"])
            output_row["citation_canonicalization"] = {
                "changed": result["changed"],
                "action_counts": result["action_counts"],
                "unresolved_count": len(result["unresolved"]),
                "semantic_support_checked": False,
                "raw_answer_preserved": True,
                "source_count": source_count,
            }
            canonical_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            action_row = {
                "model": key[0],
                "sample_id": key[1],
                "prompt_id": key[2],
                "seed": key[3],
                "benchmark": row["benchmark"],
                "domain": row["domain"],
                "source_count": source_count,
                "changed": result["changed"],
                "action_counts": result["action_counts"],
                "details": result["details"],
                "unresolved": result["unresolved"],
                "raw_answer": raw_answer if result["changed"] else None,
                "canonicalized_answer": result["answer"] if result["changed"] else None,
            }
            if result["changed"]:
                actions_file.write(json.dumps(action_row, ensure_ascii=False) + "\n")
            if result["unresolved"]:
                unresolved_file.write(json.dumps(action_row, ensure_ascii=False) + "\n")

            group_keys = (
                "overall",
                f"model::{key[0]}",
                f"benchmark::{row['benchmark']}",
                f"model_benchmark::{key[0]}::{row['benchmark']}",
            )
            flags = {
                "answer_count": True,
                "changed_answer_count": result["changed"],
                "unresolved_answer_count": bool(result["unresolved"]),
                "removed_trailing_reference_list_answer_count": bool(
                    result["action_counts"].get("remove_trailing_reference_list")
                ),
            }
            for action_name, count in result["action_counts"].items():
                flags[f"answer_with_{action_name}"] = True
                overall[f"action_occurrence::{action_name}"] += count
                if len(examples[action_name]) < 5:
                    examples[action_name].append(
                        {
                            "key": key,
                            "raw_end": raw_answer[-600:],
                            "canonicalized_end": result["answer"][-600:],
                        }
                    )
            for group_key in group_keys:
                for flag_name, enabled in flags.items():
                    if enabled:
                        groups[group_key][flag_name] += 1

    if idempotence_errors:
        raise AssertionError(
            f"Canonicalizer is not idempotent for {len(idempotence_errors)} answers"
        )

    input_sha256_after = sha256_file(args.answers)
    if input_sha256_before != input_sha256_after:
        raise AssertionError("Raw answer file changed during canonicalization")

    output_keys = [
        answer_key(json.loads(line))
        for line in canonicalized_path.open(encoding="utf-8")
        if line.strip()
    ]
    if output_keys != keys:
        raise AssertionError("Output key order differs from input")

    overall_group = groups["overall"]
    metrics = {
        "experiment_id": args.experiment_id,
        "phase": "diagnostic_cpu",
        "method": args.method_version,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "input_answer_count": len(answers),
        "output_answer_count": len(output_keys),
        "duplicate_count": duplicate_count,
        "idempotence_error_count": len(idempotence_errors),
        "input_sha256_before": input_sha256_before,
        "input_sha256_after": input_sha256_after,
        "raw_answers_modified": False,
        "canonicalized_scoring_copy_created": True,
        "changed_answer_count": overall_group["changed_answer_count"],
        "unchanged_answer_count": len(answers) - overall_group["changed_answer_count"],
        "unresolved_answer_count": overall_group["unresolved_answer_count"],
        "removed_trailing_reference_list_answer_count": overall_group[
            "removed_trailing_reference_list_answer_count"
        ],
        "action_occurrence_counts": {
            key.split("::", 1)[1]: value
            for key, value in sorted(overall.items())
            if key.startswith("action_occurrence::")
        },
        "group_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(groups.items())
        },
        "examples_by_action": dict(examples),
        "complete_accounting": len(output_keys) == len(answers),
        "integrity_passed": (
            len(output_keys) == len(answers)
            and not duplicate_count
            and not idempotence_errors
            and input_sha256_before == input_sha256_after
        ),
        "unresolved_policy": "record_and_continue",
        "semantic_support_checked": False,
        "gpu_required": False,
        "official_geo_metrics_computed": False,
        "raid_judge_called": False,
        "result_paths": {
            "canonicalized_answers": str(canonicalized_path),
            "normalization_actions": str(actions_path),
            "unresolved_cases": str(unresolved_path),
        },
        "next_gate": "user_audit_before_rescoring",
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "audit.json",
        {
            "experiment_id": args.experiment_id,
            "integrity_passed": metrics["integrity_passed"],
            "complete_accounting": metrics["complete_accounting"],
            "input_sha256_matched": input_sha256_before == input_sha256_after,
            "idempotent": not idempotence_errors,
            "unresolved_cases_are_diagnostic_only": True,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
