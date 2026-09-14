"""Audit RAID subjective-score reuse after citation canonicalization.

This CPU-only utility aligns the immutable v7 answers, their citation-
canonicalized scoring copies, and the previously completed RAID Judge output.
It creates three disjoint answer manifests:

* unchanged text with the target still cited -> reuse the old Judge scores;
* changed text with the target cited -> rerun the Judge on canonicalized text;
* target not cited after canonicalization -> assign deterministic zeros later.

The utility never edits existing answers or scores and never calls an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


DIMENSIONS = (
    "diversity_detailed",
    "follow_detailed",
    "influence_detailed",
    "relevance_detailed",
    "subjcount_detailed",
    "subjpos_detailed",
    "uniqueness_detailed",
)
CITATION_INDEX = re.compile(r"\[(\d+)\]")
ACTION_REUSE = "reuse_old_scores"
ACTION_RERUN = "rerun_glm_judge"
ACTION_ZERO = "deterministic_zero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--raw-answers", type=Path, required=True)
    parser.add_argument("--canonicalized-answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-judge-trials", type=Path, required=True)
    parser.add_argument("--old-judge-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-prompt-protocol", default="unified_answer_v7")
    parser.add_argument("--expected-answer-count", type=int, required=True)
    parser.add_argument("--expected-reuse-answer-count", type=int, required=True)
    parser.add_argument("--expected-rerun-answer-count", type=int, required=True)
    parser.add_argument("--expected-zero-answer-count", type=int, required=True)
    parser.add_argument("--expected-rerun-trial-count", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
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
            yield row


def answer_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    seed_name = "seed" if "seed" in row else "answer_seed"
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row[seed_name]),
    )


def trial_key(row: dict[str, Any]) -> tuple[str, str, str, int, str, int]:
    return (*answer_key(row), str(row["dimension"]), int(row["judge_seed"]))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_is_cited(answer_text: str, target_citation_index: int) -> bool:
    return target_citation_index in {
        int(value) for value in CITATION_INDEX.findall(answer_text)
    }


def classify_alignment(*, text_changed: bool, target_cited_after: bool) -> str:
    if not target_cited_after:
        return ACTION_ZERO
    if text_changed:
        return ACTION_RERUN
    return ACTION_REUSE


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in targets:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        candidate_count = int(row["candidate_document_count"])
        target_document_index = int(row["target_document_index"])
        if not 0 <= target_document_index < candidate_count:
            raise ValueError(
                f"Invalid target index for {sample_id}: "
                f"{target_document_index}/{candidate_count}"
            )
        targets[sample_id] = {
            "target_document_index": target_document_index,
            "target_citation_index": target_document_index + 1,
            "candidate_document_count": candidate_count,
        }
    return targets


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def lightweight_manifest_row(
    row: dict[str, Any],
    *,
    action: str,
    target: dict[str, Any],
    raw_target_cited: bool,
    canonical_target_cited: bool,
    text_changed: bool,
) -> dict[str, Any]:
    key = answer_key(row)
    return {
        "model": key[0],
        "sample_id": key[1],
        "prompt_id": key[2],
        "answer_seed": key[3],
        "benchmark": str(row["benchmark"]),
        "domain": str(row["domain"]),
        **target,
        "raw_target_source_cited": raw_target_cited,
        "canonical_target_source_cited": canonical_target_cited,
        "canonicalized_text_changed": text_changed,
        "subjective_score_action": action,
        "source_answer_sha256": str(row["source_answer_sha256"]),
        "canonicalized_answer_sha256": str(row["canonicalized_answer_sha256"]),
    }


def validate_expected(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"Expected {expected} {name}, found {actual}")


def main() -> int:
    args = parse_args()
    input_paths = (
        args.raw_answers,
        args.canonicalized_answers,
        args.manifest,
        args.old_judge_trials,
        args.old_judge_scores,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Judge seeds must be non-empty and unique: {args.seeds}")

    targets = load_targets(args.manifest)
    raw_index: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    raw_protocol_mismatch_count = 0
    for row in iter_jsonl(args.raw_answers):
        key = answer_key(row)
        if key in raw_index:
            raise ValueError(f"Duplicate raw answer key: {key}")
        if str(row.get("answer_prompt_protocol")) != args.answer_prompt_protocol:
            raw_protocol_mismatch_count += 1
        target = targets.get(key[1])
        if target is None:
            raise ValueError(f"Raw answer sample is absent from manifest: {key[1]}")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"Missing raw answer text: {key}")
        raw_index[key] = {
            "answer_sha256": sha256_text(answer),
            "target_source_cited": target_is_cited(
                answer, int(target["target_citation_index"])
            ),
            "benchmark": str(row["benchmark"]),
            "domain": str(row["domain"]),
        }
    validate_expected("raw answers", len(raw_index), args.expected_answer_count)
    if raw_protocol_mismatch_count:
        raise ValueError(
            f"Found {raw_protocol_mismatch_count} raw answer protocol mismatches"
        )

    action_by_key: dict[tuple[str, str, str, int], str] = {}
    raw_target_by_key: dict[tuple[str, str, str, int], bool] = {}
    answer_alignment_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    rerun_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    changed_counts: Counter[str] = Counter()
    canonical_keys: set[tuple[str, str, str, int]] = set()

    for row in iter_jsonl(args.canonicalized_answers):
        key = answer_key(row)
        if key in canonical_keys:
            raise ValueError(f"Duplicate canonicalized answer key: {key}")
        canonical_keys.add(key)
        raw = raw_index.get(key)
        if raw is None:
            raise ValueError(f"Canonicalized answer has no raw answer: {key}")
        target = targets[key[1]]
        answer = row.get("answer")
        raw_answer = row.get("raw_answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"Missing canonicalized answer text: {key}")
        if not isinstance(raw_answer, str) or not raw_answer:
            raise ValueError(f"Missing preserved raw_answer text: {key}")
        source_hash = str(row.get("source_answer_sha256", ""))
        canonical_hash = str(row.get("canonicalized_answer_sha256", ""))
        if source_hash != raw["answer_sha256"]:
            raise ValueError(f"Raw answer hash mismatch: {key}")
        if sha256_text(raw_answer) != raw["answer_sha256"]:
            raise ValueError(f"Preserved raw_answer mismatch: {key}")
        if canonical_hash != sha256_text(answer):
            raise ValueError(f"Canonicalized answer hash mismatch: {key}")
        canonicalization = row.get("citation_canonicalization")
        if not isinstance(canonicalization, dict):
            raise ValueError(f"Missing citation_canonicalization metadata: {key}")
        text_changed = source_hash != canonical_hash
        if bool(canonicalization.get("changed")) != text_changed:
            raise ValueError(f"Canonicalization changed flag mismatch: {key}")
        if str(row.get("benchmark")) != raw["benchmark"]:
            raise ValueError(f"Benchmark mismatch between answer copies: {key}")
        if str(row.get("domain")) != raw["domain"]:
            raise ValueError(f"Domain mismatch between answer copies: {key}")

        raw_target_cited = bool(raw["target_source_cited"])
        canonical_target_cited = target_is_cited(
            answer, int(target["target_citation_index"])
        )
        action = classify_alignment(
            text_changed=text_changed,
            target_cited_after=canonical_target_cited,
        )
        action_by_key[key] = action
        raw_target_by_key[key] = raw_target_cited
        action_counts[action] += 1
        transition = f"{str(raw_target_cited).lower()}_to_{str(canonical_target_cited).lower()}"
        transition_counts[transition] += 1
        changed_counts[str(text_changed).lower()] += 1
        manifest_row = lightweight_manifest_row(
            row,
            action=action,
            target=target,
            raw_target_cited=raw_target_cited,
            canonical_target_cited=canonical_target_cited,
            text_changed=text_changed,
        )
        answer_alignment_rows.append(manifest_row)
        if action == ACTION_REUSE:
            reuse_rows.append(manifest_row)
        elif action == ACTION_RERUN:
            rerun_rows.append(
                {
                    **row,
                    "subjective_score_action": action,
                    "raw_target_source_cited": raw_target_cited,
                    "canonical_target_source_cited": canonical_target_cited,
                }
            )
        else:
            zero_rows.append(manifest_row)

    if canonical_keys != set(raw_index):
        missing = sorted(set(raw_index) - canonical_keys)
        extra = sorted(canonical_keys - set(raw_index))
        raise ValueError(
            f"Raw/canonical key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    validate_expected("canonicalized answers", len(canonical_keys), args.expected_answer_count)
    validate_expected(
        "reuse answers", action_counts[ACTION_REUSE], args.expected_reuse_answer_count
    )
    validate_expected(
        "rerun answers", action_counts[ACTION_RERUN], args.expected_rerun_answer_count
    )
    validate_expected(
        "zero answers", action_counts[ACTION_ZERO], args.expected_zero_answer_count
    )

    old_score_keys: set[tuple[str, str, str, int]] = set()
    old_score_target_mismatch_count = 0
    old_score_unscorable_count = 0
    old_absent_nonzero_average_count = 0
    for row in iter_jsonl(args.old_judge_scores):
        key = answer_key(row)
        if key in old_score_keys:
            raise ValueError(f"Duplicate old Judge score key: {key}")
        old_score_keys.add(key)
        if key not in action_by_key:
            raise ValueError(f"Unexpected old Judge score key: {key}")
        old_target_cited = bool(row.get("target_source_cited"))
        if old_target_cited != raw_target_by_key[key]:
            old_score_target_mismatch_count += 1
        if not bool(row.get("all_dimensions_scorable")):
            old_score_unscorable_count += 1
        average = row.get("subjective_average")
        if average is None:
            old_score_unscorable_count += 1
        elif not raw_target_by_key[key] and float(average) != 0.0:
            old_absent_nonzero_average_count += 1
    if old_score_keys != set(action_by_key):
        missing = sorted(set(action_by_key) - old_score_keys)
        raise ValueError(f"Old Judge scores are missing answer keys: {missing[:5]}")
    if old_score_target_mismatch_count:
        raise ValueError(
            f"Found {old_score_target_mismatch_count} old score target-citation mismatches"
        )
    if old_score_unscorable_count:
        raise ValueError(f"Found {old_score_unscorable_count} unscorable old scores")
    if old_absent_nonzero_average_count:
        raise ValueError(
            f"Found {old_absent_nonzero_average_count} nonzero absent-target averages"
        )

    expected_combinations = {
        (dimension, seed) for dimension in DIMENSIONS for seed in args.seeds
    }
    expected_trial_count = len(action_by_key) * len(expected_combinations)
    previous_trial_key: tuple[str, str, str, int, str, int] | None = None
    current_answer_key: tuple[str, str, str, int] | None = None
    current_combinations: set[tuple[str, int]] = set()
    completed_trial_answer_keys: set[tuple[str, str, str, int]] = set()
    old_trial_count = 0
    old_trial_action_counts: Counter[str] = Counter()
    old_trial_origin_counts: Counter[str] = Counter()

    def finish_trial_group() -> None:
        if current_answer_key is None:
            return
        if current_combinations != expected_combinations:
            missing = sorted(expected_combinations - current_combinations)
            extra = sorted(current_combinations - expected_combinations)
            raise ValueError(
                f"Old trials incomplete for {current_answer_key}: "
                f"missing={missing}, extra={extra}"
            )
        completed_trial_answer_keys.add(current_answer_key)

    for row in iter_jsonl(args.old_judge_trials):
        key = trial_key(row)
        base_key = key[:4]
        if previous_trial_key is not None and key <= previous_trial_key:
            raise ValueError(
                f"Old Judge trials are not strictly sorted or contain duplicates: {key}"
            )
        previous_trial_key = key
        if base_key not in action_by_key:
            raise ValueError(f"Unexpected old Judge trial answer key: {base_key}")
        if current_answer_key != base_key:
            finish_trial_group()
            current_answer_key = base_key
            current_combinations = set()
        combination = (key[4], key[5])
        current_combinations.add(combination)
        target = targets[base_key[1]]
        for field in (
            "target_document_index",
            "target_citation_index",
            "candidate_document_count",
        ):
            if int(row[field]) != int(target[field]):
                raise ValueError(f"Old trial target metadata mismatch for {key}: {field}")
        if bool(row.get("target_source_cited")) != raw_target_by_key[base_key]:
            raise ValueError(f"Old trial target-citation mismatch: {key}")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError(f"Old trial has invalid score for {key}: {score!r}")
        old_trial_count += 1
        old_trial_action_counts[action_by_key[base_key]] += 1
        old_trial_origin_counts[str(row.get("score_origin", "missing"))] += 1
    finish_trial_group()
    if completed_trial_answer_keys != set(action_by_key):
        missing = sorted(set(action_by_key) - completed_trial_answer_keys)
        raise ValueError(f"Old Judge trials are missing answer groups: {missing[:5]}")
    validate_expected("old Judge trials", old_trial_count, expected_trial_count)

    rerun_trial_count = action_counts[ACTION_RERUN] * len(expected_combinations)
    validate_expected(
        "rerun Judge trials", rerun_trial_count, args.expected_rerun_trial_count
    )
    answer_alignment_rows.sort(key=answer_key)
    reuse_rows.sort(key=answer_key)
    rerun_rows.sort(key=answer_key)
    zero_rows.sort(key=answer_key)
    rerun_trial_rows = []
    for row in rerun_rows:
        key = answer_key(row)
        target = targets[key[1]]
        for dimension in DIMENSIONS:
            for judge_seed in args.seeds:
                rerun_trial_rows.append(
                    {
                        "model": key[0],
                        "sample_id": key[1],
                        "prompt_id": key[2],
                        "answer_seed": key[3],
                        "benchmark": str(row["benchmark"]),
                        "domain": str(row["domain"]),
                        **target,
                        "dimension": dimension,
                        "judge_seed": judge_seed,
                        "subjective_score_action": ACTION_RERUN,
                        "canonicalized_answer_sha256": str(
                            row["canonicalized_answer_sha256"]
                        ),
                    }
                )

    trial_action_counts = {
        ACTION_REUSE: action_counts[ACTION_REUSE] * len(expected_combinations),
        ACTION_RERUN: rerun_trial_count,
        ACTION_ZERO: action_counts[ACTION_ZERO] * len(expected_combinations),
    }
    metrics = {
        "experiment_id": args.experiment_id,
        "integrity_passed": True,
        "gpu_required": False,
        "llm_required": False,
        "llm_call_count": 0,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "answer_count": len(action_by_key),
        "dimension_count": len(DIMENSIONS),
        "judge_seeds": args.seeds,
        "trials_per_answer": len(expected_combinations),
        "total_trial_count": expected_trial_count,
        "answer_action_counts": dict(action_counts),
        "trial_action_counts": trial_action_counts,
        "target_citation_transition_counts": dict(transition_counts),
        "canonicalized_text_changed_counts": dict(changed_counts),
        "old_judge_score_count": len(old_score_keys),
        "old_judge_trial_count": old_trial_count,
        "old_trial_counts_by_final_action": dict(old_trial_action_counts),
        "old_trial_score_origin_counts": dict(old_trial_origin_counts),
        "old_score_target_mismatch_count": old_score_target_mismatch_count,
        "old_score_unscorable_count": old_score_unscorable_count,
        "old_absent_nonzero_average_count": old_absent_nonzero_average_count,
        "reuse_policy": (
            "reuse old RAID scores only when canonicalized text is unchanged "
            "and the canonicalized answer cites the target source"
        ),
        "rerun_policy": (
            "rerun all seven dimensions and configured Judge seeds when "
            "canonicalized text changed and the target source is cited"
        ),
        "zero_policy": (
            "assign deterministic zero to all dimensions and Judge seeds when "
            "the canonicalized answer does not cite the target source"
        ),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "output_files": {
            "answer_alignment": "answer_alignment.jsonl",
            "reuse_answer_manifest": "reuse_old_scores_answers.jsonl",
            "rerun_answer_manifest": "rerun_glm_judge_answers.jsonl",
            "rerun_trial_manifest": "rerun_glm_judge_trials.jsonl",
            "zero_answer_manifest": "deterministic_zero_answers.jsonl",
            "audit": "audit.json",
            "metrics": "metrics.json",
        },
    }
    audit = {
        "integrity_passed": True,
        "disjoint_action_partition": sum(action_counts.values()) == len(action_by_key),
        "raw_canonical_key_sets_equal": canonical_keys == set(raw_index),
        "old_score_key_set_equal": old_score_keys == set(action_by_key),
        "old_trial_answer_key_set_equal": completed_trial_answer_keys
        == set(action_by_key),
        "expected_answer_action_counts": {
            ACTION_REUSE: args.expected_reuse_answer_count,
            ACTION_RERUN: args.expected_rerun_answer_count,
            ACTION_ZERO: args.expected_zero_answer_count,
        },
        "observed_answer_action_counts": dict(action_counts),
        "expected_rerun_trial_count": args.expected_rerun_trial_count,
        "observed_rerun_trial_count": rerun_trial_count,
        "first_20_by_action": {
            ACTION_REUSE: reuse_rows[:20],
            ACTION_RERUN: [
                lightweight_manifest_row(
                    row,
                    action=ACTION_RERUN,
                    target=targets[answer_key(row)[1]],
                    raw_target_cited=bool(row["raw_target_source_cited"]),
                    canonical_target_cited=bool(row["canonical_target_source_cited"]),
                    text_changed=True,
                )
                for row in rerun_rows[:20]
            ],
            ACTION_ZERO: zero_rows[:20],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "answer_alignment.jsonl", answer_alignment_rows)
    write_jsonl(args.output_dir / "reuse_old_scores_answers.jsonl", reuse_rows)
    write_jsonl(args.output_dir / "rerun_glm_judge_answers.jsonl", rerun_rows)
    write_jsonl(args.output_dir / "rerun_glm_judge_trials.jsonl", rerun_trial_rows)
    write_jsonl(args.output_dir / "deterministic_zero_answers.jsonl", zero_rows)
    write_json(args.output_dir / "audit.json", audit)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
