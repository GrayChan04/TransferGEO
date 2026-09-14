"""Merge reused, rerun, and deterministic-zero RAID Judge trials.

The citation-canonicalized v7 subjective result is assembled without changing
any historical output:

* unchanged target-citing answers reuse their old Judge trials;
* changed target-citing answers use newly generated rerun shard trials;
* answers without the target citation receive deterministic zero trials.

All sources are checked as disjoint, complete, sorted trial streams before the
final per-dimension and Subjective Average records are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
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
    parser.add_argument("--canonicalized-answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--old-judge-trials", type=Path, required=True)
    parser.add_argument("--rerun-shards-root", type=Path, required=True)
    parser.add_argument("--num-rerun-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-answer-count", type=int, required=True)
    parser.add_argument("--expected-trial-count", type=int, required=True)
    parser.add_argument("--expected-reuse-trial-count", type=int, required=True)
    parser.add_argument("--expected-rerun-trial-count", type=int, required=True)
    parser.add_argument("--expected-zero-trial-count", type=int, required=True)
    parser.add_argument(
        "--source-old-experiment-id",
        default="unified_answer_prompt_v7_raid_judge_full_v1",
    )
    parser.add_argument(
        "--source-rerun-experiment-id",
        default="unified_answer_prompt_v7_raid_judge_canonicalized_v3_rerun_v1",
    )
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_targets(path: Path) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in targets:
            raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
        target_document_index = int(row["target_document_index"])
        candidate_document_count = int(row["candidate_document_count"])
        if not 0 <= target_document_index < candidate_document_count:
            raise ValueError(f"Invalid target metadata for {sample_id}")
        targets[sample_id] = {
            "target_document_index": target_document_index,
            "target_citation_index": target_document_index + 1,
            "candidate_document_count": candidate_document_count,
        }
    return targets


def load_alignment(path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    alignment: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = answer_key(row)
        if key in alignment:
            raise ValueError(f"Duplicate alignment key: {key}")
        action = str(row.get("subjective_score_action"))
        if action not in {ACTION_REUSE, ACTION_RERUN, ACTION_ZERO}:
            raise ValueError(f"Unknown alignment action for {key}: {action}")
        alignment[key] = row
    return alignment


def validate_trial(
    row: dict[str, Any],
    *,
    action: str,
    metadata: dict[str, Any],
    expected_combinations: set[tuple[str, int]],
) -> None:
    key = trial_key(row)
    if (key[4], key[5]) not in expected_combinations:
        raise ValueError(f"Unexpected dimension/seed: {key}")
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
        raise ValueError(f"Invalid score for {key}: {score!r}")
    for field in (
        "target_document_index",
        "target_citation_index",
        "candidate_document_count",
    ):
        if int(row[field]) != int(metadata[field]):
            raise ValueError(f"Target metadata mismatch for {key}: {field}")
    expected_target_cited = action != ACTION_ZERO
    if bool(row.get("target_source_cited")) != expected_target_cited:
        raise ValueError(f"Target-citation flag mismatch for {key}")


def tagged_sorted_stream(
    rows: Iterable[dict[str, Any]],
    *,
    action: str,
    alignment: dict[tuple[str, str, str, int], dict[str, Any]],
    answer_metadata: dict[tuple[str, str, str, int], dict[str, Any]],
    expected_combinations: set[tuple[str, int]],
    source_experiment_id: str,
) -> Iterator[dict[str, Any]]:
    previous: tuple[str, str, str, int, str, int] | None = None
    for row in rows:
        key = trial_key(row)
        base_key = key[:4]
        if alignment.get(base_key, {}).get("subjective_score_action") != action:
            continue
        if previous is not None and key <= previous:
            raise ValueError(
                f"{action} input trials are not strictly sorted: {key}"
            )
        previous = key
        metadata = answer_metadata[base_key]
        validate_trial(
            row,
            action=action,
            metadata=metadata,
            expected_combinations=expected_combinations,
        )
        yield {
            **row,
            "target_source_cited": True,
            "score_alignment_origin": action,
            "source_judge_experiment_id": source_experiment_id,
            "canonicalized_answer_sha256": metadata[
                "canonicalized_answer_sha256"
            ],
        }


def zero_trial_stream(
    *,
    zero_keys: list[tuple[str, str, str, int]],
    answer_metadata: dict[tuple[str, str, str, int], dict[str, Any]],
    seeds: list[int],
) -> Iterator[dict[str, Any]]:
    for key in zero_keys:
        metadata = answer_metadata[key]
        for dimension in DIMENSIONS:
            for judge_seed in seeds:
                yield {
                    "model": key[0],
                    "sample_id": key[1],
                    "prompt_id": key[2],
                    "answer_seed": key[3],
                    "benchmark": metadata["benchmark"],
                    "domain": metadata["domain"],
                    "target_document_index": metadata["target_document_index"],
                    "target_citation_index": metadata["target_citation_index"],
                    "candidate_document_count": metadata[
                        "candidate_document_count"
                    ],
                    "target_source_cited": False,
                    "dimension": dimension,
                    "judge_seed": judge_seed,
                    "raw_output": None,
                    "answer": None,
                    "score": 0,
                    "score_parse_method": "deterministic_target_absence",
                    "score_origin": "deterministic_target_absence",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "score_alignment_origin": ACTION_ZERO,
                    "source_judge_experiment_id": None,
                    "canonicalized_answer_sha256": metadata[
                        "canonicalized_answer_sha256"
                    ],
                }


def score_row(
    *,
    key: tuple[str, str, str, int],
    trials: list[dict[str, Any]],
    metadata: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["dimension"])].append(row)
    dimensions: dict[str, Any] = {}
    means: list[float] = []
    for dimension in DIMENSIONS:
        dimension_trials = sorted(
            grouped[dimension], key=lambda row: int(row["judge_seed"])
        )
        observed_seeds = [int(row["judge_seed"]) for row in dimension_trials]
        if observed_seeds != seeds:
            raise ValueError(
                f"Incomplete Judge seeds for {key}/{dimension}: {observed_seeds}"
            )
        scores = [int(row["score"]) for row in dimension_trials]
        mean_score = sum(scores) / len(scores)
        means.append(mean_score)
        dimensions[dimension] = {
            "trials": dimension_trials,
            "valid_scores": scores,
            "mean_score": mean_score,
            "scorable": True,
        }
    return {
        "model": key[0],
        "sample_id": key[1],
        "prompt_id": key[2],
        "answer_seed": key[3],
        "benchmark": metadata["benchmark"],
        "domain": metadata["domain"],
        "query": metadata["query"],
        "target_document_index": metadata["target_document_index"],
        "target_citation_index": metadata["target_citation_index"],
        "candidate_document_count": metadata["candidate_document_count"],
        "target_source_cited": metadata["target_source_cited"],
        "subjective_score_action": metadata["subjective_score_action"],
        "canonicalized_answer_sha256": metadata[
            "canonicalized_answer_sha256"
        ],
        "dimensions": dimensions,
        "subjective_average": sum(means) / len(means),
        "all_dimensions_scorable": True,
    }


def main() -> int:
    args = parse_args()
    if not args.seeds or sorted(set(args.seeds)) != args.seeds:
        raise ValueError(f"Seeds must be sorted and unique: {args.seeds}")
    if args.num_rerun_shards <= 0:
        raise ValueError("num-rerun-shards must be positive")
    input_paths = [
        args.canonicalized_answers,
        args.manifest,
        args.alignment_manifest,
        args.old_judge_trials,
    ]
    rerun_trial_paths = [
        args.rerun_shards_root / f"shard_{index}" / "trials.jsonl"
        for index in range(args.num_rerun_shards)
    ]
    rerun_error_paths = [
        args.rerun_shards_root / f"shard_{index}" / "errors.jsonl"
        for index in range(args.num_rerun_shards)
    ]
    for path in [*input_paths, *rerun_trial_paths, *rerun_error_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    rerun_error_count = sum(sum(1 for _ in iter_jsonl(path)) for path in rerun_error_paths)
    if rerun_error_count:
        raise ValueError(f"Rerun shards contain {rerun_error_count} Judge errors")

    targets = load_targets(args.manifest)
    alignment = load_alignment(args.alignment_manifest)
    if len(alignment) != args.expected_answer_count:
        raise ValueError(
            f"Expected {args.expected_answer_count} alignment rows, found {len(alignment)}"
        )
    action_answer_counts = Counter(
        str(row["subjective_score_action"]) for row in alignment.values()
    )

    answer_metadata: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in iter_jsonl(args.canonicalized_answers):
        key = answer_key(row)
        if key in answer_metadata:
            raise ValueError(f"Duplicate canonicalized answer key: {key}")
        alignment_row = alignment.get(key)
        if alignment_row is None:
            raise ValueError(f"Canonicalized answer missing from alignment: {key}")
        target = targets.get(key[1])
        if target is None:
            raise ValueError(f"Canonicalized answer sample missing from manifest: {key[1]}")
        answer = row.get("answer")
        query = row.get("query")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"Missing canonicalized answer text: {key}")
        if not isinstance(query, str) or not query:
            raise ValueError(f"Missing canonicalized answer query: {key}")
        answer_hash = sha256_text(answer)
        if answer_hash != str(alignment_row["canonicalized_answer_sha256"]):
            raise ValueError(f"Canonicalized answer/alignment hash mismatch: {key}")
        target_cited = target_is_cited(answer, target["target_citation_index"])
        # Incremental repair plans use target_source_cited; retain the original
        # audit schema as a supported input without rewriting historical files.
        citation_flag = alignment_row.get(
            "canonical_target_source_cited", alignment_row.get("target_source_cited")
        )
        if citation_flag is None or target_cited != bool(citation_flag):
            raise ValueError(f"Canonical target-citation mismatch: {key}")
        action = str(alignment_row["subjective_score_action"])
        if (action == ACTION_ZERO) == target_cited:
            raise ValueError(f"Action/target-citation contradiction: {key}")
        answer_metadata[key] = {
            "benchmark": str(row["benchmark"]),
            "domain": str(row["domain"]),
            "query": query,
            **target,
            "target_source_cited": target_cited,
            "subjective_score_action": action,
            "canonicalized_answer_sha256": answer_hash,
        }
    if set(answer_metadata) != set(alignment):
        missing = sorted(set(alignment) - set(answer_metadata))
        raise ValueError(f"Canonicalized answers missing keys: {missing[:5]}")

    expected_combinations = {
        (dimension, seed) for dimension in DIMENSIONS for seed in args.seeds
    }
    old_stream = tagged_sorted_stream(
        iter_jsonl(args.old_judge_trials),
        action=ACTION_REUSE,
        alignment=alignment,
        answer_metadata=answer_metadata,
        expected_combinations=expected_combinations,
        source_experiment_id=args.source_old_experiment_id,
    )
    rerun_streams = [
        tagged_sorted_stream(
            iter_jsonl(path),
            action=ACTION_RERUN,
            alignment=alignment,
            answer_metadata=answer_metadata,
            expected_combinations=expected_combinations,
            source_experiment_id=args.source_rerun_experiment_id,
        )
        for path in rerun_trial_paths
    ]
    merged_rerun_stream = heapq.merge(*rerun_streams, key=trial_key)
    zero_keys = sorted(
        key
        for key, row in alignment.items()
        if row["subjective_score_action"] == ACTION_ZERO
    )
    zero_stream = zero_trial_stream(
        zero_keys=zero_keys,
        answer_metadata=answer_metadata,
        seeds=args.seeds,
    )
    all_trials = heapq.merge(
        old_stream,
        merged_rerun_stream,
        zero_stream,
        key=trial_key,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "judge_trials.jsonl"
    scores_path = args.output_dir / "judge_scores.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    errors_path.write_text("", encoding="utf-8")
    expected_answer_keys = sorted(answer_metadata)
    answer_index = 0
    current_key: tuple[str, str, str, int] | None = None
    current_trials: list[dict[str, Any]] = []
    previous_trial_key: tuple[str, str, str, int, str, int] | None = None
    action_trial_counts: Counter[str] = Counter()
    score_origin_counts: Counter[str] = Counter()
    input_tokens_by_action: Counter[str] = Counter()
    output_tokens_by_action: Counter[str] = Counter()
    completed_trial_count = 0

    def finish_answer(scores_handle: Any) -> None:
        nonlocal answer_index, current_trials
        if current_key is None:
            return
        if answer_index >= len(expected_answer_keys):
            raise ValueError(f"Unexpected extra answer trial group: {current_key}")
        expected_key = expected_answer_keys[answer_index]
        if current_key != expected_key:
            raise ValueError(
                f"Answer trial group mismatch: expected {expected_key}, found {current_key}"
            )
        combinations = {
            (str(row["dimension"]), int(row["judge_seed"]))
            for row in current_trials
        }
        if combinations != expected_combinations or len(current_trials) != len(
            expected_combinations
        ):
            raise ValueError(f"Incomplete final trials for {current_key}")
        scores_handle.write(
            json.dumps(
                score_row(
                    key=current_key,
                    trials=current_trials,
                    metadata=answer_metadata[current_key],
                    seeds=args.seeds,
                ),
                ensure_ascii=False,
            )
            + "\n"
        )
        answer_index += 1
        current_trials = []

    with trials_path.open("w", encoding="utf-8") as trials_handle, scores_path.open(
        "w", encoding="utf-8"
    ) as scores_handle:
        for row in all_trials:
            key = trial_key(row)
            base_key = key[:4]
            if previous_trial_key is not None and key <= previous_trial_key:
                raise ValueError(f"Duplicate or unsorted final trial key: {key}")
            previous_trial_key = key
            if current_key != base_key:
                finish_answer(scores_handle)
                current_key = base_key
            action = str(row["score_alignment_origin"])
            expected_action = answer_metadata[base_key]["subjective_score_action"]
            if action != expected_action:
                raise ValueError(f"Final trial action mismatch for {key}")
            validate_trial(
                row,
                action=action,
                metadata=answer_metadata[base_key],
                expected_combinations=expected_combinations,
            )
            trials_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            current_trials.append(row)
            action_trial_counts[action] += 1
            score_origin_counts[str(row.get("score_origin", "missing"))] += 1
            input_tokens_by_action[action] += int(row.get("input_tokens") or 0)
            output_tokens_by_action[action] += int(row.get("output_tokens") or 0)
            completed_trial_count += 1
        finish_answer(scores_handle)

    if answer_index != len(expected_answer_keys):
        raise ValueError(
            f"Expected {len(expected_answer_keys)} final answer groups, found {answer_index}"
        )
    if completed_trial_count != args.expected_trial_count:
        raise ValueError(
            f"Expected {args.expected_trial_count} final trials, found {completed_trial_count}"
        )
    expected_action_trial_counts = {
        ACTION_REUSE: args.expected_reuse_trial_count,
        ACTION_RERUN: args.expected_rerun_trial_count,
        ACTION_ZERO: args.expected_zero_trial_count,
    }
    if dict(action_trial_counts) != expected_action_trial_counts:
        raise ValueError(
            f"Final action trial counts mismatch: {dict(action_trial_counts)}"
        )

    input_hash_paths = [*input_paths, *rerun_trial_paths, *rerun_error_paths]
    metrics = {
        "experiment_id": args.experiment_id,
        "integrity_passed": True,
        "gpu_required": False,
        "llm_required": False,
        "answer_count": len(answer_metadata),
        "dimension_count": len(DIMENSIONS),
        "judge_seeds": args.seeds,
        "trials_per_answer": len(expected_combinations),
        "completed_trial_count": completed_trial_count,
        "fully_scorable_answer_count": answer_index,
        "unscorable_answer_count": 0,
        "rerun_error_count": rerun_error_count,
        "answer_action_counts": dict(action_answer_counts),
        "trial_action_counts": dict(action_trial_counts),
        "score_origin_counts": dict(score_origin_counts),
        "input_tokens_by_action": dict(input_tokens_by_action),
        "output_tokens_by_action": dict(output_tokens_by_action),
        "source_old_experiment_id": args.source_old_experiment_id,
        "source_rerun_experiment_id": args.source_rerun_experiment_id,
        "citation_canonicalization_version": "deterministic_citation_canonicalization_v3",
        "subjective_score_version": "raid_subjective_canonicalized_v3_v1",
        "input_sha256": {
            str(path): sha256_file(path) for path in input_hash_paths
        },
        "result_paths": {
            "judge_trials": str(trials_path),
            "judge_scores": str(scores_path),
            "errors": str(errors_path),
            "audit": str(args.output_dir / "audit.json"),
            "metrics": str(args.output_dir / "metrics.json"),
        },
        "provides_h1_h2_test": False,
    }
    audit = {
        "integrity_passed": True,
        "answer_key_sets_equal": set(answer_metadata) == set(alignment),
        "final_answer_order_complete": answer_index == len(expected_answer_keys),
        "final_trial_count_complete": completed_trial_count
        == args.expected_trial_count,
        "action_trial_counts_match": dict(action_trial_counts)
        == expected_action_trial_counts,
        "rerun_error_count": rerun_error_count,
        "expected_action_trial_counts": expected_action_trial_counts,
        "observed_action_trial_counts": dict(action_trial_counts),
    }
    write_json(args.output_dir / "audit.json", audit)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
