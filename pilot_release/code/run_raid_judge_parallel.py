"""Orchestrate checkpointed RAID judge shards on disjoint GPU pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_raid_judge_shard import (
    DIMENSIONS,
    answer_key,
    load_target_map,
    read_jsonl,
    resolve_answer_query,
    select_shard_answers,
    target_source_is_cited,
)


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_gpu_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("GPU pair must look like 0,1")
    try:
        pair = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid GPU pair: {value}") from exc
    if pair[0] == pair[1] or min(pair) < 0:
        raise argparse.ArgumentTypeError(
            f"GPU pair must contain two distinct nonnegative indices: {value}"
        )
    return pair


def gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory: dict[int, int] = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) == 2:
            memory[int(row[0].strip())] = int(row[1].strip())
    return memory


def unavailable_gpus(
    memory: dict[int, int],
    selected_gpus: list[int],
    max_preexisting_memory_mib: int,
) -> dict[int, int | None]:
    """Return selected GPUs that fail the frozen startup-memory guard."""

    return {
        gpu: memory.get(gpu)
        for gpu in selected_gpus
        if memory.get(gpu, max_preexisting_memory_mib + 1)
        > max_preexisting_memory_mib
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prompt-root", type=Path, default=ROOT / "prompts/evaluation/raid"
    )
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--gpu-pairs",
        nargs="+",
        type=parse_gpu_pair,
        default=[
            parse_gpu_pair("0,1"),
            parse_gpu_pair("2,3"),
            parse_gpu_pair("4,5"),
            parse_gpu_pair("6,7"),
        ],
    )
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument("--expected-answer-count", type=int, default=None)
    parser.add_argument("--limit-answers-per-shard", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-all-scores-parsed",
        action="store_true",
        help="Fail final aggregation if any saved Judge answer has no concrete score.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resource-check-only",
        action="store_true",
        help="Report the startup GPU guard without creating outputs or loading models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if len(args.gpu_pairs) != args.num_shards:
        raise ValueError(
            f"Expected {args.num_shards} GPU pairs, got {len(args.gpu_pairs)}"
        )
    selected_gpus = [gpu for pair in args.gpu_pairs for gpu in pair]
    if len(selected_gpus) != len(set(selected_gpus)):
        raise ValueError(f"GPU pairs must be disjoint: {args.gpu_pairs}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Judge seeds must be unique: {args.seeds}")
    if args.limit_answers_per_shard is not None and args.limit_answers_per_shard <= 0:
        raise ValueError("limit-answers-per-shard must be positive")

    if args.resource_check_only:
        memory = gpu_memory_used()
        unavailable = unavailable_gpus(
            memory,
            selected_gpus,
            args.max_preexisting_memory_mib,
        )
        payload = {
            "resource_check_only": True,
            "selected_gpus": selected_gpus,
            "max_preexisting_memory_mib": args.max_preexisting_memory_mib,
            "memory_used_mib": {
                str(gpu): memory.get(gpu) for gpu in selected_gpus
            },
            "unavailable_gpus": {
                str(gpu): used for gpu, used in unavailable.items()
            },
            "ready": not unavailable,
            "no_process_modified": True,
            "no_output_created": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["ready"] else 2

    answers = read_jsonl(args.input)
    answer_keys = [answer_key(row) for row in answers]
    if len(answer_keys) != len(set(answer_keys)):
        raise ValueError("Judge input contains duplicate Answer keys")
    if (
        args.expected_answer_count is not None
        and len(answers) != args.expected_answer_count
    ):
        raise ValueError(
            f"Expected {args.expected_answer_count} answers, found {len(answers)}"
        )
    target_map = load_target_map(args.manifest)
    missing_manifest_samples = sorted(
        {str(row["sample_id"]) for row in answers} - set(target_map)
    )
    if missing_manifest_samples:
        raise ValueError(
            f"Judge input contains {len(missing_manifest_samples)} samples missing "
            "from manifest"
        )
    query_source_counts = {"answer_verified": 0, "manifest_fallback": 0}
    for answer in answers:
        _, source = resolve_answer_query(
            answer, target_map[str(answer["sample_id"])]
        )
        query_source_counts[source] += 1
    shard_answers: list[list[dict[str, Any]]] = []
    for shard_index in range(args.num_shards):
        shard_answers.append(
            select_shard_answers(
                answers,
                shard_index=shard_index,
                num_shards=args.num_shards,
                limit_answers=args.limit_answers_per_shard,
            )
        )
    shard_answer_counts = [len(rows) for rows in shard_answers]
    shard_target_cited_answer_counts = [
        sum(
            target_source_is_cited(row, target_map[str(row["sample_id"])])
            for row in rows
        )
        for rows in shard_answers
    ]
    shard_target_absent_answer_counts = [
        total - cited
        for total, cited in zip(
            shard_answer_counts, shard_target_cited_answer_counts, strict=True
        )
    ]
    trials_per_answer = len(DIMENSIONS) * len(args.seeds)
    expected_trials_per_shard = [
        count * trials_per_answer for count in shard_answer_counts
    ]
    expected_glm_trials_per_shard = [
        count * trials_per_answer for count in shard_target_cited_answer_counts
    ]
    expected_deterministic_zero_trials_per_shard = [
        count * trials_per_answer for count in shard_target_absent_answer_counts
    ]
    expected_trials = sum(expected_trials_per_shard)

    commands: dict[int, list[str]] = {}
    for shard_index in range(args.num_shards):
        command = [
            sys.executable,
            str(ROOT / "code/run_raid_judge_shard.py"),
            "--experiment-id",
            args.experiment_id,
            "--input",
            str(args.input),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.shards_root / f"shard_{shard_index}"),
            "--prompt-root",
            str(args.prompt_root),
            "--model-root",
            str(args.model_root),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(args.num_shards),
        ]
        if args.resume:
            command.append("--resume")
        if args.limit_answers_per_shard is not None:
            command.extend(
                ["--limit-answers", str(args.limit_answers_per_shard)]
            )
        commands[shard_index] = command

    plan = {
        "experiment_id": args.experiment_id,
        "answer_count": len(answers),
        "num_shards": args.num_shards,
        "gpu_pairs": [list(pair) for pair in args.gpu_pairs],
        "shard_answer_counts": shard_answer_counts,
        "shard_target_cited_answer_counts": shard_target_cited_answer_counts,
        "shard_target_absent_answer_counts": shard_target_absent_answer_counts,
        "expected_trials_per_shard": expected_trials_per_shard,
        "expected_glm_trials_per_shard": expected_glm_trials_per_shard,
        "expected_deterministic_zero_trials_per_shard": (
            expected_deterministic_zero_trials_per_shard
        ),
        "expected_trials": expected_trials,
        "target_cited_answer_count": sum(shard_target_cited_answer_counts),
        "target_absent_answer_count": sum(shard_target_absent_answer_counts),
        "expected_glm_trials": sum(expected_glm_trials_per_shard),
        "expected_deterministic_zero_trials": sum(
            expected_deterministic_zero_trials_per_shard
        ),
        "dimension_count": len(DIMENSIONS),
        "judge_seeds": args.seeds,
        "checkpointed": True,
        "require_all_scores_parsed": args.require_all_scores_parsed,
        "score_output_protocol": "extract one unambiguous integer score from 0-5",
        "score_normalization": "store extracted score as integer",
        "query_policy": "verify answer query; otherwise use frozen manifest query",
        "query_source_counts": query_source_counts,
        "target_source_policy": "manifest target_document_index + 1",
        "target_absence_policy": (
            "if exact [target citation index] is absent, assign 0 to all "
            "dimensions/seeds without calling GLM"
        ),
        "commands": commands,
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    memory = gpu_memory_used()
    unavailable = unavailable_gpus(
        memory,
        selected_gpus,
        args.max_preexisting_memory_mib,
    )
    if unavailable:
        raise RuntimeError(
            "Refusing to start because selected GPUs already have substantial memory "
            f"use (MiB): {unavailable}. No existing process was modified."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.shards_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        nonempty_shards = [
            path
            for path in args.shards_root.iterdir()
            if path.is_dir() and any(path.iterdir())
        ]
        if nonempty_shards:
            raise FileExistsError(
                "Non-empty Judge shards require --resume: "
                + ", ".join(str(path) for path in nonempty_shards)
            )

    state_path = args.shards_root / "orchestrator_state.json"
    children: dict[int, subprocess.Popen[Any]] = {}
    log_handles: list[Any] = []

    def terminate_owned_children(signum: int, _frame: Any) -> None:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_owned_children)
    signal.signal(signal.SIGINT, terminate_owned_children)
    try:
        for shard_index, command in commands.items():
            shard_dir = args.shards_root / f"shard_{shard_index}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            stdout_handle = (shard_dir / "stdout.log").open("a", encoding="utf-8")
            stderr_handle = (shard_dir / "stderr.log").open("a", encoding="utf-8")
            log_handles.extend((stdout_handle, stderr_handle))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu) for gpu in args.gpu_pairs[shard_index]
            )
            children[shard_index] = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        atomic_json(
            state_path,
            {
                **plan,
                "status": "running",
                "pids": {
                    str(index): process.pid for index, process in children.items()
                },
            },
        )
        return_codes = {
            index: process.wait() for index, process in children.items()
        }
    finally:
        for handle in log_handles:
            handle.close()

    shard_metrics = []
    for shard_index in range(args.num_shards):
        metrics_path = args.shards_root / f"shard_{shard_index}" / "metrics.json"
        if metrics_path.exists():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    complete_accounting = len(shard_metrics) == args.num_shards and all(
        bool(metrics.get("complete_accounting")) for metrics in shard_metrics
    )
    atomic_json(
        state_path,
        {
            **plan,
            "status": "shards_finished" if complete_accounting else "incomplete",
            "pids": {
                str(index): process.pid for index, process in children.items()
            },
            "return_codes": {str(k): v for k, v in return_codes.items()},
            "complete_accounting": complete_accounting,
        },
    )
    if not complete_accounting:
        return 1

    aggregate_command = [
        sys.executable,
        str(ROOT / "code/aggregate_raid_judge_shards.py"),
        "--experiment-id",
        args.experiment_id,
        "--input",
        str(args.input),
        "--manifest",
        str(args.manifest),
        "--shards-root",
        str(args.shards_root),
        "--output-dir",
        str(args.output_dir),
        "--num-shards",
        str(args.num_shards),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--zero-when-target-not-cited",
    ]
    if args.limit_answers_per_shard is not None:
        aggregate_command.extend(
            ["--limit-answers-per-shard", str(args.limit_answers_per_shard)]
        )
    if args.require_all_scores_parsed:
        aggregate_command.append("--require-all-scores-parsed")
    aggregate = subprocess.run(aggregate_command, cwd=ROOT, check=False)
    return aggregate.returncode


if __name__ == "__main__":
    raise SystemExit(main())
