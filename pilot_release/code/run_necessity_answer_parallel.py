"""Orchestrate three checkpointed Answer Model shards on disjoint GPU pairs."""

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

from answer_prompt import ANSWER_PROMPT_PROTOCOLS, BENCHMARK_OFFICIAL_V1
from model_loader import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_PAIRS = {"qwen": "0,1", "llama": "2,3", "mistral": "4,5"}


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
        raise argparse.ArgumentTypeError("GPU pair must contain two indices: e.g. 0,1")
    try:
        pair = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid GPU pair: {value}") from exc
    if pair[0] == pair[1] or min(pair) < 0:
        raise argparse.ArgumentTypeError(f"GPU pair must be two distinct nonnegative indices: {value}")
    return pair  # type: ignore[return-value]


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
        if len(row) != 2:
            continue
        memory[int(row[0].strip())] = int(row[1].strip())
    return memory


def gpu_uuid_to_index() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mapping: dict[str, int] = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 2:
            continue
        index = int(row[0].strip())
        uuid = row[1].strip()
        if not uuid:
            raise RuntimeError(f"Empty GPU UUID for physical index {index}")
        if uuid in mapping:
            raise RuntimeError(f"Duplicate GPU UUID reported by nvidia-smi: {uuid}")
        mapping[uuid] = index
    if not mapping:
        raise RuntimeError("nvidia-smi returned no GPU UUID inventory")
    return mapping


def active_compute_gpu_indices() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mapping = gpu_uuid_to_index()
    active_uuids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    unknown = active_uuids - set(mapping)
    if unknown:
        raise RuntimeError(
            f"Compute processes reported unknown GPU UUIDs: {sorted(unknown)}"
        )
    return {mapping[uuid] for uuid in active_uuids}


def require_selected_gpus_safe(
    *,
    selected: list[int],
    memory: dict[int, int],
    active_compute: set[int],
    max_preexisting_memory_mib: int,
) -> dict[str, Any]:
    missing = sorted(gpu for gpu in selected if gpu not in memory)
    over_memory = {
        gpu: memory[gpu]
        for gpu in selected
        if gpu in memory and memory[gpu] > max_preexisting_memory_mib
    }
    active_selected = sorted(set(selected) & active_compute)
    if missing or over_memory or active_selected:
        raise RuntimeError(
            "Refusing to start because selected GPUs are not safely idle: "
            f"missing={missing}, memory_over_limit_mib={over_memory}, "
            f"selected_gpus_with_existing_compute_processes={active_selected}. "
            "No existing process was modified."
        )
    return {
        "selected_memory_mib": {str(gpu): memory[gpu] for gpu in selected},
        "active_compute_gpu_indices": sorted(active_compute),
        "max_preexisting_memory_mib": max_preexisting_memory_mib,
    }


def gpu_safety_snapshot(
    selected: list[int], *, max_preexisting_memory_mib: int
) -> dict[str, Any]:
    return require_selected_gpus_safe(
        selected=selected,
        memory=gpu_memory_used(),
        active_compute=active_compute_gpu_indices(),
        max_preexisting_memory_mib=max_preexisting_memory_mib,
    )


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl",
    )
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--input-budget", type=int, default=20480)
    parser.add_argument(
        "--qwen-gpus", type=parse_gpu_pair, default=parse_gpu_pair("0,1")
    )
    parser.add_argument(
        "--llama-gpus", type=parse_gpu_pair, default=parse_gpu_pair("2,3")
    )
    parser.add_argument(
        "--mistral-gpus", type=parse_gpu_pair, default=parse_gpu_pair("4,5")
    )
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument("--allowed-error-types", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_pairs = {
        "qwen": args.qwen_gpus,
        "llama": args.llama_gpus,
        "mistral": args.mistral_gpus,
    }
    selected = [gpu for pair in gpu_pairs.values() for gpu in pair]
    if len(selected) != len(set(selected)):
        raise ValueError(f"GPU pairs must be disjoint: {gpu_pairs}")
    if set(gpu_pairs) != set(MODEL_SPECS):
        raise RuntimeError(
            f"GPU map and Answer Model panel differ: {gpu_pairs.keys()} vs {MODEL_SPECS.keys()}"
        )

    commands: dict[str, list[str]] = {}
    for model in MODEL_SPECS:
        command = [
            sys.executable,
            str(ROOT / "code/run_necessity_answer_model_shard.py"),
            "--experiment-id",
            args.experiment_id,
            "--answer-prompt-protocol",
            args.answer_prompt_protocol,
            "--model",
            model,
            "--manifest",
            str(args.manifest),
            "--rewrites",
            str(args.rewrites),
            "--model-root",
            str(args.model_root),
            "--output-dir",
            str(args.shards_root / model),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--input-budget",
            str(args.input_budget),
        ]
        if args.resume:
            command.append("--resume")
        commands[model] = command

    plan = {
        "experiment_id": args.experiment_id,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "gpu_pairs": {model: list(pair) for model, pair in gpu_pairs.items()},
        "commands": commands,
        "checkpointed": True,
        "parallel_models": True,
        "expected_generations": 39600,
        "allowed_error_types": sorted(set(args.allowed_error_types)),
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    initial_gpu_safety = gpu_safety_snapshot(
        selected,
        max_preexisting_memory_mib=args.max_preexisting_memory_mib,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.shards_root.mkdir(parents=True, exist_ok=True)
    state_path = args.shards_root / "orchestrator_state.json"
    children: dict[str, subprocess.Popen[Any]] = {}
    log_handles: list[Any] = []

    def terminate_owned_children(signum: int, _frame: Any) -> None:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_owned_children)
    signal.signal(signal.SIGINT, terminate_owned_children)
    try:
        final_gpu_safety = gpu_safety_snapshot(
            selected,
            max_preexisting_memory_mib=args.max_preexisting_memory_mib,
        )
        for model, command in commands.items():
            shard_dir = args.shards_root / model
            shard_dir.mkdir(parents=True, exist_ok=True)
            stdout_handle = (shard_dir / "stdout.log").open("a", encoding="utf-8")
            stderr_handle = (shard_dir / "stderr.log").open("a", encoding="utf-8")
            log_handles.extend((stdout_handle, stderr_handle))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu) for gpu in gpu_pairs[model]
            )
            children[model] = subprocess.Popen(
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
                "gpu_safety": {
                    "initial": initial_gpu_safety,
                    "immediately_before_worker_start": final_gpu_safety,
                    "modifies_existing_processes": False,
                },
                "pids": {model: process.pid for model, process in children.items()},
            },
        )
        return_codes = {model: process.wait() for model, process in children.items()}
    finally:
        for handle in log_handles:
            handle.close()

    completed_metrics: dict[str, dict[str, Any]] = {}
    for model in MODEL_SPECS:
        metrics_path = args.shards_root / model / "metrics.json"
        if metrics_path.exists():
            completed_metrics[model] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
    all_accounted = len(completed_metrics) == len(MODEL_SPECS) and all(
        bool(metrics.get("complete_accounting"))
        for metrics in completed_metrics.values()
    )
    atomic_json(
        state_path,
        {
            **plan,
            "status": "shards_finished" if all_accounted else "incomplete",
            "gpu_safety": {
                "initial": initial_gpu_safety,
                "immediately_before_worker_start": final_gpu_safety,
                "modifies_existing_processes": False,
            },
            "pids": {model: process.pid for model, process in children.items()},
            "return_codes": return_codes,
            "complete_accounting": all_accounted,
        },
    )
    if not all_accounted:
        return 1

    aggregate_command = [
        sys.executable,
        str(ROOT / "code/aggregate_necessity_answer_shards.py"),
        "--experiment-id",
        args.experiment_id,
        "--answer-prompt-protocol",
        args.answer_prompt_protocol,
        "--shards-root",
        str(args.shards_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.allowed_error_types:
        aggregate_command.extend(
            ["--allowed-error-types", *args.allowed_error_types]
        )
    aggregate = subprocess.run(aggregate_command, cwd=ROOT, check=False)
    return aggregate.returncode


if __name__ == "__main__":
    raise SystemExit(main())
