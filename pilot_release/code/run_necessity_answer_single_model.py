"""Safely launch one checkpointed Answer Model shard on one physical GPU pair."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from answer_prompt import ANSWER_PROMPT_PROTOCOLS, BENCHMARK_OFFICIAL_V1
from model_loader import MODEL_SPECS
from run_necessity_answer_parallel import (
    active_compute_gpu_indices,
    atomic_json,
    gpu_memory_used,
    parse_gpu_pair,
    require_selected_gpus_safe,
)


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
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
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--input-budget", type=int, default=20480)
    parser.add_argument("--allowed-error-types", nargs="*", default=[])
    parser.add_argument("--gpus", type=parse_gpu_pair, required=True)
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--monitor-state-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def build_worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "code/run_necessity_answer_model_shard.py"),
        "--experiment-id",
        args.experiment_id,
        "--answer-prompt-protocol",
        args.answer_prompt_protocol,
        "--model",
        args.model,
        "--manifest",
        str(args.manifest),
        "--rewrites",
        str(args.rewrites),
        "--model-root",
        str(args.model_root),
        "--output-dir",
        str(args.output_dir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--input-budget",
        str(args.input_budget),
    ]
    if args.allowed_error_types:
        command.extend(["--allowed-error-types", *args.allowed_error_types])
    if args.resume:
        command.append("--resume")
    return command


def build_launch_plan(
    args: argparse.Namespace,
    *,
    worker_command: list[str],
    worker_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "answer_prompt_protocol": args.answer_prompt_protocol,
        "model": args.model,
        "physical_gpus": list(args.gpus),
        "max_preexisting_memory_mib": args.max_preexisting_memory_mib,
        "requires_no_preexisting_compute_process": True,
        "modifies_existing_processes": False,
        "checkpointed": True,
        "allowed_error_types": sorted(set(args.allowed_error_types)),
        "resume": args.resume,
        "monitoring": {
            "enabled": args.wait_for_gpus,
            "poll_interval_seconds": args.poll_interval_seconds,
            "monitor_state_path": (
                str(args.monitor_state_path) if args.monitor_state_path else None
            ),
            "maximum_wait_seconds": None,
            "preoccupies_gpu_while_waiting": False,
        },
        "worker_command": worker_command,
    }
    if worker_plan is not None:
        plan["worker_plan"] = worker_plan
    return plan


def acquire_monitor_lock(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another GPU monitor already owns {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def check_selected_gpus(
    selected: list[int], *, max_preexisting_memory_mib: int
) -> tuple[dict[str, Any], dict[int, int], set[int]]:
    memory = gpu_memory_used()
    active_compute = active_compute_gpu_indices()
    safety = require_selected_gpus_safe(
        selected=selected,
        memory=memory,
        active_compute=active_compute,
        max_preexisting_memory_mib=max_preexisting_memory_mib,
    )
    return safety, memory, active_compute


def wait_for_two_safe_checks(
    args: argparse.Namespace,
    *,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], TextIO | None]:
    selected = list(args.gpus)
    monitor_state_path = args.monitor_state_path or (
        ROOT
        / "results/diagnostics/gpu_monitor"
        / f"{args.experiment_id}__{args.model}.json"
    )
    monitor_lock_path = monitor_state_path.with_suffix(".lock")
    monitor_lock = (
        acquire_monitor_lock(monitor_lock_path) if args.wait_for_gpus else None
    )
    check_count = 0
    while True:
        check_count += 1
        memory: dict[int, int] = {}
        active_compute: set[int] = set()
        try:
            initial_safety, memory, active_compute = check_selected_gpus(
                selected,
                max_preexisting_memory_mib=args.max_preexisting_memory_mib,
            )
            final_safety, final_memory, final_active_compute = check_selected_gpus(
                selected,
                max_preexisting_memory_mib=args.max_preexisting_memory_mib,
            )
        except RuntimeError as exc:
            if not args.wait_for_gpus:
                if monitor_lock is not None:
                    monitor_lock.close()
                raise
            waiting = {
                **plan,
                "status": "waiting_for_requested_safe_gpu_pair",
                "updated_at": now(),
                "check_count": check_count,
                "observed_gpu_memory_mib": dict(sorted(memory.items())),
                "active_compute_gpu_indices": sorted(active_compute),
                "last_gate_message": str(exc),
                "model_weights_loaded": False,
                "modifies_existing_processes": False,
                "preoccupies_gpu_while_waiting": False,
            }
            atomic_json(monitor_state_path, waiting)
            print(json.dumps(waiting, ensure_ascii=False), flush=True)
            time.sleep(args.poll_interval_seconds)
            continue

        ready = {
            **plan,
            "status": "safe_gpu_pair_selected",
            "updated_at": now(),
            "check_count": check_count,
            "observed_gpu_memory_mib_initial": dict(sorted(memory.items())),
            "active_compute_gpu_indices_initial": sorted(active_compute),
            "observed_gpu_memory_mib_final": dict(sorted(final_memory.items())),
            "active_compute_gpu_indices_final": sorted(final_active_compute),
            "model_weights_loaded": False,
            "modifies_existing_processes": False,
        }
        atomic_json(monitor_state_path, ready)
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        return initial_safety, final_safety, monitor_lock


def main() -> int:
    args = parse_args()
    if args.poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    worker_command = build_worker_command(args)
    plan = build_launch_plan(args, worker_command=worker_command)
    if args.plan_only:
        worker_plan_result = subprocess.run(
            [*worker_command, "--plan-only"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        worker_plan = json.loads(worker_plan_result.stdout)
        print(
            json.dumps(
                build_launch_plan(
                    args,
                    worker_command=worker_command,
                    worker_plan=worker_plan,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty {args.output_dir}; use --resume only "
            "after explicitly approving continuation of this shard."
        )
    initial_gpu_safety, final_gpu_safety, monitor_lock = wait_for_two_safe_checks(
        args,
        plan=plan,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launch_record = {
        **plan,
        "status": "launching",
        "gpu_safety": {
            "initial": initial_gpu_safety,
            "immediately_before_worker_start": final_gpu_safety,
        },
    }
    atomic_json(args.output_dir / "launch_safety.json", launch_record)
    print(json.dumps(launch_record, ensure_ascii=False), flush=True)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    if monitor_lock is not None:
        # Python file descriptors are close-on-exec, so the lock is released
        # only when this process becomes the real model worker.
        monitor_lock.flush()
    os.execvpe(worker_command[0], worker_command, environment)
    raise AssertionError("os.execvpe unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
