"""Wait for and safely reserve an idle physical-GPU pair for one RAID shard."""

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


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def acquire_lock(path: Path, *, nonblocking: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        fcntl.flock(handle.fileno(), flags)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another process owns lock {path}") from exc
    return handle


def parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def query_gpu_snapshot() -> dict[str, Any]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for raw_line in inventory.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise ValueError(f"Malformed GPU inventory row: {raw_line!r}")
        index = int(fields[0])
        uuid_to_index[fields[1]] = index
        gpus[index] = {
            "index": index,
            "uuid": fields[1],
            "memory_used_mib": parse_int(fields[2]),
            "memory_total_mib": parse_int(fields[3]),
            "utilization_gpu_percent": parse_int(fields[4]),
            "compute_processes": [],
        }
    for raw_line in processes.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise ValueError(f"Malformed GPU process row: {raw_line!r}")
        index = uuid_to_index.get(fields[0])
        if index is None:
            continue
        gpus[index]["compute_processes"].append(
            {
                "pid": parse_int(fields[1]),
                "used_memory_mib": parse_int(fields[2]),
                "name": fields[3],
            }
        )
    return {"updated_at": now(), "gpus": gpus}


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prune_reservations(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("reservations", {})
    if not isinstance(raw, dict):
        return {}
    reservations: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        pid = value.get("pid")
        if isinstance(pid, int) and pid_is_alive(pid):
            reservations[str(key)] = value
    return reservations


def select_safe_gpus(
    snapshot: dict[str, Any],
    *,
    candidate_gpus: list[int],
    reserved_gpus: set[int],
    gpus_per_shard: int,
    max_preexisting_memory_mib: int,
) -> list[int] | None:
    gpus = snapshot.get("gpus", {})
    safe: list[int] = []
    for index in sorted(candidate_gpus):
        gpu = gpus.get(index)
        if not isinstance(gpu, dict) or index in reserved_gpus:
            continue
        memory_used = gpu.get("memory_used_mib")
        compute_processes = gpu.get("compute_processes")
        if (
            isinstance(memory_used, int)
            and memory_used <= max_preexisting_memory_mib
            and isinstance(compute_processes, list)
            and not compute_processes
        ):
            safe.append(index)
    return safe[:gpus_per_shard] if len(safe) >= gpus_per_shard else None


def release_own_reservation(
    reservation_state_path: Path,
    reservation_lock_path: Path,
    reservation_key: str,
) -> None:
    with acquire_lock(reservation_lock_path, nonblocking=False):
        payload = (
            json.loads(reservation_state_path.read_text(encoding="utf-8"))
            if reservation_state_path.exists()
            else {}
        )
        reservations = prune_reservations(payload)
        reservations.pop(reservation_key, None)
        atomic_json(
            reservation_state_path,
            {"updated_at": now(), "reservations": reservations},
        )


def try_reserve_gpus(args: argparse.Namespace) -> tuple[list[int] | None, dict[str, Any]]:
    snapshot = query_gpu_snapshot()
    reservation_key = str(args.shard_index)
    with acquire_lock(args.reservation_lock_path, nonblocking=False):
        payload = (
            json.loads(args.reservation_state_path.read_text(encoding="utf-8"))
            if args.reservation_state_path.exists()
            else {}
        )
        reservations = prune_reservations(payload)
        existing = reservations.get(reservation_key)
        if existing is not None and int(existing.get("pid", -1)) != os.getpid():
            raise RuntimeError(
                f"Shard {args.shard_index} is already reserved by PID "
                f"{existing.get('pid')}"
            )
        reserved_gpus = {
            int(gpu)
            for key, value in reservations.items()
            if key != reservation_key
            for gpu in value.get("physical_gpus", [])
        }
        selected = select_safe_gpus(
            snapshot,
            candidate_gpus=args.candidate_gpus,
            reserved_gpus=reserved_gpus,
            gpus_per_shard=args.gpus_per_shard,
            max_preexisting_memory_mib=args.max_preexisting_memory_mib,
        )
        if selected is not None:
            reservations[reservation_key] = {
                "pid": os.getpid(),
                "shard_index": args.shard_index,
                "physical_gpus": selected,
                "reserved_at": now(),
            }
        atomic_json(
            args.reservation_state_path,
            {"updated_at": now(), "reservations": reservations},
        )
    return selected, snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-root", type=Path, default=ROOT / "prompts/evaluation/raid"
    )
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--candidate-gpus", nargs="+", type=int, required=True)
    parser.add_argument("--gpus-per-shard", type=int, default=2)
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--monitor-state-path", type=Path, required=True)
    parser.add_argument("--worker-lock-path", type=Path, required=True)
    parser.add_argument("--reservation-state-path", type=Path, required=True)
    parser.add_argument("--reservation-lock-path", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if len(args.candidate_gpus) != len(set(args.candidate_gpus)):
        raise ValueError("candidate-gpus must be unique")
    if args.gpus_per_shard <= 0:
        raise ValueError("gpus-per-shard must be positive")
    if args.poll_interval_seconds <= 0:
        raise ValueError("poll-interval-seconds must be positive")

    worker_lock = acquire_lock(args.worker_lock_path, nonblocking=True)
    worker_lock.write(f"pid={os.getpid()}\n")
    worker_lock.flush()
    selected: list[int] | None = None
    snapshot: dict[str, Any] = {}
    while True:
        while selected is None:
            try:
                selected, snapshot = try_reserve_gpus(args)
            except Exception as exc:
                atomic_json(
                    args.monitor_state_path,
                    {
                        "updated_at": now(),
                        "status": "gpu_query_or_reservation_failed",
                        "shard_index": args.shard_index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "read_only_gpu_checks": True,
                    },
                )
                time.sleep(args.poll_interval_seconds)
                continue
            if selected is None:
                atomic_json(
                    args.monitor_state_path,
                    {
                        "updated_at": now(),
                        "status": "waiting_for_two_safe_gpus",
                        "shard_index": args.shard_index,
                        "candidate_gpus": args.candidate_gpus,
                        "gpu_snapshot": snapshot,
                        "read_only_gpu_checks": True,
                        "existing_processes_modified": False,
                    },
                )
                time.sleep(args.poll_interval_seconds)

        time.sleep(2)
        verification = query_gpu_snapshot()
        verified = select_safe_gpus(
            verification,
            candidate_gpus=selected,
            reserved_gpus=set(),
            gpus_per_shard=args.gpus_per_shard,
            max_preexisting_memory_mib=args.max_preexisting_memory_mib,
        )
        if verified == selected:
            break
        release_own_reservation(
            args.reservation_state_path,
            args.reservation_lock_path,
            str(args.shard_index),
        )
        atomic_json(
            args.monitor_state_path,
            {
                "updated_at": now(),
                "status": "gpu_pair_changed_retrying",
                "shard_index": args.shard_index,
                "selected_physical_gpus": selected,
                "verified_physical_gpus": verified,
                "read_only_gpu_checks": True,
                "existing_processes_modified": False,
            },
        )
        selected = None
        time.sleep(args.poll_interval_seconds)

    atomic_json(
        args.monitor_state_path,
        {
            "updated_at": now(),
            "status": "gpu_pair_reserved_starting_shard",
            "shard_index": args.shard_index,
            "physical_gpus": selected,
            "read_only_gpu_checks": True,
            "existing_processes_modified": False,
        },
    )
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
        str(args.output_dir),
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
        str(args.shard_index),
        "--num-shards",
        str(args.num_shards),
        "--physical-gpus",
        *[str(gpu) for gpu in selected],
    ]
    if args.resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in selected)
    os.set_inheritable(worker_lock.fileno(), True)
    os.execvpe(command[0], command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
