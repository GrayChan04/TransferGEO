from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_necessity_answer_single_model import (  # noqa: E402
    build_launch_plan,
    build_worker_command,
)


def arguments() -> argparse.Namespace:
    return argparse.Namespace(
        experiment_id="unified_answer_prompt_v7_generation_full_v1",
        answer_prompt_protocol="unified_answer_v7",
        model="qwen",
        manifest=Path("manifest.jsonl"),
        rewrites=Path("rewrites.jsonl"),
        model_root=Path("models"),
        output_dir=Path("shards/qwen"),
        seeds=[0, 1, 2],
        max_new_tokens=2048,
        input_budget=20480,
        allowed_error_types=[],
        gpus=(0, 3),
        max_preexisting_memory_mib=1024,
        wait_for_gpus=False,
        poll_interval_seconds=30,
        monitor_state_path=None,
        resume=True,
    )


class SingleModelLauncherTest(unittest.TestCase):
    def test_worker_command_preserves_frozen_generation_parameters(self) -> None:
        command = build_worker_command(arguments())
        self.assertIn("unified_answer_v7", command)
        self.assertIn("qwen", command)
        self.assertEqual(command[command.index("--seeds") + 1 : command.index("--max-new-tokens")], ["0", "1", "2"])
        self.assertEqual(command[command.index("--input-budget") + 1], "20480")
        self.assertEqual(command[-1], "--resume")

    def test_worker_command_passes_predeclared_protocol_exclusions(self) -> None:
        args = arguments()
        args.allowed_error_types = ["input_budget_overflow"]
        command = build_worker_command(args)
        self.assertEqual(
            command[
                command.index("--allowed-error-types") + 1 : command.index("--resume")
            ],
            ["input_budget_overflow"],
        )

    def test_launch_plan_records_physical_pair_and_process_policy(self) -> None:
        args = arguments()
        command = build_worker_command(args)
        plan = build_launch_plan(args, worker_command=command)
        self.assertEqual(plan["physical_gpus"], [0, 3])
        self.assertTrue(plan["requires_no_preexisting_compute_process"])
        self.assertFalse(plan["modifies_existing_processes"])
        self.assertTrue(plan["checkpointed"])
        self.assertFalse(plan["monitoring"]["enabled"])
        self.assertFalse(plan["monitoring"]["preoccupies_gpu_while_waiting"])

    def test_launch_plan_records_safe_wait_without_gpu_preoccupation(self) -> None:
        args = arguments()
        args.wait_for_gpus = True
        args.poll_interval_seconds = 10
        args.monitor_state_path = Path("monitor/llama.json")
        plan = build_launch_plan(args, worker_command=build_worker_command(args))
        self.assertTrue(plan["monitoring"]["enabled"])
        self.assertEqual(plan["monitoring"]["poll_interval_seconds"], 10)
        self.assertEqual(
            plan["monitoring"]["monitor_state_path"], "monitor/llama.json"
        )
        self.assertFalse(plan["monitoring"]["preoccupies_gpu_while_waiting"])


if __name__ == "__main__":
    unittest.main()
