from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_necessity_answer_parallel import require_selected_gpus_safe  # noqa: E402


class ParallelGpuSafetyTest(unittest.TestCase):
    def test_accepts_six_disjoint_idle_gpus(self) -> None:
        snapshot = require_selected_gpus_safe(
            selected=[0, 1, 2, 3, 4, 5],
            memory={0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 7: 700},
            active_compute={7},
            max_preexisting_memory_mib=1024,
        )
        self.assertEqual(snapshot["selected_memory_mib"]["5"], 19)
        self.assertEqual(snapshot["active_compute_gpu_indices"], [7])

    def test_rejects_low_memory_gpu_with_existing_compute_process(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "existing_compute_processes=\[3\]"):
            require_selected_gpus_safe(
                selected=[0, 1, 2, 3, 4, 5],
                memory={0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19},
                active_compute={3},
                max_preexisting_memory_mib=1024,
            )

    def test_rejects_missing_or_over_limit_gpu(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing=\[5\]"):
            require_selected_gpus_safe(
                selected=[0, 1, 2, 3, 4, 5],
                memory={0: 19, 1: 19, 2: 19, 3: 2048, 4: 19},
                active_compute=set(),
                max_preexisting_memory_mib=1024,
            )


if __name__ == "__main__":
    unittest.main()
