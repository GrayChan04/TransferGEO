"""Unit tests for the unified Answer Prompt tokenizer-length audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_necessity_answer_input_lengths import (  # noqa: E402
    build_metrics,
    build_result_paths,
)


class NecessityAnswerLengthAuditTest(unittest.TestCase):
    def test_semantic_result_prefix_does_not_change_historical_defaults(self) -> None:
        output_dir = Path("results")
        self.assertEqual(
            build_result_paths(output_dir, ""),
            {
                "lengths": output_dir / "lengths.jsonl",
                "overflows": output_dir / "overflows.jsonl",
                "metrics": output_dir / "metrics.json",
            },
        )
        self.assertEqual(
            build_result_paths(output_dir, "answer_prompt_v2__")["metrics"],
            output_dir / "answer_prompt_v2__input_length_audit_metrics.json",
        )

    def test_overflow_is_a_recorded_finding_not_an_integrity_failure(self) -> None:
        rows = []
        for model, tokens in (("qwen", 100), ("llama", 101), ("mistral", 120)):
            rows.append(
                {
                    "model": model,
                    "sample_id": "sample-1",
                    "prompt_id": "prompt-1",
                    "benchmark": "geo_bench",
                    "input_tokens": tokens,
                    "over_budget": tokens > 110,
                }
            )
        metrics = build_metrics(
            experiment_id="test",
            protocol="unified_answer_v1",
            seed=0,
            input_budget=110,
            rows=rows,
            expected_condition_keys={("sample-1", "prompt-1")},
            models=("qwen", "llama", "mistral"),
            diagnostics={
                "manifest_count": 1,
                "rewrite_count": 1,
                "conditions_per_instance": 1,
            },
            model_metadata={},
        )
        self.assertTrue(metrics["complete_accounting"])
        self.assertFalse(metrics["all_conditions_within_common_budget"])
        self.assertEqual(metrics["overflow_record_count"], 1)
        self.assertEqual(metrics["common_budget_overflow_condition_count"], 1)
        self.assertEqual(
            metrics["next_gate"],
            "review_overflow_conditions_before_gpu_smoke",
        )

    def test_missing_model_row_fails_complete_accounting(self) -> None:
        rows = [
            {
                "model": "qwen",
                "sample_id": "sample-1",
                "prompt_id": "prompt-1",
                "benchmark": "geo_bench",
                "input_tokens": 100,
                "over_budget": False,
            }
        ]
        metrics = build_metrics(
            experiment_id="test",
            protocol="unified_answer_v1",
            seed=0,
            input_budget=110,
            rows=rows,
            expected_condition_keys={("sample-1", "prompt-1")},
            models=("qwen", "llama", "mistral"),
            diagnostics={
                "manifest_count": 1,
                "rewrite_count": 1,
                "conditions_per_instance": 1,
            },
            model_metadata={},
        )
        self.assertFalse(metrics["complete_accounting"])
        self.assertEqual(metrics["missing_count"], 2)

    def test_v2_metrics_record_v2_prompt_file_provenance(self) -> None:
        rows = [
            {
                "model": model,
                "sample_id": "sample-1",
                "prompt_id": "prompt-1",
                "benchmark": "geo_bench",
                "input_tokens": 100,
                "over_budget": False,
            }
            for model in ("qwen", "llama", "mistral")
        ]
        metrics = build_metrics(
            experiment_id="test-v2",
            protocol="unified_answer_v2",
            seed=0,
            input_budget=20_480,
            rows=rows,
            expected_condition_keys={("sample-1", "prompt-1")},
            models=("qwen", "llama", "mistral"),
            diagnostics={
                "manifest_count": 1,
                "rewrite_count": 1,
                "conditions_per_instance": 1,
            },
            model_metadata={},
        )
        self.assertEqual(metrics["answer_prompt_protocol"], "unified_answer_v2")
        self.assertIn(
            "prompts/answer_generation/unified_v2/system_prompt.txt",
            metrics["prompt_files"]["system"]["path"],
        )

    def test_v3_metrics_record_v3_prompt_file_provenance(self) -> None:
        rows = [
            {
                "model": model,
                "sample_id": "sample-1",
                "prompt_id": "prompt-1",
                "benchmark": "geo_bench",
                "input_tokens": 100,
                "over_budget": False,
            }
            for model in ("qwen", "llama", "mistral")
        ]
        metrics = build_metrics(
            experiment_id="test-v3",
            protocol="unified_answer_v3",
            seed=0,
            input_budget=20_480,
            rows=rows,
            expected_condition_keys={("sample-1", "prompt-1")},
            models=("qwen", "llama", "mistral"),
            diagnostics={
                "manifest_count": 1,
                "rewrite_count": 1,
                "conditions_per_instance": 1,
            },
            model_metadata={},
        )
        self.assertEqual(metrics["answer_prompt_protocol"], "unified_answer_v3")
        self.assertIn(
            "prompts/answer_generation/unified_v3/system_prompt.txt",
            metrics["prompt_files"]["system"]["path"],
        )


if __name__ == "__main__":
    unittest.main()
