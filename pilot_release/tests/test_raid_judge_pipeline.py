"""Regression tests for RAID Judge selection and source retargeting."""

from __future__ import annotations

import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_raid_judge_shard import (  # noqa: E402
    DIMENSIONS,
    answer_key,
    load_target_map,
    parse_score,
    parse_score_with_method,
    read_jsonl,
    render_targeted_prompt,
    resolve_answer_query,
    select_shard_answers,
    shard_for,
    target_source_is_cited,
)
from run_raid_judge_parallel import unavailable_gpus  # noqa: E402
from run_raid_judge_shard_on_available_gpus import select_safe_gpus  # noqa: E402


class RaidJudgePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.answers = read_jsonl(
            ROOT
            / "results/processed/necessity_pilot/unified_answer_prompt_v1"
            / "03_answer_generation_full"
            / "answer_prompt_v1__full_generated_answers.jsonl"
        )
        cls.targets = load_target_map(
            ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl"
        )

    def test_smoke_selects_exactly_one_unique_answer_per_shard(self) -> None:
        selected = []
        for shard_index in range(4):
            shard_answers = select_shard_answers(
                self.answers,
                shard_index=shard_index,
                num_shards=4,
                limit_answers=1,
            )
            self.assertEqual(len(shard_answers), 1)
            self.assertEqual(shard_for(shard_answers[0], 4), shard_index)
            selected.extend(shard_answers)
        self.assertEqual(len({answer_key(row) for row in selected}), 4)

    def test_gpu_guard_treats_missing_or_over_threshold_as_unavailable(self) -> None:
        memory = {0: 19, 1: 1024, 2: 1025}
        self.assertEqual(
            unavailable_gpus(memory, [0, 1, 2, 3], 1024),
            {2: 1025, 3: None},
        )

    def test_dynamic_shard_selects_only_idle_unreserved_gpus(self) -> None:
        snapshot = {
            "gpus": {
                0: {"memory_used_mib": 20, "compute_processes": []},
                1: {"memory_used_mib": 700, "compute_processes": [{"pid": 1}]},
                2: {"memory_used_mib": 20, "compute_processes": []},
                3: {"memory_used_mib": 2048, "compute_processes": []},
                4: {"memory_used_mib": 20, "compute_processes": []},
            }
        }
        self.assertEqual(
            select_safe_gpus(
                snapshot,
                candidate_gpus=[0, 1, 2, 3, 4],
                reserved_gpus={2},
                gpus_per_shard=2,
                max_preexisting_memory_mib=1024,
            ),
            [0, 4],
        )
        self.assertIsNone(
            select_safe_gpus(
                snapshot,
                candidate_gpus=[0, 1, 2, 3, 4],
                reserved_gpus={0, 2},
                gpus_per_shard=2,
                max_preexisting_memory_mib=1024,
            )
        )

    def test_smoke_exercises_nonfirst_target_source(self) -> None:
        selected = [
            select_shard_answers(
                self.answers,
                shard_index=shard_index,
                num_shards=4,
                limit_answers=1,
            )[0]
            for shard_index in range(4)
        ]
        citation_indices = {
            self.targets[str(row["sample_id"])]["target_citation_index"]
            for row in selected
        }
        self.assertTrue(any(index != 1 for index in citation_indices))

    def test_manifest_exposes_nonempty_query(self) -> None:
        for sample_id, metadata in self.targets.items():
            self.assertIsInstance(metadata["query"], str, sample_id)
            self.assertTrue(metadata["query"].strip(), sample_id)

    def test_missing_answer_query_falls_back_to_manifest(self) -> None:
        smoke_answers = read_jsonl(
            ROOT
            / "results/processed/necessity_pilot/unified_answer_prompt_v7"
            / "02_answer_generation_smoke"
            / "answer_prompt_v7__smoke_generated_answers.jsonl"
        )
        answer = smoke_answers[0]
        self.assertNotIn("query", answer)
        expected = self.targets[str(answer["sample_id"])]["query"]
        query, source = resolve_answer_query(
            answer, self.targets[str(answer["sample_id"])]
        )
        self.assertEqual(query, expected)
        self.assertEqual(source, "manifest_fallback")

    def test_matching_answer_query_is_verified(self) -> None:
        answer = dict(self.answers[0])
        metadata = self.targets[str(answer["sample_id"])]
        self.assertEqual(answer["query"], metadata["query"])
        query, source = resolve_answer_query(answer, metadata)
        self.assertEqual(query, metadata["query"])
        self.assertEqual(source, "answer_verified")

    def test_mismatched_answer_query_is_rejected(self) -> None:
        answer = dict(self.answers[0])
        metadata = self.targets[str(answer["sample_id"])]
        answer["query"] = metadata["query"] + " changed"
        with self.assertRaisesRegex(ValueError, "Query mismatch"):
            resolve_answer_query(answer, metadata)

    def test_score_parser_extracts_concrete_scores_from_extra_text(self) -> None:
        self.assertEqual(parse_score("[0]"), 0)
        self.assertEqual(parse_score(" [5] "), 5)
        self.assertEqual(parse_score("3"), 3)
        self.assertEqual(parse_score(" 4 "), 4)
        self.assertEqual(parse_score("score: [3]"), 3)
        self.assertEqual(parse_score("Final score is 3."), 3)
        self.assertEqual(parse_score("评分：4"), 4)
        self.assertEqual(parse_score("3/5"), 3)
        self.assertEqual(parse_score("I would rate it a 2."), 2)
        self.assertEqual(parse_score("Source [1] is absent; score: 0."), 0)
        self.assertIsNone(parse_score("[6]"))
        self.assertIsNone(parse_score("The score could be 2 or 3."))
        self.assertIsNone(parse_score("score: 2 or 3"))
        self.assertEqual(
            parse_score("I considered 2 or 3. Final score: 3"), 3
        )
        self.assertIsNone(parse_score("No numeric rating was given."))

    def test_score_parser_records_the_extraction_method(self) -> None:
        self.assertEqual(parse_score_with_method("3"), (3, "exact_bare"))
        self.assertEqual(
            parse_score_with_method("[3]"), (3, "exact_bracketed")
        )
        self.assertEqual(
            parse_score_with_method("Final score: 3"), (3, "final_labeled")
        )

    def test_all_prompts_retarget_every_source_mention(self) -> None:
        prompt_root = ROOT / "prompts/evaluation/raid"
        for dimension in DIMENSIONS:
            template = (prompt_root / f"{dimension}.txt").read_text(
                encoding="utf-8"
            )
            rendered = render_targeted_prompt(
                template,
                query="test query",
                answer="test answer [5]",
                target_citation_index=5,
            )
            self.assertNotIn("Source [1]", rendered)
            self.assertIn("Source [5]", rendered)
            self.assertIn("test query", rendered)
            self.assertIn("test answer [5]", rendered)

    def test_target_presence_uses_exact_separate_square_bracket_index(self) -> None:
        metadata = {"target_citation_index": 3}
        self.assertTrue(
            target_source_is_cited({"answer": "Supported claim [3]."}, metadata)
        )
        self.assertTrue(
            target_source_is_cited(
                {"answer": "Supported by several sources [1][3]."}, metadata
            )
        )
        self.assertFalse(
            target_source_is_cited({"answer": "Other source [1][2]."}, metadata)
        )
        self.assertFalse(
            target_source_is_cited({"answer": "Combined form [3,4]."}, metadata)
        )
        self.assertFalse(
            target_source_is_cited({"answer": "Decorated form [(3)]."}, metadata)
        )

    def test_absent_target_is_scored_zero_without_loading_glm(self) -> None:
        smoke_answers = read_jsonl(
            ROOT
            / "results/processed/necessity_pilot/unified_answer_prompt_v7"
            / "02_answer_generation_smoke"
            / "answer_prompt_v7__smoke_generated_answers.jsonl"
        )
        answer = next(
            row
            for row in smoke_answers
            if not target_source_is_cited(
                row, self.targets[str(row["sample_id"])]
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            input_path = temporary_root / "answers.jsonl"
            input_path.write_text(
                json.dumps(answer, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output_dir = temporary_root / "judge"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/run_raid_judge_shard.py"),
                    "--experiment-id",
                    "absence_zero_test",
                    "--input",
                    str(input_path),
                    "--manifest",
                    str(
                        ROOT
                        / "data/processed/necessity_pilot_v1/sample_manifest.jsonl"
                    ),
                    "--output-dir",
                    str(output_dir),
                    "--shard-index",
                    "0",
                    "--num-shards",
                    "1",
                    "--seeds",
                    "0",
                    "1",
                    "2",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            trials = read_jsonl(output_dir / "trials.jsonl")
            self.assertEqual(len(trials), 21)
            self.assertTrue(all(row["score"] == 0 for row in trials))
            self.assertTrue(
                all(
                    row["score_origin"] == "deterministic_target_absence"
                    for row in trials
                )
            )
            self.assertTrue(all(row["raw_output"] is None for row in trials))
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["glm_judge_trial_count"], 0)
            self.assertEqual(metrics["deterministic_zero_trial_count"], 21)
            self.assertTrue(metrics["complete_accounting"])

    def test_diagnostic_aggregate_expects_only_selected_answers(self) -> None:
        selected_by_shard = {
            shard_index: select_shard_answers(
                self.answers,
                shard_index=shard_index,
                num_shards=4,
                limit_answers=1,
            )[0]
            for shard_index in range(4)
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            input_path = temporary_root / "answers.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in selected_by_shard.values()
                ),
                encoding="utf-8",
            )
            shards_root = temporary_root / "shards"
            for shard_index, answer in selected_by_shard.items():
                shard_dir = shards_root / f"shard_{shard_index}"
                shard_dir.mkdir(parents=True)
                target = self.targets[str(answer["sample_id"])]
                rows = []
                for dimension in DIMENSIONS:
                    for judge_seed in (0, 1, 2):
                        rows.append(
                            {
                                "model": answer["model"],
                                "sample_id": answer["sample_id"],
                                "prompt_id": answer["prompt_id"],
                                "answer_seed": int(answer["seed"]),
                                "benchmark": answer["benchmark"],
                                "domain": answer["domain"],
                                **target,
                                "dimension": dimension,
                                "judge_seed": judge_seed,
                                "raw_output": "[3]",
                                "answer": "[3]",
                                "score": 3,
                                "input_tokens": 1,
                                "output_tokens": 1,
                            }
                        )
                (shard_dir / "trials.jsonl").write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )

            output_dir = temporary_root / "aggregate"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/aggregate_raid_judge_shards.py"),
                    "--experiment-id",
                    "diagnostic_test",
                    "--input",
                    str(input_path),
                    "--manifest",
                    str(
                        ROOT
                        / "data/processed/necessity_pilot_v1/sample_manifest.jsonl"
                    ),
                    "--shards-root",
                    str(shards_root),
                    "--output-dir",
                    str(output_dir),
                    "--num-shards",
                    "4",
                    "--seeds",
                    "0",
                    "1",
                    "2",
                    "--limit-answers-per-shard",
                    "1",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["experiment_id"], "diagnostic_test")
            self.assertEqual(metrics["input_answer_count"], 4)
            self.assertEqual(metrics["answer_count"], 4)
            self.assertEqual(metrics["expected_trials"], 84)
            self.assertEqual(metrics["completed_trials"], 84)
            self.assertEqual(metrics["selected_answer_counts_per_shard"], [1, 1, 1, 1])
            self.assertTrue(metrics["complete_accounting"])


if __name__ == "__main__":
    unittest.main()
