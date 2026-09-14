"""End-to-end test for canonicalized RAID Judge score merging."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from merge_canonicalized_raid_judge_scores import (  # noqa: E402
    ACTION_RERUN,
    ACTION_REUSE,
    ACTION_ZERO,
    DIMENSIONS,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class MergeCanonicalizedRaidJudgeScoresTest(unittest.TestCase):
    def test_three_sources_form_one_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.jsonl"
            answers_path = root / "answers.jsonl"
            alignment_path = root / "alignment.jsonl"
            old_trials_path = root / "old_trials.jsonl"
            rerun_root = root / "rerun_shards"
            output_dir = root / "output"
            actions = (ACTION_REUSE, ACTION_RERUN, ACTION_ZERO)
            manifest_rows = []
            answer_rows = []
            alignment_rows = []
            old_trials = []
            rerun_trials = []
            for index, action in enumerate(actions):
                sample_id = f"sample_{index}"
                target_cited = action != ACTION_ZERO
                answer = "Target is visible [1]." if target_cited else "Other [2]."
                answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_document_count": 2,
                        "target_document_index": 0,
                    }
                )
                answer_rows.append(
                    {
                        "model": "model",
                        "sample_id": sample_id,
                        "prompt_id": "prompt",
                        "seed": 0,
                        "benchmark": "bench",
                        "domain": "domain",
                        "query": "query",
                        "answer": answer,
                    }
                )
                alignment_rows.append(
                    {
                        "model": "model",
                        "sample_id": sample_id,
                        "prompt_id": "prompt",
                        "answer_seed": 0,
                        "canonical_target_source_cited": target_cited,
                        "subjective_score_action": action,
                        "canonicalized_answer_sha256": answer_hash,
                    }
                )
                for dimension in DIMENSIONS:
                    for judge_seed in (0, 1, 2):
                        base = {
                            "model": "model",
                            "sample_id": sample_id,
                            "prompt_id": "prompt",
                            "answer_seed": 0,
                            "benchmark": "bench",
                            "domain": "domain",
                            "target_document_index": 0,
                            "target_citation_index": 1,
                            "candidate_document_count": 2,
                            "target_source_cited": True,
                            "dimension": dimension,
                            "judge_seed": judge_seed,
                            "raw_output": "3",
                            "answer": "3",
                            "score": 3,
                            "score_parse_method": "exact_bare",
                            "score_origin": "glm_judge",
                            "input_tokens": 10,
                            "output_tokens": 1,
                        }
                        if action == ACTION_REUSE:
                            old_trials.append(base)
                        elif action == ACTION_RERUN:
                            rerun_trials.append(base)
            old_trials.sort(
                key=lambda row: (
                    row["model"],
                    row["sample_id"],
                    row["prompt_id"],
                    row["answer_seed"],
                    row["dimension"],
                    row["judge_seed"],
                )
            )
            rerun_trials.sort(
                key=lambda row: (
                    row["model"],
                    row["sample_id"],
                    row["prompt_id"],
                    row["answer_seed"],
                    row["dimension"],
                    row["judge_seed"],
                )
            )
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(answers_path, answer_rows)
            write_jsonl(alignment_path, alignment_rows)
            write_jsonl(old_trials_path, old_trials)
            write_jsonl(rerun_root / "shard_0/trials.jsonl", rerun_trials)
            write_jsonl(rerun_root / "shard_0/errors.jsonl", [])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/merge_canonicalized_raid_judge_scores.py"),
                    "--experiment-id",
                    "merge_test",
                    "--canonicalized-answers",
                    str(answers_path),
                    "--manifest",
                    str(manifest_path),
                    "--alignment-manifest",
                    str(alignment_path),
                    "--old-judge-trials",
                    str(old_trials_path),
                    "--rerun-shards-root",
                    str(rerun_root),
                    "--num-rerun-shards",
                    "1",
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "0",
                    "1",
                    "2",
                    "--expected-answer-count",
                    "3",
                    "--expected-trial-count",
                    "63",
                    "--expected-reuse-trial-count",
                    "21",
                    "--expected-rerun-trial-count",
                    "21",
                    "--expected-zero-trial-count",
                    "21",
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
            self.assertTrue(metrics["integrity_passed"])
            self.assertEqual(metrics["answer_count"], 3)
            self.assertEqual(metrics["completed_trial_count"], 63)
            self.assertEqual(
                metrics["trial_action_counts"],
                {
                    ACTION_REUSE: 21,
                    ACTION_RERUN: 21,
                    ACTION_ZERO: 21,
                },
            )
            with (output_dir / "judge_scores.jsonl").open(encoding="utf-8") as handle:
                scores = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(scores), 3)
            by_sample = {row["sample_id"]: row for row in scores}
            self.assertEqual(by_sample["sample_0"]["subjective_average"], 3.0)
            self.assertEqual(by_sample["sample_1"]["subjective_average"], 3.0)
            self.assertEqual(by_sample["sample_2"]["subjective_average"], 0.0)


if __name__ == "__main__":
    unittest.main()
