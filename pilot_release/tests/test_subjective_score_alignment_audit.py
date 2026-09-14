"""Tests for the post-canonicalization subjective-score alignment audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_subjective_score_alignment import (  # noqa: E402
    ACTION_RERUN,
    ACTION_REUSE,
    ACTION_ZERO,
    DIMENSIONS,
    classify_alignment,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SubjectiveScoreAlignmentAuditTest(unittest.TestCase):
    def test_classification_prioritizes_final_target_absence(self) -> None:
        self.assertEqual(
            classify_alignment(text_changed=False, target_cited_after=True),
            ACTION_REUSE,
        )
        self.assertEqual(
            classify_alignment(text_changed=True, target_cited_after=True),
            ACTION_RERUN,
        )
        self.assertEqual(
            classify_alignment(text_changed=False, target_cited_after=False),
            ACTION_ZERO,
        )
        self.assertEqual(
            classify_alignment(text_changed=True, target_cited_after=False),
            ACTION_ZERO,
        )

    def test_end_to_end_creates_three_disjoint_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.jsonl"
            raw_path = root / "raw.jsonl"
            canonical_path = root / "canonical.jsonl"
            trials_path = root / "trials.jsonl"
            scores_path = root / "scores.jsonl"
            output_dir = root / "output"
            manifest = [
                {
                    "sample_id": f"sample_{index}",
                    "candidate_document_count": 2,
                    "target_document_index": 0,
                }
                for index in range(3)
            ]
            raw_answers = [
                {
                    "model": "model",
                    "sample_id": "sample_0",
                    "prompt_id": "prompt",
                    "seed": 0,
                    "benchmark": "bench",
                    "domain": "domain",
                    "answer_prompt_protocol": "unified_answer_v7",
                    "answer": "Kept target [1].",
                },
                {
                    "model": "model",
                    "sample_id": "sample_1",
                    "prompt_id": "prompt",
                    "seed": 0,
                    "benchmark": "bench",
                    "domain": "domain",
                    "answer_prompt_protocol": "unified_answer_v7",
                    "answer": "Combined target [1,2].",
                },
                {
                    "model": "model",
                    "sample_id": "sample_2",
                    "prompt_id": "prompt",
                    "seed": 0,
                    "benchmark": "bench",
                    "domain": "domain",
                    "answer_prompt_protocol": "unified_answer_v7",
                    "answer": "Trailing target.\n\nReferences: [1]",
                },
            ]

            def canonical(raw: dict, answer: str, changed: bool) -> dict:
                import hashlib

                raw_text = raw["answer"]
                return {
                    **raw,
                    "answer": answer,
                    "raw_answer": raw_text,
                    "source_answer_sha256": hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    "canonicalized_answer_sha256": hashlib.sha256(
                        answer.encode("utf-8")
                    ).hexdigest(),
                    "citation_canonicalization": {"changed": changed},
                }

            canonical_answers = [
                canonical(raw_answers[0], "Kept target [1].", False),
                canonical(raw_answers[1], "Split target [1][2].", True),
                canonical(raw_answers[2], "Trailing target.", True),
            ]
            score_rows = []
            trial_rows = []
            for raw in raw_answers:
                raw_target = raw["sample_id"] != "sample_1"
                score_rows.append(
                    {
                        "model": "model",
                        "sample_id": raw["sample_id"],
                        "prompt_id": "prompt",
                        "answer_seed": 0,
                        "target_source_cited": raw_target,
                        "subjective_average": 3.0 if raw_target else 0.0,
                        "all_dimensions_scorable": True,
                    }
                )
                for dimension in DIMENSIONS:
                    for judge_seed in (0, 1, 2):
                        trial_rows.append(
                            {
                                "model": "model",
                                "sample_id": raw["sample_id"],
                                "prompt_id": "prompt",
                                "answer_seed": 0,
                                "target_document_index": 0,
                                "target_citation_index": 1,
                                "candidate_document_count": 2,
                                "target_source_cited": raw_target,
                                "dimension": dimension,
                                "judge_seed": judge_seed,
                                "score": 3 if raw_target else 0,
                                "score_origin": (
                                    "glm_judge"
                                    if raw_target
                                    else "deterministic_target_absence"
                                ),
                            }
                        )
            trial_rows.sort(
                key=lambda row: (
                    row["model"],
                    row["sample_id"],
                    row["prompt_id"],
                    row["answer_seed"],
                    row["dimension"],
                    row["judge_seed"],
                )
            )
            write_jsonl(manifest_path, manifest)
            write_jsonl(raw_path, raw_answers)
            write_jsonl(canonical_path, canonical_answers)
            write_jsonl(trials_path, trial_rows)
            write_jsonl(scores_path, score_rows)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/audit_subjective_score_alignment.py"),
                    "--experiment-id",
                    "test_alignment",
                    "--raw-answers",
                    str(raw_path),
                    "--canonicalized-answers",
                    str(canonical_path),
                    "--manifest",
                    str(manifest_path),
                    "--old-judge-trials",
                    str(trials_path),
                    "--old-judge-scores",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--expected-answer-count",
                    "3",
                    "--expected-reuse-answer-count",
                    "1",
                    "--expected-rerun-answer-count",
                    "1",
                    "--expected-zero-answer-count",
                    "1",
                    "--expected-rerun-trial-count",
                    "21",
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
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metrics["integrity_passed"])
            self.assertEqual(
                metrics["answer_action_counts"],
                {
                    ACTION_REUSE: 1,
                    ACTION_RERUN: 1,
                    ACTION_ZERO: 1,
                },
            )
            self.assertEqual(metrics["trial_action_counts"][ACTION_RERUN], 21)
            with (output_dir / "rerun_glm_judge_trials.jsonl").open(
                encoding="utf-8"
            ) as handle:
                self.assertEqual(sum(1 for _ in handle), 21)


if __name__ == "__main__":
    unittest.main()
