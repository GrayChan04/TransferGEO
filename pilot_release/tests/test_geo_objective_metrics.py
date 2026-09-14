"""Golden and integration tests for official-compatible GEO metrics."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluate_geo_objective import main as evaluate_main  # noqa: E402
from geo_objective_metrics import (  # noqa: E402
    EXPECTED_NLTK_VERSION,
    EXPECTED_PUNKT_ARCHIVE_SHA256,
    extract_official_citations,
    implementation_provenance,
    paired_improvement,
    parse_answer_official,
    score_answer,
    target_scores,
)


class OfficialCitationParsingTest(unittest.TestCase):
    def test_preserves_official_regex_semantics(self) -> None:
        sentence = "Separate [1][2], decorated [(3)], combined [4,5], named [ref]."
        self.assertEqual(extract_official_citations(sentence), (1, 2, 3))

    def test_counts_only_tokens_longer_than_two_characters(self) -> None:
        parsed = parse_answer_official("An AI is on web [1].")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].eligible_word_count, 1)


class OfficialMetricGoldenTest(unittest.TestCase):
    def test_two_sentence_hand_computed_vectors(self) -> None:
        score = score_answer(
            "Alpha evidence supports claim [1][2]. Beta evidence extends context [2].",
            source_count=2,
        )
        decay = math.exp(-1.0)

        self.assertEqual(score["raw"]["word"], [2.0, 6.0])
        self.assertAlmostEqual(score["normalized"]["word"][0], 0.25)
        self.assertAlmostEqual(score["normalized"]["word"][1], 0.75)

        position_total = 1.0 + decay
        self.assertAlmostEqual(
            score["normalized"]["position"][0], 0.5 / position_total
        )
        self.assertAlmostEqual(
            score["normalized"]["position"][1], (0.5 + decay) / position_total
        )

        overall_total = 4.0 + 4.0 * decay
        self.assertAlmostEqual(
            score["normalized"]["overall"][0], 2.0 / overall_total
        )
        self.assertAlmostEqual(
            score["normalized"]["overall"][1],
            (2.0 + 4.0 * decay) / overall_total,
        )

    def test_official_uniform_and_zero_fallbacks_are_separate(self) -> None:
        official = score_answer("No citation appears.", source_count=4)
        sensitivity = score_answer(
            "No citation appears.", source_count=4, fallback_policy="zero"
        )
        for metric_name in ("word", "position", "overall"):
            self.assertEqual(official["normalized"][metric_name], [0.25] * 4)
            self.assertEqual(sensitivity["normalized"][metric_name], [0.0] * 4)
            self.assertTrue(official["fallback_used"][metric_name])
            self.assertTrue(sensitivity["fallback_used"][metric_name])

    def test_out_of_range_citation_stays_in_divisor_but_gets_no_credit(self) -> None:
        score = score_answer("Alpha evidence supports claim [1][99].", source_count=2)
        self.assertEqual(score["citation_occurrence_count"], 2)
        self.assertEqual(score["in_range_citation_occurrence_count"], 1)
        self.assertEqual(score["out_of_range_citation_occurrence_count"], 1)
        self.assertEqual(score["raw"]["word"], [2.0, 0.0])
        self.assertEqual(score["normalized"]["word"], [1.0, 0.0])

    def test_zero_citation_preserves_official_negative_index_behavior(self) -> None:
        score = score_answer("Alpha evidence supports claim [0].", source_count=3)
        self.assertEqual(score["zero_citation_occurrence_count"], 1)
        self.assertEqual(score["raw"]["word"], [0.0, 0.0, 4.0])
        self.assertEqual(score["normalized"]["word"], [0.0, 0.0, 1.0])

    def test_target_scores_use_zero_based_dynamic_source_index(self) -> None:
        score = score_answer("Alpha evidence supports claim [3].", source_count=3)
        target = target_scores(score, 2)
        self.assertEqual(target["word"], 1.0)
        self.assertEqual(target["word_percent"], 100.0)
        with self.assertRaises(IndexError):
            target_scores(score, 3)

    def test_relative_improvement_has_no_epsilon(self) -> None:
        regular = paired_improvement(0.2, 0.3)
        self.assertAlmostEqual(regular["delta"], 0.1)
        self.assertAlmostEqual(regular["relative_percent"], 50.0)
        zero = paired_improvement(0.0, 0.3)
        self.assertEqual(zero["delta"], 0.3)
        self.assertIsNone(zero["relative_percent"])

    def test_tokenizer_provenance_is_pinned(self) -> None:
        tokenizer = implementation_provenance()["tokenizer"]
        self.assertEqual(tokenizer["nltk_version"], EXPECTED_NLTK_VERSION)
        self.assertEqual(
            tokenizer["punkt_archive_sha256"], EXPECTED_PUNKT_ARCHIVE_SHA256
        )
        self.assertTrue(tokenizer["punkt_archive_hash_matches"])


class ObjectiveEvaluatorIntegrationTest(unittest.TestCase):
    def test_scores_and_pairs_a_synthetic_grid_without_mutating_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.jsonl"
            answers_path = root / "answers.jsonl"
            errors_path = root / "errors.jsonl"
            output_dir = root / "output"
            manifest_row = {
                "sample_id": "sample-1",
                "benchmark": "geo_bench",
                "domain": "all",
                "query": "test query",
                "candidate_document_count": 2,
                "target_document_index": 0,
            }
            answer_rows = [
                {
                    "model": "model-a",
                    "sample_id": "sample-1",
                    "prompt_id": "original.no_rewrite",
                    "seed": 0,
                    "benchmark": "geo_bench",
                    "domain": "all",
                    "query": "test query",
                    "answer_prompt_protocol": "unified_answer_v1",
                    "answer": "Baseline evidence [2].",
                    "cap_hit": False,
                },
                {
                    "model": "model-a",
                    "sample_id": "sample-1",
                    "prompt_id": "evidence_structure_rewriting.conclusion_first",
                    "seed": 0,
                    "benchmark": "geo_bench",
                    "domain": "all",
                    "query": "test query",
                    "answer_prompt_protocol": "unified_answer_v1",
                    "answer": "Target evidence [1].",
                    "cap_hit": False,
                },
            ]
            manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            answers_payload = "".join(json.dumps(row) + "\n" for row in answer_rows)
            answers_path.write_text(answers_payload, encoding="utf-8")
            errors_path.write_text("", encoding="utf-8")

            argv = [
                "evaluate_geo_objective.py",
                "--experiment-id",
                "synthetic-objective-test",
                "--answers",
                str(answers_path),
                "--errors",
                str(errors_path),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--expected-answer-count",
                "2",
                "--expected-error-count",
                "0",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(evaluate_main(), 0)

            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["integrity_passed"])
            self.assertEqual(metrics["objective_score_count"], 2)
            self.assertEqual(
                metrics["next_gate"],
                "audit_full_objective_scores_then_freeze_h1_h2_statistics",
            )
            scored_rows = [
                json.loads(line)
                for line in (output_dir / "objective_scores.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            optimized = next(
                row for row in scored_rows if row["prompt_id"] != "original.no_rewrite"
            )
            self.assertEqual(optimized["official_uniform"]["overall"], 1.0)
            self.assertEqual(
                optimized["paired_improvement"]["official_uniform"]["overall"][
                    "delta"
                ],
                1.0,
            )
            self.assertIsNone(
                optimized["paired_improvement"]["official_uniform"]["overall"][
                    "relative_percent"
                ]
            )
            self.assertEqual(answers_path.read_text(encoding="utf-8"), answers_payload)


if __name__ == "__main__":
    unittest.main()
