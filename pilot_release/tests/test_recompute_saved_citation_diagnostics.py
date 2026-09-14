"""Tests for read-only post-hoc citation rediagnosis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from citation_validator import (  # noqa: E402
    validate_v2_citation_contract,
    validate_v4_optional_citation_contract,
)
from recompute_saved_citation_diagnostics import (  # noqa: E402
    build_summary,
    recompute_rows,
)


class RecomputeSavedCitationDiagnosticsTest(unittest.TestCase):
    def test_rediagnoses_named_parentheses_without_editing_answer(self) -> None:
        answer = "A supported statement (Source 1)."
        row = {
            "experiment_id": "old-smoke",
            "model": "mistral",
            "sample_id": "s",
            "prompt_id": "p",
            "seed": 0,
            "answer": answer,
            "citation_diagnostics": {"source_count": 2},
            "v2_citation_contract": validate_v2_citation_contract(
                answer,
                source_count=2,
            ),
            # Reproduce the old detector's mistaken result.
            "v4_citation_contract": {
                "source_count": 2,
                "contract_compliant": True,
                "citationless_answer": True,
            },
        }

        result = recompute_rows([row])[0]

        self.assertEqual(row["answer"], answer)
        self.assertFalse(result["input_answer_edited"])
        self.assertFalse(
            result["recomputed_v4_citation_contract"]["contract_compliant"]
        )
        self.assertFalse(
            result["recomputed_v4_citation_contract"]["citationless_answer"]
        )
        self.assertEqual(
            result["recomputed_v4_citation_contract"][
                "malformed_citation_sentence_count"
            ],
            1,
        )
        self.assertTrue(result["v4_diagnostic_changed"])

    def test_summary_exposes_before_and_after_counts(self) -> None:
        answer = "A supported statement (Source 1)."
        rows = recompute_rows(
            [
                {
                    "model": "mistral",
                    "sample_id": "s",
                    "prompt_id": "p",
                    "seed": 0,
                    "answer": answer,
                    "citation_diagnostics": {"source_count": 2},
                    "v2_citation_contract": {"source_count": 2},
                    "v4_citation_contract": {
                        "source_count": 2,
                        "contract_compliant": True,
                        "citationless_answer": True,
                    },
                }
            ]
        )

        summary = build_summary(
            rows,
            input_path=Path("answers.jsonl"),
            input_sha256="abc",
        )

        self.assertEqual(summary["answer_count"], 1)
        self.assertEqual(summary["stored_v4_contract_compliant_count"], 1)
        self.assertEqual(summary["recomputed_v4_contract_compliant_count"], 0)
        self.assertEqual(summary["named_source_form_answer_count"], 1)
        self.assertEqual(summary["v4_diagnostic_changed_count"], 1)
        self.assertFalse(summary["input_answers_edited"])
        self.assertFalse(summary["model_generation_rerun"])


if __name__ == "__main__":
    unittest.main()
