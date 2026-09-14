from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from aggregate_necessity_answer_shards import (  # noqa: E402
    summarize_citation_diagnostics,
    summarize_error_types,
)
from run_necessity_answer_model_shard import (  # noqa: E402
    build_citation_diagnostics,
)


class AnswerCitationDiagnosticsTest(unittest.TestCase):
    def test_full_answer_row_diagnostics_include_all_versioned_views(self) -> None:
        diagnostics = build_citation_diagnostics(
            "First supported sentence [1]. Second supported sentence [2].",
            source_count=2,
        )

        self.assertEqual(
            diagnostics["citation_diagnostics"]["numeric_citations"],
            [1, 2],
        )
        self.assertTrue(
            diagnostics["v2_citation_contract"]["contract_compliant"]
        )
        self.assertTrue(
            diagnostics["v4_citation_contract"]["contract_compliant"]
        )
        self.assertTrue(
            diagnostics["v6_citation_contract"]["contract_compliant"]
        )

    def test_diagnostics_record_format_and_range_failures_without_repair(self) -> None:
        answer = "Uncited opening. Supported claim [3]."
        diagnostics = build_citation_diagnostics(answer, source_count=2)

        self.assertEqual(
            diagnostics["citation_diagnostics"]["invalid_numeric_citations"],
            [3],
        )
        self.assertFalse(
            diagnostics["v2_citation_contract"]["contract_compliant"]
        )
        self.assertTrue(diagnostics["citation_diagnostics"]["raw_answer_preserved"])
        self.assertTrue(diagnostics["v2_citation_contract"]["raw_answer_preserved"])
        self.assertTrue(diagnostics["v4_citation_contract"]["raw_answer_preserved"])
        self.assertTrue(diagnostics["v6_citation_contract"]["raw_answer_preserved"])

    def test_aggregate_summary_counts_diagnostics_but_does_not_gate_content(self) -> None:
        rows = [
            build_citation_diagnostics("Supported [1].", source_count=2),
            build_citation_diagnostics("Uncited. Claim [3].", source_count=2),
        ]
        summary = summarize_citation_diagnostics(rows)

        self.assertEqual(summary["answer_count"], 2)
        self.assertEqual(summary["stored_strict_diagnostic_count"], 2)
        self.assertEqual(summary["stored_optional_citation_diagnostic_count"], 2)
        self.assertEqual(summary["stored_v6_strict_citation_diagnostic_count"], 2)
        self.assertEqual(summary["strict_citation_contract_compliant_count"], 1)
        self.assertEqual(summary["strict_citation_contract_failure_count"], 1)
        self.assertEqual(summary["optional_citation_contract_compliant_count"], 1)
        self.assertEqual(summary["optional_citation_contract_failure_count"], 1)
        self.assertEqual(summary["invalid_numeric_citation_answer_count"], 1)
        self.assertEqual(summary["v6_strict_citation_contract_compliant_count"], 1)

    def test_only_predeclared_input_overflow_is_an_allowed_exclusion(self) -> None:
        counts, unexpected = summarize_error_types(
            [
                {"error_type": "input_budget_overflow"},
                {"error_type": "input_budget_overflow"},
            ],
            allowed_error_types=["input_budget_overflow"],
        )
        self.assertEqual(counts, {"input_budget_overflow": 2})
        self.assertEqual(unexpected, [])

        _, unexpected = summarize_error_types(
            [{"error_type": "RuntimeError"}],
            allowed_error_types=["input_budget_overflow"],
        )
        self.assertEqual(unexpected, ["RuntimeError"])


if __name__ == "__main__":
    unittest.main()
