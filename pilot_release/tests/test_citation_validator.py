"""Regression tests for the frozen unified-answer-v2 citation syntax gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from citation_validator import validate_v2_citation_contract  # noqa: E402


class V2CitationContractTest(unittest.TestCase):
    def test_accepts_separate_in_range_citations_before_punctuation(self) -> None:
        result = validate_v2_citation_contract(
            "First claim [1]. Second claim [2][3]!",
            source_count=3,
        )
        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["compliant_sentence_count"], 2)

    def test_rejects_citation_after_sentence_punctuation(self) -> None:
        result = validate_v2_citation_contract("A claim. [1]", source_count=1)
        self.assertFalse(result["contract_compliant"])
        self.assertEqual(result["noncompliant_sentence_count"], 1)

    def test_rejects_any_citationless_sentence(self) -> None:
        result = validate_v2_citation_contract(
            "Uncited claim. Supported claim [1].",
            source_count=1,
        )
        self.assertFalse(result["contract_compliant"])
        self.assertEqual(result["noncompliant_sentence_count"], 1)

    def test_rejects_named_combined_ranged_and_out_of_range_forms(self) -> None:
        cases = (
            ("Claim [Source 1].", "named_source_citations"),
            ("Claim (Source 1).", "named_source_citations"),
            ("Claim Source 1.", "named_source_citations"),
            ("Claim [1,2].", "combined_or_ranged_citations"),
            ("Claim [1-2].", "combined_or_ranged_citations"),
            ("Claim [3].", "invalid_numeric_citations"),
        )
        for answer, diagnostic in cases:
            with self.subTest(answer=answer):
                result = validate_v2_citation_contract(answer, source_count=2)
                self.assertFalse(result["contract_compliant"])
                self.assertTrue(result[diagnostic])

    def test_checks_each_nonempty_line_and_chinese_sentence(self) -> None:
        multiline = validate_v2_citation_contract(
            "First claim [1].\nSecond claim without a citation.",
            source_count=1,
        )
        chinese = validate_v2_citation_contract(
            "第一句 [1]。第二句没有引用。",
            source_count=1,
        )
        self.assertFalse(multiline["contract_compliant"])
        self.assertEqual(multiline["sentence_count"], 2)
        self.assertFalse(chinese["contract_compliant"])
        self.assertEqual(chinese["sentence_count"], 2)


if __name__ == "__main__":
    unittest.main()
