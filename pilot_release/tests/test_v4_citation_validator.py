"""Regression tests for the unified-answer-v4 optional citation contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from citation_validator import (  # noqa: E402
    validate_v4_optional_citation_contract,
)


class V4OptionalCitationContractTest(unittest.TestCase):
    def test_accepts_a_completely_uncited_nonempty_answer_but_flags_it(self) -> None:
        result = validate_v4_optional_citation_contract(
            "An uncited opening. Another uncited sentence.",
            source_count=2,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertTrue(result["citationless_answer"])
        self.assertEqual(result["citation_bearing_sentence_count"], 0)
        self.assertEqual(result["uncited_sentence_count"], 2)

    def test_accepts_mixed_uncited_and_well_formed_cited_sentences(self) -> None:
        result = validate_v4_optional_citation_contract(
            "An uncited transition. A supported claim [1]. "
            "A multi-source claim [1][2]!",
            source_count=2,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertFalse(result["citationless_answer"])
        self.assertEqual(result["citation_bearing_sentence_count"], 2)
        self.assertEqual(result["compliant_citation_bearing_sentence_count"], 2)
        self.assertEqual(result["uncited_sentence_count"], 1)

    def test_rejects_mid_sentence_and_after_punctuation_citations(self) -> None:
        cases = (
            "A [1] supported claim.",
            "A claim [1] with more text [2].",
            "A supported claim. [1]",
        )
        for answer in cases:
            with self.subTest(answer=answer):
                result = validate_v4_optional_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertFalse(result["contract_compliant"])
                self.assertEqual(result["malformed_citation_sentence_count"], 1)

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
                result = validate_v4_optional_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertFalse(result["contract_compliant"])
                self.assertTrue(result[diagnostic])

    def test_named_source_forms_are_not_misclassified_as_citationless(self) -> None:
        for answer in ("Claim [Source 1].", "Claim (Source 1).", "Claim Source 1."):
            with self.subTest(answer=answer):
                result = validate_v4_optional_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertFalse(result["citationless_answer"])
                self.assertEqual(result["citation_bearing_sentence_count"], 1)
                self.assertEqual(result["malformed_citation_sentence_count"], 1)

    def test_rejects_reference_sections_and_restated_source_mappings(self) -> None:
        references = validate_v4_optional_citation_contract(
            "A claim [1].\n\nReferences:\n[1] Example",
            source_count=1,
        )
        mapping = validate_v4_optional_citation_contract(
            "A claim [1].\nSource 1: Example",
            source_count=1,
        )

        self.assertFalse(references["contract_compliant"])
        self.assertTrue(references["separate_reference_section_detected"])
        self.assertFalse(mapping["contract_compliant"])
        self.assertTrue(mapping["restated_source_mapping_detected"])

    def test_applies_the_same_optional_rule_to_chinese_sentences(self) -> None:
        result = validate_v4_optional_citation_contract(
            "这是没有引用的过渡句。这是有引用的句子 [1]。",
            source_count=1,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["uncited_sentence_count"], 1)
        self.assertEqual(result["citation_bearing_sentence_count"], 1)

    def test_rejects_an_empty_answer(self) -> None:
        result = validate_v4_optional_citation_contract("", source_count=1)

        self.assertFalse(result["contract_compliant"])
        self.assertFalse(result["citationless_answer"])


if __name__ == "__main__":
    unittest.main()
