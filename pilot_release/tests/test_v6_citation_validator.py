"""Regression tests for the unified-answer-v6 strict citation contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from citation_validator import validate_v6_strict_citation_contract  # noqa: E402


class V6StrictCitationContractTest(unittest.TestCase):
    def test_accepts_only_fully_cited_sentences_with_final_blocks(self) -> None:
        result = validate_v6_strict_citation_contract(
            "First claim [1]. Second claim [2][3]!",
            source_count=3,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["uncited_sentence_count"], 0)

    def test_rejects_citationless_sentence(self) -> None:
        result = validate_v6_strict_citation_contract(
            "Uncited transition. Supported claim [1].",
            source_count=1,
        )

        self.assertFalse(result["contract_compliant"])
        self.assertEqual(result["uncited_sentence_count"], 1)

    def test_rejects_mid_sentence_token_even_with_valid_final_token(self) -> None:
        result = validate_v6_strict_citation_contract(
            "A claim [1] with more text [2].",
            source_count=2,
        )

        self.assertFalse(result["contract_compliant"])
        self.assertEqual(result["noncompliant_sentence_count"], 1)

    def test_splits_lowercase_sentence_start_without_splitting_titles(self) -> None:
        lowercase = validate_v6_strict_citation_contract(
            "Uncited first. second cited [1].",
            source_count=1,
        )
        title = validate_v6_strict_citation_contract(
            "Dr. Smith reported the result [1].",
            source_count=1,
        )

        self.assertFalse(lowercase["contract_compliant"])
        self.assertEqual(lowercase["sentence_count"], 2)
        self.assertTrue(title["contract_compliant"])
        self.assertEqual(title["sentence_count"], 1)

    def test_does_not_split_at_question_mark_inside_markdown_title(self) -> None:
        result = validate_v6_strict_citation_contract(
            "A relevant title is *Am I The Only Sane One Working Here?*, "
            "which discusses office conflict [1].",
            source_count=1,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["uncited_sentence_count"], 0)

    def test_does_not_split_markdown_title_before_its_subtitle(self) -> None:
        result = validate_v6_strict_citation_contract(
            "A title is *Am I The Only Sane One Working Here?: "
            "101 Solutions for Surviving Office Insanity*, which helps [1].",
            source_count=1,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["uncited_sentence_count"], 0)

    def test_accepts_spaces_inside_the_final_citation_sequence(self) -> None:
        for answer in (
            "Claim [1] .",
            "Claim [1][2]  !",
            "Claim [1] [2].",
            "Claim [1] [2] .",
            "Claim [1]\t[2]\t?",
        ):
            with self.subTest(answer=answer):
                result = validate_v6_strict_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertTrue(result["contract_compliant"])

    def test_rejects_every_noncanonical_or_malformed_bracket_form(self) -> None:
        cases = (
            "Claim [Source 1].",
            "Claim [01].",
            "Claim [1,2].",
            "Claim [1-2].",
            "Claim [1], [2].",
            "Claim [note] [1].",
            "Claim [[1]].",
            "Claim [1]. Extra ] text [1].",
        )
        for answer in cases:
            with self.subTest(answer=answer):
                result = validate_v6_strict_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertFalse(result["contract_compliant"])

    def test_rejects_after_punctuation_out_of_range_and_reference_sections(self) -> None:
        cases = (
            "Claim. [1]",
            "Claim [3].",
            "Claim [1].\nReferences:\n[1] Example",
            "Claim [1].\nSource 1: Example",
        )
        for answer in cases:
            with self.subTest(answer=answer):
                result = validate_v6_strict_citation_contract(
                    answer,
                    source_count=2,
                )
                self.assertFalse(result["contract_compliant"])

    def test_rejects_heading_fragment_and_uncited_list_item(self) -> None:
        result = validate_v6_strict_citation_contract(
            "Overview\n- Supported item [1].\n- Uncited item.",
            source_count=1,
        )

        self.assertFalse(result["contract_compliant"])
        self.assertEqual(result["uncited_sentence_count"], 2)

    def test_rejects_markdown_heading_table_and_citation_only_unit(self) -> None:
        cases = (
            "# Overview [1].",
            "| Claim | Value | [1].",
            "[1].",
            "Sources [1].",
        )
        for answer in cases:
            with self.subTest(answer=answer):
                result = validate_v6_strict_citation_contract(
                    answer,
                    source_count=1,
                )
                self.assertFalse(result["contract_compliant"])

    def test_supports_chinese_sentence_boundaries(self) -> None:
        result = validate_v6_strict_citation_contract(
            "第一句话 [1]。第二句话 [2]！",
            source_count=2,
        )

        self.assertTrue(result["contract_compliant"])
        self.assertEqual(result["sentence_count"], 2)

    def test_never_claims_to_check_semantic_support(self) -> None:
        result = validate_v6_strict_citation_contract(
            "Potentially unsupported claim [1].",
            source_count=1,
        )

        self.assertFalse(result["semantic_support_checked"])
        self.assertTrue(result["raw_answer_preserved"])


if __name__ == "__main__":
    unittest.main()
