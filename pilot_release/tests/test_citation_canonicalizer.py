"""Regression tests for deterministic citation canonicalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from citation_canonicalizer import canonicalize_citations  # noqa: E402


class CitationCanonicalizerTest(unittest.TestCase):
    def test_unheaded_bibliography(self):
        text = 'Body [1].\n\n[1] Dr. Biology. (2014). https://example.org/a\n[2] Author. Journal (2003).'
        self.assertEqual(self.canonicalize(text)['answer'], 'Body [1].')
        self.assert_idempotent(text)

    def test_preserves_numbered_explanations_and_whole_answer(self):
        for text in ('Body.\n\n[1] Explains the first point.\n[2] Explains another point.',
                     '[1] Author (2020).\n[2] Author (2021).'):
            self.assertEqual(self.canonicalize(text)['answer'], text)

    def canonicalize(self, answer: str, source_count: int = 5) -> dict:
        return canonicalize_citations(answer, source_count=source_count)

    def assert_idempotent(self, answer: str, source_count: int = 5) -> None:
        first = self.canonicalize(answer, source_count)
        second = self.canonicalize(first["answer"], source_count)
        self.assertEqual(first["answer"], second["answer"])

    def test_splits_combined_and_range_citations(self) -> None:
        result = self.canonicalize("Claim [1, 2][2-4].")
        self.assertEqual(result["answer"], "Claim [1][2][3][4].")
        self.assertEqual(result["action_counts"]["split_combined_citation"], 1)
        self.assertEqual(result["action_counts"]["expand_ranged_citation"], 1)

    def test_deduplicates_one_citation_cluster(self) -> None:
        result = self.canonicalize("Claim [1] [1][2].")
        self.assertEqual(result["answer"], "Claim [1][2].")
        self.assertGreaterEqual(
            result["action_counts"]["remove_duplicate_label"], 1
        )

    def test_removes_invalid_and_out_of_range_labels(self) -> None:
        result = self.canonicalize("Claim [0][2][7].", source_count=3)
        self.assertEqual(result["answer"], "Claim [2].")
        self.assertEqual(
            result["details"]["invalid_or_out_of_range_numbers"], [0, 7]
        )

    def test_invalid_only_cluster_does_not_leave_space_before_punctuation(self) -> None:
        result = self.canonicalize("Claim [19].", source_count=3)
        self.assertEqual(result["answer"], "Claim.")

    def test_moves_post_punctuation_citations_before_punctuation(self) -> None:
        self.assertEqual(
            self.canonicalize("First claim. [1] Second claim！ [2][3]")["answer"],
            "First claim [1]. Second claim [2][3]！",
        )

    def test_accepts_spaced_labels_and_punctuation(self) -> None:
        result = self.canonicalize("Claim [1] [2] .")
        self.assertEqual(result["answer"], "Claim [1][2].")

    def test_converts_named_source_labels(self) -> None:
        result = self.canonicalize(
            "A [Source 1]. B (Source 2). C Source [3]. D Source 4."
        )
        self.assertEqual(
            result["answer"], "A [1]. B [2]. C Source [3]. D Source [4]."
        )

    def test_converts_parenthesized_named_source_list(self) -> None:
        result = self.canonicalize("Claim. (Source 1, Source 4)")
        self.assertEqual(result["answer"], "Claim [1][4].")

    def test_removes_explicit_trailing_reference_section(self) -> None:
        answer = "Supported claim [1].\n\nReferences:\n[1] Example title"
        result = self.canonicalize(answer)
        self.assertEqual(result["answer"], "Supported claim [1].")
        self.assertEqual(
            result["action_counts"]["remove_trailing_reference_list"], 1
        )

    def test_removes_standalone_citation_only_final_paragraph(self) -> None:
        answer = "Supported claim [1].\n\n[1] [2] [3]."
        result = self.canonicalize(answer)
        self.assertEqual(result["answer"], "Supported claim [1].")
        self.assertEqual(
            result["action_counts"]["remove_standalone_citation_paragraph"], 1
        )

    def test_removes_reference_list_after_named_labels_are_normalized(self) -> None:
        answer = "Supported claim [1].\n\n[1] Source 1\n[2] Source 2"
        result = self.canonicalize(answer)
        self.assertEqual(result["answer"], "Supported claim [1].")
        self.assertEqual(
            result["action_counts"]["remove_standalone_citation_paragraph"], 1
        )

    def test_removes_explicit_unused_sources_note(self) -> None:
        answer = (
            "Supported claim [1].\n\n"
            "[Sources not used in the answer: 2, 3, 4]"
        )
        result = self.canonicalize(answer)
        self.assertEqual(result["answer"], "Supported claim [1].")
        self.assertEqual(
            result["action_counts"]["remove_explicit_unused_sources_note"], 1
        )

    def test_preserves_citation_directly_following_last_sentence(self) -> None:
        result = self.canonicalize("Supported claim. [1]")
        self.assertEqual(result["answer"], "Supported claim [1].")
        self.assertNotIn(
            "remove_trailing_reference_list", result["action_counts"]
        )

    def test_records_ambiguous_format_without_guessing(self) -> None:
        result = self.canonicalize("Claim [1;2].")
        self.assertEqual(result["answer"], "Claim [1;2].")
        self.assertEqual(result["unresolved"][0]["type"], "malformed_numeric_bracket")

    def test_idempotence_for_representative_cases(self) -> None:
        cases = (
            "Claim. [1,2]",
            "Claim [2-4] .",
            "Claim [Source 1].",
            "Claim [1].\n\nSources:\n[1] title",
            "Claim [1].\n\n[1] [2]",
        )
        for answer in cases:
            with self.subTest(answer=answer):
                self.assert_idempotent(answer)


if __name__ == "__main__":
    unittest.main()
