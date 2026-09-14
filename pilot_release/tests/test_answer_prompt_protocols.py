"""Regression tests for frozen Answer Model prompt protocols."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from answer_prompt import (  # noqa: E402
    ANSWER_PROMPT_PROTOCOLS,
    BENCHMARK_OFFICIAL_V1,
    UNIFIED_ANSWER_PROMPT_ROOTS,
    UNIFIED_ANSWER_SYSTEM_PROMPT,
    UNIFIED_ANSWER_SYSTEM_PROMPTS,
    UNIFIED_ANSWER_USER_PROMPT,
    UNIFIED_ANSWER_USER_PROMPTS,
    UNIFIED_ANSWER_V1,
    UNIFIED_ANSWER_V2,
    UNIFIED_ANSWER_V3,
    UNIFIED_ANSWER_V4,
    UNIFIED_ANSWER_V5,
    UNIFIED_ANSWER_V6,
    UNIFIED_ANSWER_V7,
    UNIFIED_ANSWER_V8,
    UNIFIED_ANSWER_V9,
    build_answer_messages,
    build_cseo_messages,
    build_geo_messages,
    build_unified_answer_messages,
)


class AnswerPromptProtocolTest(unittest.TestCase):
    def test_protocol_registry_contains_separate_v1_through_v9(self) -> None:
        self.assertEqual(
            ANSWER_PROMPT_PROTOCOLS,
            (
                BENCHMARK_OFFICIAL_V1,
                UNIFIED_ANSWER_V1,
                UNIFIED_ANSWER_V2,
                UNIFIED_ANSWER_V3,
                UNIFIED_ANSWER_V4,
                UNIFIED_ANSWER_V5,
                UNIFIED_ANSWER_V6,
                UNIFIED_ANSWER_V7,
                UNIFIED_ANSWER_V8,
                UNIFIED_ANSWER_V9,
            ),
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V1],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V2],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V2],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V3],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V3],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V4],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V4],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V5],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V5],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V6],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V6],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V7],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V7],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V8],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V8],
            UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V9],
        )

    def test_unified_v1_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "89158ca7b44f02d3d0bf259b551433cd127e5cf016df3cdd513aa78825ce1abe"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V1]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_unified_protocol_uses_system_and_user_roles(self) -> None:
        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], UNIFIED_ANSWER_SYSTEM_PROMPT)
        self.assertIn("Query: What changed?", messages[1]["content"])
        self.assertIn("Source 1:\nFirst source.", messages[1]["content"])
        self.assertIn("Source 2:\nSecond source.", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("### Output\nAnswer:"))

    def test_unified_prompt_files_have_only_expected_runtime_fields(self) -> None:
        self.assertNotIn("{query}", UNIFIED_ANSWER_SYSTEM_PROMPT)
        self.assertNotIn("{sources}", UNIFIED_ANSWER_SYSTEM_PROMPT)
        self.assertEqual(UNIFIED_ANSWER_USER_PROMPT.count("{query}"), 1)
        self.assertEqual(UNIFIED_ANSWER_USER_PROMPT.count("{sources}"), 1)

    def test_v2_changes_only_the_system_prompt_contract(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V2],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V1],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V2],
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V1],
        )
        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V2,
        )
        self.assertEqual(
            messages[0]["content"],
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V2],
        )
        self.assertIn("Source 1:\nFirst source.", messages[1]["content"])

    def test_unified_v2_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "d6e6088f195a347f3eb5d6de25326080ece340316835cdc5d655620141c18061"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V2]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v3_changes_only_the_system_prompt_contract(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V3],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V2],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V3],
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V2],
        )
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V3]
        self.assertIn("non-negotiable output constraint", system_prompt)
        self.assertIn("never place citations only at the end of a paragraph", system_prompt)
        self.assertIn("Do not output a separate references", system_prompt)
        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V3,
        )
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("Source 2:\nSecond source.", messages[1]["content"])

    def test_unified_v3_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "3d9b34ecda4f9041d626fc79350ab54bcbef1cf27e6d2bc8c614186487b84f55"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V3]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v4_changes_only_the_system_prompt_contract(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V4],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V3],
        )
        self.assertNotEqual(
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V4],
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V3],
        )
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V4]
        self.assertIn("Citation use is optional at the sentence level", system_prompt)
        self.assertIn("immediately before the sentence-final punctuation", system_prompt)
        self.assertIn("Never place a citation in the middle", system_prompt)
        self.assertIn("Do not output a separate references", system_prompt)
        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V4,
        )
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("Source 1:\nFirst source.", messages[1]["content"])

    def test_unified_v4_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "2de92162d0a556d4563431cca040b130e51e1b3a54c317110453cc401107b258"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V4]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v5_changes_only_v4s_final_self_review_instruction(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V5],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V4],
        )
        v4_lines = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V4].splitlines()
        v5_lines = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V5].splitlines()
        self.assertEqual(v5_lines[:-1], v4_lines[:-1])
        self.assertNotEqual(v5_lines[-1], v4_lines[-1])
        self.assertIn("repair every citation-placement error", v5_lines[-1])
        self.assertIn("move that citation marker", v5_lines[-1])
        self.assertIn("without changing its source identifier", v5_lines[-1])
        self.assertNotIn("must include at least one citation", v5_lines[-1])

        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V5,
        )
        self.assertEqual(
            messages[0]["content"],
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V5],
        )

    def test_unified_v5_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "55dbd85534d003dc6159cc9c2e74a4edccb15c2580b622b83a342bf19555c509"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V5]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v6_requires_every_sentence_and_repairs_without_false_citations(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V6],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V5],
        )
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V6]
        self.assertIn("every sentence in the final answer must contain", system_prompt)
        self.assertIn("split it into separately supported sentences", system_prompt)
        self.assertIn("Never attach an unsupported citation", system_prompt)
        self.assertIn("Repeat this scan and repair", system_prompt)
        self.assertNotIn("Citation use is optional", system_prompt)

        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V6,
        )
        self.assertEqual(messages[0]["content"], system_prompt)

    def test_unified_v6_user_prompt_matches_frozen_versions(self) -> None:
        self.assertEqual(
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V6],
            UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V1],
        )

    def test_unified_v6_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "252379f954051dee1bc1dd715c962edddcb8ad799a74f22826051dd3b45a461c"
            ),
            "user_prompt.txt": (
                "aaf7521969707d11038aa6841fb51f5aff2dee4d087490f7b6bf6b1c0a3adbf1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V6]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v7_splits_quality_and_tail_citation_responsibilities(self) -> None:
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V7]
        user_prompt = UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V7]

        self.assertIn("Using only the provided web source contents", system_prompt)
        self.assertIn("Use the same language as the query", system_prompt)
        self.assertIn("mandatory citation and output-format contract", system_prompt)
        self.assertNotIn("The only valid citation identifiers", system_prompt)
        self.assertEqual(user_prompt.count("{query}"), 1)
        self.assertEqual(user_prompt.count("{sources}"), 1)
        self.assertEqual(user_prompt.count("{valid_citation_identifiers}"), 1)
        self.assertLess(
            user_prompt.index("{sources}"),
            user_prompt.index("### Mandatory citation and output-format contract"),
        )
        self.assertTrue(user_prompt.endswith("### Output\nAnswer:"))

        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V7,
        )
        rendered_user = messages[1]["content"]
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("Source 1:\nFirst source.", rendered_user)
        self.assertIn("Source 2:\nSecond source.", rendered_user)
        self.assertIn("are:\n[1], [2]", rendered_user)
        self.assertLess(
            rendered_user.index("Source 2:\nSecond source."),
            rendered_user.index("### Mandatory citation and output-format contract"),
        )
        self.assertNotIn("{valid_citation_identifiers}", rendered_user)
        self.assertNotIn("target document", rendered_user.lower())

    def test_unified_v7_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "8877f9bc92c08d6199d868542e964c26e91264bb9297b09378d2e1eba0a8251a"
            ),
            "user_prompt.txt": (
                "57a4f5c7143c1c592acb4ad04872cb497ab5e7d3387c755b3bde04bd16319338"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V7]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v8_allows_optional_synthesis_and_joint_support(self) -> None:
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V8]
        user_prompt = UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V8]

        self.assertEqual(
            system_prompt,
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V7],
        )
        self.assertIn("Natural paragraph breaks are allowed", user_prompt)
        self.assertIn(
            "Every factual or explanatory sentence that draws on the Source Contents",
            user_prompt,
        )
        self.assertIn("jointly supported by multiple sources", user_prompt)
        self.assertIn(
            "A high-level overall conclusion or synthesis may be left uncited",
            user_prompt,
        )
        self.assertIn("Do not attach a citation", user_prompt)
        self.assertIn("Do not output headings, lists", user_prompt)
        self.assertNotIn("Return only one concise answer paragraph", user_prompt)
        self.assertNotIn("Do not output any uncited sentence", user_prompt)
        self.assertNotIn("Prefer one supporting source per sentence", user_prompt)

        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V8,
        )
        rendered_user = messages[1]["content"]
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("Source 1:\nFirst source.", rendered_user)
        self.assertIn("Source 2:\nSecond source.", rendered_user)
        self.assertIn("are:\n[1], [2]", rendered_user)
        self.assertNotIn("{valid_citation_identifiers}", rendered_user)
        self.assertNotIn("target document", rendered_user.lower())

    def test_unified_v8_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "8877f9bc92c08d6199d868542e964c26e91264bb9297b09378d2e1eba0a8251a"
            ),
            "user_prompt.txt": (
                "57adcc2a2de8f52ba17e1e60be577d4bc2a30e8562eb5d2b5fc244a1b77378bc"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V8]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_v9_combines_v8_semantics_with_mechanical_closure(self) -> None:
        system_prompt = UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V9]
        user_prompt = UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V9]

        self.assertEqual(
            system_prompt,
            UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V8],
        )
        self.assertIn("Return only a concise answer", user_prompt)
        self.assertIn("Natural paragraph breaks are allowed", user_prompt)
        self.assertIn(
            "Every factual or explanatory sentence that draws on the Source Contents",
            user_prompt,
        )
        self.assertIn("multiple sources jointly support", user_prompt)
        self.assertIn(
            "A high-level overall conclusion or synthesis may be left uncited",
            user_prompt,
        )
        self.assertIn("including `(Source 2)` or `[Source 2]`", user_prompt)
        self.assertIn("A citation after the sentence-final punctuation is also invalid", user_prompt)
        self.assertIn("silently scan the completed answer sentence by sentence", user_prompt)
        self.assertNotIn("Do not output any uncited sentence", user_prompt)
        self.assertNotIn("Prefer one supporting source per sentence", user_prompt)

        messages = build_unified_answer_messages(
            "What changed?",
            ["First source.", "Second source."],
            protocol=UNIFIED_ANSWER_V9,
        )
        rendered_user = messages[1]["content"]
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("Source 1:\nFirst source.", rendered_user)
        self.assertIn("Source 2:\nSecond source.", rendered_user)
        self.assertIn("are:\n[1], [2]", rendered_user)
        self.assertNotIn("{valid_citation_identifiers}", rendered_user)

    def test_unified_v9_files_remain_byte_for_byte_frozen(self) -> None:
        expected_hashes = {
            "system_prompt.txt": (
                "8877f9bc92c08d6199d868542e964c26e91264bb9297b09378d2e1eba0a8251a"
            ),
            "user_prompt.txt": (
                "2089763aa12e4ac272414415dbb269dfeaa144469ff677b9fc7b5912eb650ae1"
            ),
        }
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V9]
        observed = {
            name: hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }
        self.assertEqual(observed, expected_hashes)

    def test_unified_protocol_is_identical_across_benchmarks(self) -> None:
        for protocol in (
            UNIFIED_ANSWER_V1,
            UNIFIED_ANSWER_V2,
            UNIFIED_ANSWER_V3,
            UNIFIED_ANSWER_V4,
            UNIFIED_ANSWER_V5,
            UNIFIED_ANSWER_V6,
            UNIFIED_ANSWER_V7,
            UNIFIED_ANSWER_V8,
            UNIFIED_ANSWER_V9,
        ):
            common = {
                "protocol": protocol,
                "query": "Test query",
                "sources": ["A", "B"],
            }
            geo = build_answer_messages(
                benchmark="geo_bench",
                domain="all",
                **common,
            )
            cseo = build_answer_messages(
                benchmark="cseo_bench",
                domain="retail",
                **common,
            )
            self.assertEqual(geo, cseo)
            self.assertNotIn("product recommender", str(cseo).lower())

    def test_official_v1_builders_remain_unchanged_and_separate(self) -> None:
        geo = build_geo_messages("Q", ["A"])
        cseo = build_cseo_messages("retail", "Q", ["A"])
        self.assertEqual([message["role"] for message in geo], ["user"])
        self.assertEqual([message["role"] for message in cseo], ["system", "user"])
        dispatched_geo = build_answer_messages(
            protocol=BENCHMARK_OFFICIAL_V1,
            benchmark="geo_bench",
            domain="all",
            query="Q",
            sources=["A"],
        )
        dispatched_cseo = build_answer_messages(
            protocol=BENCHMARK_OFFICIAL_V1,
            benchmark="cseo_bench",
            domain="retail",
            query="Q",
            sources=["A"],
        )
        self.assertEqual(dispatched_geo, geo)
        self.assertEqual(dispatched_cseo, cseo)


if __name__ == "__main__":
    unittest.main()
