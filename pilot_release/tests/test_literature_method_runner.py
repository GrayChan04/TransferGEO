from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from literature_method_runner import (  # noqa: E402
    BackendGeneration,
    DeterministicDryRunBackend,
    LiteratureMethodCatalog,
    LiteratureMethodExecutor,
    StageBudgets,
    parse_json_strict_or_single_json_fence,
    render_template,
)
from run_literature_method_rewrite_stage import (  # noqa: E402
    OPTIMIZER_MANIFEST_FIELDS,
    build_condition_error_payload,
    select_manifest_rows,
)
from audit_literature_method_rewrite_lengths import stage_plans  # noqa: E402
from diagnose_if_geo_query_mining import analyze_query_mining_output  # noqa: E402


CATALOG_ROOT = ROOT / "prompts/necessity_pilot/literature_methods_v1"


class ScriptedBackend:
    def __init__(self, scripted: dict[str, str]) -> None:
        self.scripted = scripted
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        stage_id: str,
        max_new_tokens: int,
        seed: int,
    ) -> BackendGeneration:
        self.calls.append((stage_id, list(messages)))
        answer = self.scripted[stage_id]
        return BackendGeneration(
            answer=answer,
            raw_answer=answer,
            input_tokens=10,
            output_tokens=5,
            model_id="test-backend",
        )

    def count_text_tokens(self, text: str) -> int:
        return 100


class FencedQueryDryRunBackend(DeterministicDryRunBackend):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        stage_id: str,
        max_new_tokens: int,
        seed: int,
    ) -> BackendGeneration:
        generation = super().generate(
            messages,
            stage_id=stage_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if stage_id != "if_geo.query_mining":
            return generation
        fenced = "```json\n" + generation.answer + "\n```"
        return BackendGeneration(
            answer=fenced,
            raw_answer=fenced,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            model_id=generation.model_id,
        )


def executor(backend: object) -> LiteratureMethodExecutor:
    return LiteratureMethodExecutor(
        catalog=LiteratureMethodCatalog(CATALOG_ROOT),
        backend=backend,
        budgets=StageBudgets(
            intermediate_max_new_tokens=128,
            rewrite_growth_ratio=1.25,
            rewrite_growth_slack_tokens=256,
        ),
    )


class LiteratureMethodCatalogTest(unittest.TestCase):
    def test_catalog_and_every_prompt_hash_validate(self) -> None:
        catalog = LiteratureMethodCatalog(CATALOG_ROOT)
        self.assertEqual(len(catalog.method_ids), 7)
        self.assertFalse(catalog.payload["query_visible_to_optimizer"])

    def test_runtime_renderer_rejects_unresolved_query_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unresolved runtime placeholders"):
            render_template("Document: {{DOCUMENT}} Query: {{QUERY}}", DOCUMENT="x")

    def test_optimizer_manifest_allowlist_excludes_real_query(self) -> None:
        self.assertNotIn("query", OPTIMIZER_MANIFEST_FIELDS)

    def test_smoke_sample_selection_preserves_requested_order(self) -> None:
        manifest = [{"sample_id": "a"}, {"sample_id": "b"}]
        selected = select_manifest_rows(manifest, ["b", "a"])
        self.assertEqual([row["sample_id"] for row in selected], ["b", "a"])

    def test_smoke_sample_selection_rejects_unknown_id(self) -> None:
        with self.assertRaisesRegex(KeyError, "absent from manifest"):
            select_manifest_rows([{"sample_id": "a"}], ["missing"])


class FixedAndRaidMethodTest(unittest.TestCase):
    def test_geo_statistics_extracts_the_final_fenced_document(self) -> None:
        backend = ScriptedBackend(
            {
                "geo.statistics_addition.rewrite": (
                    "1. Add a statistic\nUpdated Output:\n```\nRewritten document.\n```"
                )
            }
        )
        result = executor(backend).run(
            "Original document.", method_id="geo.statistics_addition"
        )
        self.assertEqual(result.realized_document, "Rewritten document.")
        self.assertEqual(result.source_document_tokens, 100)
        self.assertEqual(result.rewrite_max_new_tokens, 381)
        self.assertEqual(result.stage_records[0].max_new_tokens, 381)
        self.assertEqual(len(result.stage_records), 1)
        self.assertFalse(result.query_visible_to_optimizer)

    def test_cseo_guidance_is_prepended_without_replacing_original(self) -> None:
        backend = ScriptedBackend(
            {"cseo.llm_guidance.rewrite": "# Guidance\n\nUseful summary."}
        )
        result = executor(backend).run(
            "Original document.", method_id="cseo.llm_guidance"
        )
        self.assertEqual(
            result.realized_document,
            "# Guidance\n\nUseful summary.\n\nOriginal document.",
        )

    def test_raid_runs_three_role_separated_stages(self) -> None:
        backend = ScriptedBackend(
            {
                "raid_gseo.summary": "Summary: concise summary",
                "raid_gseo.generalized_intent": "Generalized Intent: broad intent",
                "raid_gseo.rewrite": "Optimized Source: improved document",
            }
        )
        result = executor(backend).run(
            "Original document.", method_id="raid_gseo.full_pipeline"
        )
        self.assertEqual(result.realized_document, "improved document")
        self.assertEqual(
            [stage.stage_id for stage in result.stage_records],
            [
                "raid_gseo.summary",
                "raid_gseo.generalized_intent",
                "raid_gseo.rewrite",
            ],
        )
        for _, messages in backend.calls:
            self.assertEqual([message["role"] for message in messages], ["system", "user"])


class IfGeoMethodTest(unittest.TestCase):
    def test_all_methods_use_uniform_doubled_final_budget(self) -> None:
        catalog = LiteratureMethodCatalog(CATALOG_ROOT)
        for method_id in catalog.method_ids:
            result = LiteratureMethodExecutor(
                catalog=catalog, backend=DeterministicDryRunBackend(),
                budgets=StageBudgets(2048, 2.0, 256, 8192),
            ).run("Source sentence.", method_id=method_id)
            expected = result.source_document_tokens * 2 + 256
            self.assertEqual(result.rewrite_max_new_tokens, expected)
            self.assertEqual(result.stage_records[-1].max_new_tokens, expected)

    def test_audit_uses_aggregation_cap_for_downstream_inputs(self) -> None:
        from audit_literature_method_rewrite_lengths import stage_plans
        plans = stage_plans(LiteratureMethodCatalog(CATALOG_ROOT),
            method_id="if_geo.full_pipeline", document="Source sentence.",
            intermediate_cap=2048, rewrite_cap=3650, aggregation_cap=8192)
        self.assertEqual([p["max_new_tokens"] for p in plans], [2048, 2048, 8192, 8192, 8192, 3650])
        self.assertEqual([p["upstream_output_token_allowance"] for p in plans], [0, 2048, 12288, 8192, 8192, 8192])

    def test_aggregation_budget_changes_only_three_stages(self) -> None:
        baseline = executor(DeterministicDryRunBackend()).run(
            "Source sentence.", method_id="if_geo.full_pipeline"
        )
        changed = LiteratureMethodExecutor(
            catalog=LiteratureMethodCatalog(CATALOG_ROOT),
            backend=DeterministicDryRunBackend(),
            budgets=StageBudgets(128, 1.25, 256, 8192),
        ).run("Source sentence.", method_id="if_geo.full_pipeline")
        aggregate = {"if_geo.prioritization_deduplication", "if_geo.conflict_resolution", "if_geo.blueprint_construction"}
        self.assertEqual(len(changed.stage_records), 10)
        for old, new in zip(baseline.stage_records, changed.stage_records):
            self.assertEqual(old.messages_sha256, new.messages_sha256)
            self.assertEqual(new.max_new_tokens, 8192 if new.stage_id in aggregate else old.max_new_tokens)
        self.assertEqual(baseline.rewrite_max_new_tokens, changed.rewrite_max_new_tokens)

    def test_aggregation_budget_rejects_nonpositive_values(self) -> None:
        for value in (0, -1):
            with self.assertRaisesRegex(ValueError, "aggregation_max_new_tokens"):
                StageBudgets(128, 1.25, 256, value)

    def test_if_geo_runs_ten_calls_and_prefilters_necessity_below_60(self) -> None:
        backend = DeterministicDryRunBackend()
        result = executor(backend).run(
            "Source sentence.", method_id="if_geo.full_pipeline"
        )
        self.assertEqual(len(result.stage_records), 10)
        self.assertEqual(result.realized_document, "dry-run IF-GEO document")
        dedup_messages = dict(backend.calls)["if_geo.prioritization_deduplication"]
        dedup_user = dedup_messages[-1]["content"]
        self.assertIn('"necessity": 80', dedup_user)
        self.assertNotIn('"necessity": 40', dedup_user)
        self.assertNotIn("g < 0.7", dedup_user)

    def test_if_geo_invalid_json_stops_without_retry(self) -> None:
        backend = ScriptedBackend({"if_geo.query_mining": "Here is JSON: {}"})
        method_executor = executor(backend)
        with self.assertRaisesRegex(ValueError, "no semantic repair or retry") as raised:
            method_executor.run(
                "Source sentence.", method_id="if_geo.full_pipeline"
            )
        self.assertEqual(len(backend.calls), 1)
        payload = build_condition_error_payload(
            experiment_id="diagnostic-test",
            sample_id="sample-1",
            method_id="if_geo.full_pipeline",
            executor=method_executor,
            exc=raised.exception,
        )
        self.assertEqual(payload["completed_generation_call_count"], 1)
        self.assertEqual(
            payload["last_completed_stage_record"]["stage_id"],
            "if_geo.query_mining",
        )
        self.assertEqual(
            payload["last_completed_stage_record"]["output"],
            "Here is JSON: {}",
        )
        self.assertTrue(payload["diagnostic_only_no_output_repair"])

    def test_if_geo_accepts_one_json_fence_and_records_normalization(self) -> None:
        result = executor(FencedQueryDryRunBackend()).run(
            "Source sentence.", method_id="if_geo.full_pipeline"
        )
        self.assertEqual(len(result.stage_records), 10)
        self.assertTrue(result.stage_records[0].serialization_normalized)
        self.assertTrue(
            all(
                not record.serialization_normalized
                for record in result.stage_records[1:]
            )
        )

    def test_json_parser_rejects_prose_around_valid_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "no semantic repair or retry"):
            parse_json_strict_or_single_json_fence(
                'Here is the result: {"queries": []}',
                stage_id="if_geo.query_mining",
            )

    def test_query_mining_diagnostic_accepts_exact_json_fence(self) -> None:
        output = json.dumps(
            {
                "queries": [
                    {"query": f"query {index}", "probability": 90 - index}
                    for index in range(5)
                ]
            }
        )
        analysis = analyze_query_mining_output(
            "```json\n" + output + "\n```", expected_queries=5
        )
        self.assertFalse(analysis["raw_json_parse_success"])
        self.assertTrue(analysis["protocol_parse_success"])
        self.assertTrue(analysis["serialization_normalized"])
        self.assertTrue(analysis["schema_valid"])
        self.assertTrue(analysis["protocol_compliant"])

    def test_json_parser_rejects_ambiguous_wrappers(self) -> None:
        for text in (
            'Explanation\n```json\n{}\n```',
            '```json\n{}\n```\nExplanation',
            '```json\n{}\n```\n```json\n{}\n```',
            '```\n{}\n```',
            '```python\n{}\n```',
            '```json\n{"broken":}\n```',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json_strict_or_single_json_fence(text, stage_id="test")

    def test_fenced_schema_failure_retains_original_output(self) -> None:
        text = '```json\n{"queries": []}\n```'
        backend = ScriptedBackend({"if_geo.query_mining": text})
        method_executor = executor(backend)
        with self.assertRaisesRegex(ValueError, "exactly 5 queries"):
            method_executor.run("Source sentence.", method_id="if_geo.full_pipeline")
        record = method_executor.completed_generation_records[-1]
        self.assertTrue(record.serialization_normalized)
        self.assertEqual(record.output, text)
        self.assertEqual(record.raw_output, text)
        self.assertEqual(len(backend.calls), 1)

    def test_query_mining_diagnostic_accepts_exact_valid_payload(self) -> None:
        output = json.dumps(
            {
                "queries": [
                    {"query": f"query {index}", "probability": 90 - index}
                    for index in range(5)
                ]
            }
        )
        analysis = analyze_query_mining_output(output, expected_queries=5)
        self.assertTrue(analysis["raw_json_parse_success"])
        self.assertTrue(analysis["protocol_parse_success"])
        self.assertFalse(analysis["serialization_normalized"])
        self.assertTrue(analysis["schema_valid"])
        self.assertTrue(analysis["protocol_compliant"])

    def test_if_geo_requires_exactly_five_unique_queries(self) -> None:
        backend = ScriptedBackend(
            {
                "if_geo.query_mining": json.dumps(
                    {"queries": [{"query": "only one", "probability": 90}]}
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly 5 queries"):
            executor(backend).run(
                "Source sentence.", method_id="if_geo.full_pipeline"
            )

    def test_pre_smoke_length_plans_cover_all_method_stages(self) -> None:
        catalog = LiteratureMethodCatalog(CATALOG_ROOT)
        fixed = stage_plans(
            catalog,
            method_id="geo.cite_sources",
            document="Source sentence.",
            intermediate_cap=2048,
            rewrite_cap=512,
        )
        raid = stage_plans(
            catalog,
            method_id="raid_gseo.full_pipeline",
            document="Source sentence.",
            intermediate_cap=2048,
            rewrite_cap=512,
        )
        if_geo = stage_plans(
            catalog,
            method_id="if_geo.full_pipeline",
            document="Source sentence.",
            intermediate_cap=2048,
            rewrite_cap=512,
        )
        self.assertEqual(len(fixed), 1)
        self.assertEqual(len(raid), 3)
        self.assertEqual(len(if_geo), 6)
        self.assertEqual(
            if_geo[2]["upstream_output_token_allowance"], 6 * 2048
        )


if __name__ == "__main__":
    unittest.main()
