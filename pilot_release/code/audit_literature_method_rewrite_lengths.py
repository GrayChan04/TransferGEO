"""CPU-only context audit for the literature-method rewrite runner.

Entry stages are measured exactly.  Downstream multi-stage inputs depend on
outputs that do not exist before the rewrite smoke, so they are reported as
conservative upper bounds derived from the frozen intermediate generation cap.
The audit loads only the local GLM tokenizer/config and never loads model
weights or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from literature_method_runner import (
    LiteratureMethodCatalog,
    StageBudgets,
    build_messages,
    render_template,
)
from run_literature_method_rewrite_stage import (
    DEFAULT_CSEO_ROOT,
    DEFAULT_GEO_PATH,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ROOT,
    load_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_MARGIN_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--geo-path", type=Path, default=DEFAULT_GEO_PATH)
    parser.add_argument("--cseo-root", type=Path, default=DEFAULT_CSEO_ROOT)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "prompts/necessity_pilot/literature_methods_v1",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--intermediate-max-new-tokens", type=int, required=True)
    parser.add_argument("--if-geo-aggregation-max-new-tokens", type=int)
    parser.add_argument("--rewrite-growth-ratio", type=float, required=True)
    parser.add_argument("--rewrite-growth-slack-tokens", type=int, required=True)
    parser.add_argument("--expected-manifest-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_ids(value: Any) -> int:
    if hasattr(value, "shape"):
        return int(value.shape[-1])
    if value and isinstance(value[0], list):
        return len(value[0])
    return len(value)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return count_ids(encoded["input_ids"])


def count_message_tokens(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
) -> int:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    return count_ids(encoded["input_ids"])


def stage_plans(
    catalog: LiteratureMethodCatalog,
    *,
    method_id: str,
    document: str,
    intermediate_cap: int,
    rewrite_cap: int,
    aggregation_cap: int | None = None,
) -> list[dict[str, Any]]:
    spec = catalog.get(method_id)
    if method_id.startswith("geo.") or method_id.startswith("cseo."):
        system_path = spec.payload.get("system_prompt_path")
        system = catalog.read_prompt(str(system_path)) if system_path else None
        user = render_template(
            catalog.read_prompt(str(spec.payload["user_prompt_path"])),
            DOCUMENT=document,
        )
        return [
            {
                "stage_id": f"{method_id}.rewrite",
                "messages": build_messages(system, user),
                "max_new_tokens": rewrite_cap,
                "upstream_output_token_allowance": 0,
                "bound_type": "exact_entry_stage",
            }
        ]

    if method_id == "raid_gseo.full_pipeline":
        system = catalog.read_prompt(str(spec.payload["system_prompt_path"]))
        paths = [str(path) for path in spec.payload["stage_prompt_paths"]]
        summary = render_template(catalog.read_prompt(paths[0]), DOCUMENT=document)
        intent = render_template(
            catalog.read_prompt(paths[1]), DOCUMENT=document, SUMMARY="x"
        )
        rewrite = render_template(
            catalog.read_prompt(paths[2]),
            DOCUMENT=document,
            SUMMARY="x",
            GENERALIZED_INTENT="x",
        )
        return [
            {
                "stage_id": "raid_gseo.summary",
                "messages": build_messages(system, summary),
                "max_new_tokens": intermediate_cap,
                "upstream_output_token_allowance": 0,
                "bound_type": "exact_entry_stage",
            },
            {
                "stage_id": "raid_gseo.generalized_intent",
                "messages": build_messages(system, intent),
                "max_new_tokens": intermediate_cap,
                "upstream_output_token_allowance": intermediate_cap,
                "bound_type": "conservative_pre_smoke_upper_bound",
            },
            {
                "stage_id": "raid_gseo.rewrite",
                "messages": build_messages(system, rewrite),
                "max_new_tokens": rewrite_cap,
                "upstream_output_token_allowance": 2 * intermediate_cap,
                "bound_type": "conservative_pre_smoke_upper_bound",
            },
        ]

    if method_id == "if_geo.full_pipeline":
        aggregation_cap = aggregation_cap or intermediate_cap
        system_paths = [str(path) for path in spec.payload["system_prompt_paths"]]
        user_paths = [str(path) for path in spec.payload["stage_user_template_paths"]]
        params = spec.payload["fixed_parameters_from_paper"]
        systems = [catalog.read_prompt(path) for path in system_paths]
        systems[0] = systems[0].replace("{num_queries}", str(params["num_queries"]))
        systems[1] = systems[1].replace(
            "{suggestions_num}", str(params["suggestions_num"])
        )
        users = [
            render_template(catalog.read_prompt(user_paths[0]), DOCUMENT=document),
            render_template(
                catalog.read_prompt(user_paths[1]),
                LATENT_QUERY="x",
                DOCUMENT=document,
            ),
            render_template(
                catalog.read_prompt(user_paths[2]),
                DOCUMENT=document,
                GROUPED_SUGGESTIONS_JSON="x",
            ),
            render_template(
                catalog.read_prompt(user_paths[3]),
                DOCUMENT=document,
                DEDUPLICATED_SUGGESTIONS_JSON="x",
            ),
            render_template(
                catalog.read_prompt(user_paths[4]),
                DOCUMENT=document,
                RESOLVED_SUGGESTIONS_JSON="x",
            ),
            render_template(
                catalog.read_prompt(user_paths[5]),
                DOCUMENT=document,
                REVISION_BLUEPRINT_JSON="x",
            ),
        ]
        stage_ids = (
            "if_geo.query_mining",
            "if_geo.query_request",
            "if_geo.prioritization_deduplication",
            "if_geo.conflict_resolution",
            "if_geo.blueprint_construction",
            "if_geo.blueprint_guided_revision",
        )
        upstream_allowances = (
            0,
            intermediate_cap,
            6 * intermediate_cap,
            aggregation_cap,
            aggregation_cap,
            aggregation_cap,
        )
        return [
            {
                "stage_id": stage_id,
                "messages": build_messages(system, user),
                "max_new_tokens": (
                    rewrite_cap if stage_id.endswith("guided_revision") else
                    aggregation_cap if stage_id in stage_ids[2:5] else intermediate_cap
                ),
                "upstream_output_token_allowance": upstream_allowance,
                "bound_type": (
                    "exact_entry_stage"
                    if stage_id == "if_geo.query_mining"
                    else "conservative_pre_smoke_upper_bound"
                ),
            }
            for stage_id, system, user, upstream_allowance in zip(
                stage_ids, systems, users, upstream_allowances, strict=True
            )
        ]
    raise NotImplementedError(method_id)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    catalog = LiteratureMethodCatalog(args.catalog_root)
    manifest, source_map = load_inputs(args.manifest, args.geo_path, args.cseo_root)
    if len(manifest) != args.expected_manifest_count:
        raise ValueError(
            f"Expected {args.expected_manifest_count} manifest rows, found {len(manifest)}"
        )
    budgets = StageBudgets(
        intermediate_max_new_tokens=args.intermediate_max_new_tokens,
        if_geo_aggregation_max_new_tokens=args.if_geo_aggregation_max_new_tokens,
        rewrite_growth_ratio=args.rewrite_growth_ratio,
        rewrite_growth_slack_tokens=args.rewrite_growth_slack_tokens,
    )

    from transformers import AutoConfig, AutoTokenizer

    model_path = args.model_root / "GLM-4-9B-0414"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    context_limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit <= 0:
        raise RuntimeError("GLM config lacks a valid max_position_embeddings")

    rows: list[dict[str, Any]] = []
    stage_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "overflow_count": 0, "max_total_tokens": 0}
    )
    for item in manifest:
        sample_id = str(item["sample_id"])
        document = str(
            source_map[sample_id]["documents"][int(item["target_document_index"])]
        )
        source_tokens = count_text_tokens(tokenizer, document)
        rewrite_cap = budgets.rewrite_max_new_tokens(source_tokens)
        for method_id in catalog.method_ids:
            for plan in stage_plans(
                catalog,
                method_id=method_id,
                document=document,
                intermediate_cap=budgets.intermediate_max_new_tokens,
                rewrite_cap=rewrite_cap,
                aggregation_cap=budgets.if_geo_aggregation_max_new_tokens,
            ):
                measured_base = count_message_tokens(tokenizer, plan["messages"])
                input_bound = (
                    measured_base
                    + int(plan["upstream_output_token_allowance"])
                    + (
                        BOUNDARY_MARGIN_TOKENS
                        if plan["upstream_output_token_allowance"]
                        else 0
                    )
                )
                total_bound = input_bound + int(plan["max_new_tokens"])
                within = total_bound <= context_limit
                row = {
                    "sample_id": sample_id,
                    "benchmark": item["benchmark"],
                    "method_id": method_id,
                    "stage_id": plan["stage_id"],
                    "bound_type": plan["bound_type"],
                    "source_tokens": source_tokens,
                    "measured_base_input_tokens": measured_base,
                    "upstream_output_token_allowance": plan[
                        "upstream_output_token_allowance"
                    ],
                    "boundary_margin_tokens": (
                        BOUNDARY_MARGIN_TOKENS
                        if plan["upstream_output_token_allowance"]
                        else 0
                    ),
                    "input_token_bound": input_bound,
                    "max_new_tokens": plan["max_new_tokens"],
                    "total_token_bound": total_bound,
                    "context_limit": context_limit,
                    "within_context": within,
                    "query_visible_to_optimizer": False,
                }
                rows.append(row)
                summary = stage_summary[plan["stage_id"]]
                summary["count"] += 1
                summary["overflow_count"] += int(not within)
                summary["max_total_tokens"] = max(
                    summary["max_total_tokens"], total_bound
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "length_audit_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    overflow_count = sum(not row["within_context"] for row in rows)
    metrics = {
        "experiment_id": args.experiment_id,
        "phase": "literature_method_rewrite_input_length_audit",
        "catalog_id": catalog.catalog_id,
        "manifest_count": len(manifest),
        "method_count": len(catalog.method_ids),
        "audit_row_count": len(rows),
        "context_limit": context_limit,
        "intermediate_max_new_tokens": budgets.intermediate_max_new_tokens,
        "if_geo_aggregation_max_new_tokens": budgets.if_geo_aggregation_max_new_tokens,
        "rewrite_growth_ratio": budgets.rewrite_growth_ratio,
        "rewrite_growth_slack_tokens": budgets.rewrite_growth_slack_tokens,
        "boundary_margin_tokens": BOUNDARY_MARGIN_TOKENS,
        "overflow_count": overflow_count,
        "all_bounds_within_context": overflow_count == 0,
        "gpu_required": False,
        "model_weights_loaded": False,
        "query_visible_to_optimizer": False,
        "bound_interpretation": {
            "exact_entry_stage": "Exact tokenizer count before any generated intermediate exists.",
            "conservative_pre_smoke_upper_bound": "Base prompt plus upstream generation caps and a 64-token boundary margin; replace with realized counts after smoke.",
        },
        "stage_summary": dict(sorted(stage_summary.items())),
        "input_provenance": {
            "manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256_file(args.manifest),
            },
            "method_catalog": {
                "path": str(catalog.catalog_path),
                "sha256": sha256_file(catalog.catalog_path),
            },
            "prompt_hash_manifest": {
                "path": str(catalog.hash_path),
                "sha256": sha256_file(catalog.hash_path),
            },
        },
        "result_paths": {
            "rows": str(rows_path.resolve()),
            "metrics": str((args.output_dir / "metrics.json").resolve()),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest_count": len(manifest),
                "audit_row_count": len(rows),
                "overflow_count": overflow_count,
                "all_bounds_within_context": overflow_count == 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if overflow_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
