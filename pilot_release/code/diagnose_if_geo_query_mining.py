"""Capture one real IF-GEO query-mining output without repairing it.

This diagnostic performs exactly one GLM call.  It stores the decoded and raw
model text before applying the existing strict JSON and schema checks.  A
non-compliant output is a successful diagnosis, not a crashed experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_method_runner import (
    LiteratureMethodCatalog,
    LiteratureMethodExecutor,
    StageBudgets,
    build_messages,
    canonical_json,
    parse_json_strict_or_single_json_fence,
    render_template,
    sha256_text,
)
from run_literature_method_rewrite_stage import GlmChatBackend, load_inputs


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geo-path", type=Path, required=True)
    parser.add_argument("--cseo-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--device-map-strategy", choices=("balanced", "none"), default="balanced"
    )
    return parser.parse_args()


def analyze_query_mining_output(text: str, *, expected_queries: int) -> dict[str, Any]:
    """Report both raw-JSON and current serialization-protocol outcomes."""

    try:
        json.loads(text)
        raw_json_parse_success = True
        raw_json_parse_error = None
    except json.JSONDecodeError as exc:
        raw_json_parse_success = False
        raw_json_parse_error = str(exc)

    try:
        payload, normalized = parse_json_strict_or_single_json_fence(
            text, stage_id="if_geo.query_mining"
        )
    except ValueError as exc:
        return {
            "raw_json_parse_success": raw_json_parse_success,
            "raw_json_parse_error": raw_json_parse_error,
            "protocol_parse_success": False,
            "serialization_normalized": False,
            "schema_valid": False,
            "protocol_compliant": False,
            "protocol_parse_error": str(exc),
            "schema_error": None,
        }

    try:
        LiteratureMethodExecutor._validate_queries(payload, expected=expected_queries)
    except ValueError as exc:
        return {
            "raw_json_parse_success": raw_json_parse_success,
            "raw_json_parse_error": raw_json_parse_error,
            "protocol_parse_success": True,
            "serialization_normalized": normalized,
            "schema_valid": False,
            "protocol_compliant": False,
            "protocol_parse_error": None,
            "schema_error": str(exc),
        }
    return {
        "raw_json_parse_success": raw_json_parse_success,
        "raw_json_parse_error": raw_json_parse_error,
        "protocol_parse_success": True,
        "serialization_normalized": normalized,
        "schema_valid": True,
        "protocol_compliant": True,
        "protocol_parse_error": None,
        "schema_error": None,
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog = LiteratureMethodCatalog(args.catalog_root)
    spec = catalog.get("if_geo.full_pipeline")
    manifest, source_map = load_inputs(args.manifest, args.geo_path, args.cseo_root)
    selected = [row for row in manifest if str(row["sample_id"]) == args.sample_id]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for {args.sample_id}, found {len(selected)}"
        )
    item = selected[0]
    document = str(
        source_map[args.sample_id]["documents"][int(item["target_document_index"])]
    )

    params = spec.payload["fixed_parameters_from_paper"]
    expected_queries = int(params["num_queries"])
    system_path = str(spec.payload["system_prompt_paths"][0])
    user_path = str(spec.payload["stage_user_template_paths"][0])
    system_prompt = catalog.read_prompt(system_path).replace(
        "{num_queries}", str(expected_queries)
    )
    user_prompt = render_template(catalog.read_prompt(user_path), DOCUMENT=document)
    messages = build_messages(system_prompt, user_prompt)

    strategy = None if args.device_map_strategy == "none" else "balanced"
    with GlmChatBackend(
        model_root=args.model_root,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map_strategy=strategy,
    ) as backend:
        generation = backend.generate(
            messages,
            stage_id="if_geo.query_mining",
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

    generation_record = {
        "experiment_id": args.experiment_id,
        "sample_id": args.sample_id,
        "method_id": "if_geo.full_pipeline",
        "stage_id": "if_geo.query_mining",
        "query_visible_to_optimizer": False,
        "messages_sha256": sha256_text(canonical_json(messages)),
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model_id": generation.model_id,
        "output_sha256": sha256_text(generation.answer),
        "output": generation.answer,
        "raw_output": generation.raw_answer,
        "diagnostic_only_no_output_repair": True,
    }
    generation_path = args.output_dir / "query_mining_generation.json"
    generation_path.write_text(
        json.dumps(generation_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    analysis = analyze_query_mining_output(
        generation.answer, expected_queries=expected_queries
    )
    metrics = {
        "experiment_id": args.experiment_id,
        "phase": "if_geo_query_mining_format_diagnostic",
        "diagnostic_completed": True,
        "model_call_count": 1,
        "sample_id": args.sample_id,
        "stage_id": "if_geo.query_mining",
        "query_visible_to_optimizer": False,
        "output_changed_or_repaired": False,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "max_new_tokens": args.max_new_tokens,
        "generation_path": str(generation_path.resolve()),
        **analysis,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
