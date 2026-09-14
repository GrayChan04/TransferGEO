"""Generate the frozen realized texts for the formal necessity pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from create_necessity_manifest import load_cseo, load_geo
from glm_rewriter import GlmRewriter, PromptCatalog


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ORDER = (
    "neutral_rewrite_control.neutral_rewrite_control",
    "evidence_structure_rewriting.conclusion_first",
    "evidence_structure_rewriting.key_point_enumeration",
    "evidence_structure_rewriting.hierarchical_framing",
    "latent_intent_coverage.explicit_purpose",
    "latent_intent_coverage.single_latent_intent",
    "latent_intent_coverage.multiple_latent_intents",
    "readability_technicality.general_audience_register",
    "readability_technicality.mixed_audience_register",
    "readability_technicality.specialist_register",
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl")
    p.add_argument("--geo-path", type=Path, default=ROOT / "data/raw/benchmarks/geo_bench/test.jsonl")
    p.add_argument("--cseo-root", type=Path, default=ROOT / "data/raw/benchmarks/cseo_bench/data")
    p.add_argument("--prompt-catalog", type=Path, default=ROOT / "prompts/necessity_pilot/v12")
    p.add_argument("--model-root", type=Path, default=ROOT / "data/raw/models")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    a = args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {a.output_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in a.manifest.read_text(encoding="utf-8").splitlines()]
    source_rows = load_geo(a.geo_path) + load_cseo(a.cseo_root)
    source_map = {f"{r['benchmark']}__{r['domain']}__{r['instance_id']}": r for r in source_rows}
    catalog = PromptCatalog(a.prompt_catalog)
    records: list[dict] = []
    errors: list[dict] = []
    with GlmRewriter(model_root=a.model_root, device_map_strategy="balanced") as rewriter:
        for index, item in enumerate(manifest, start=1):
            key = item["sample_id"]
            source = source_map[key]
            target_index = int(item["target_document_index"])
            document = source["documents"][target_index]
            for prompt_id in PROMPT_ORDER:
                try:
                    result = rewriter.rewrite(document, prompt_id=prompt_id, catalog=catalog, seed=a.seed)
                    records.append({
                        "sample_id": key,
                        "benchmark": item["benchmark"],
                        "domain": item["domain"],
                        "instance_id": item["instance_id"],
                        "query": item["query"],
                        "target_document_index": target_index,
                        "prompt_id": prompt_id,
                        "source_text": document,
                        "rewritten_text": result.answer,
                        "rewrite": result.to_dict(),
                    })
                except Exception as exc:  # preserve per-condition failure and continue
                    errors.append({"sample_id": key, "prompt_id": prompt_id, "error_type": type(exc).__name__, "error": str(exc)})
            if index % 10 == 0:
                print(f"rewritten {index}/{len(manifest)}", flush=True)
    (a.output_dir / "rewrites.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    metrics = {"experiment_id": "necessity_rewrite_stage_v1", "manifest_count": len(manifest), "expected_conditions": len(manifest) * len(PROMPT_ORDER), "completed_conditions": len(records), "error_count": len(errors), "errors": errors, "prompt_ids": list(PROMPT_ORDER), "query_visible_to_rewriter": False}
    (a.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("manifest_count", "expected_conditions", "completed_conditions", "error_count")}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
