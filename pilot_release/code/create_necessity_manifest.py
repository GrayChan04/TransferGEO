"""Create a deterministic, query-visible Answer Model pilot manifest.

The Rewriter never consumes this manifest's query field.  The manifest binds
the official benchmark instance, target document, candidate ordering, and
baseline model-native input lengths before any rewritten text is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from answer_prompt import build_cseo_messages, build_geo_messages
from model_loader import DEFAULT_MODEL_ROOT, MODEL_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEO_PATH = PROJECT_ROOT / "data/raw/benchmarks/geo_bench/test.jsonl"
DEFAULT_CSEO_ROOT = PROJECT_ROOT / "data/raw/benchmarks/cseo_bench/data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/necessity_pilot_v1"
DEFAULT_BUDGET = 20_480

CALIBRATION_EXCLUSIONS = {
    ("geo_bench", "all", "142"),
    ("geo_bench", "all", "189"),
    ("geo_bench", "all", "360"),
    ("geo_bench", "all", "457"),
    ("geo_bench", "all", "940"),
    ("cseo_bench", "books", "57"),
    ("cseo_bench", "news", "342"),
    ("cseo_bench", "retail", "80681"),
    ("cseo_bench", "videogames", "35"),
    ("cseo_bench", "videogames", "413"),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-path", type=Path, default=DEFAULT_GEO_PATH)
    parser.add_argument("--cseo-root", type=Path, default=DEFAULT_CSEO_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geo-samples", type=int, default=200)
    parser.add_argument("--cseo-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--input-budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_geo(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            documents = [str(source["cleaned_text"]) for source in row["sources"]]
            rows.append(
                {
                    "benchmark": "geo_bench",
                    "domain": "all",
                    "instance_id": str(index),
                    "query": str(row["query"]),
                    "documents": documents,
                    "target_document_index": int(row["sugg_idx"]),
                    "tags": list(row.get("tags", [])),
                }
            )
    return rows


def load_cseo(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*.parquet")):
        domain = path.name.split("-", maxsplit=1)[0]
        table = pq.read_table(path, columns=["query_id", "query", "document"])
        grouped: dict[int, dict[str, Any]] = {}
        for query_id, query, document in zip(
            table.column("query_id").to_pylist(),
            table.column("query").to_pylist(),
            table.column("document").to_pylist(),
            strict=True,
        ):
            query_id = int(query_id)
            entry = grouped.setdefault(query_id, {"query": str(query), "documents": []})
            if entry["query"] != str(query):
                raise ValueError(f"Inconsistent query text: {domain}/{query_id}")
            if len(entry["documents"]) < 10:
                entry["documents"].append(str(document))
        for query_id, entry in grouped.items():
            rows.append(
                {
                    "benchmark": "cseo_bench",
                    "domain": domain,
                    "instance_id": str(query_id),
                    "query": entry["query"],
                    "documents": entry["documents"],
                    "target_document_index": None,
                }
            )
    return rows


def choose_balanced(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    domains = sorted(by_domain)
    base, remainder = divmod(count, len(domains))
    selected = []
    for index, domain in enumerate(domains):
        quota = base + (index < remainder)
        candidates = [
            row for row in by_domain[domain]
            if (row["benchmark"], row["domain"], row["instance_id"]) not in CALIBRATION_EXCLUSIONS
        ]
        if len(candidates) < quota:
            raise ValueError(f"Not enough eligible {domain} instances for quota {quota}")
        selected.extend(rng.sample(candidates, quota))
    rng.shuffle(selected)
    return selected


def count_tokens(tokenizer: Any, messages: list[dict[str, str]], disable_thinking: bool) -> int:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True, "return_dict": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    ids = encoded["input_ids"]
    return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)


def main() -> int:
    args = parse_args()
    if args.geo_samples <= 0 or args.cseo_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}; use --force explicitly")

    rng = random.Random(args.seed)
    all_rows = load_geo(args.geo_path) + load_cseo(args.cseo_root)

    tokenizers = {}
    model_metadata = {}
    for name, spec in MODEL_SPECS.items():
        tokenizer = AutoTokenizer.from_pretrained(spec.path(args.model_root), local_files_only=True, use_fast=True)
        tokenizers[name] = tokenizer
        model_metadata[name] = {"model_id": spec.model_id, "model_path": str(spec.path(args.model_root))}

    eligible_rows: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    for row in all_rows:
        key = (row["benchmark"], row["domain"], row["instance_id"])
        if key in CALIBRATION_EXCLUSIONS:
            continue
        messages = (
            build_geo_messages(row["query"], row["documents"])
            if row["benchmark"] == "geo_bench"
            else build_cseo_messages(row["domain"], row["query"], row["documents"])
        )
        row["baseline_input_tokens"] = {
            name: count_tokens(tokenizer, messages, MODEL_SPECS[name].disable_thinking)
            for name, tokenizer in tokenizers.items()
        }
        if all(value <= args.input_budget for value in row["baseline_input_tokens"].values()):
            eligible_rows.append(row)
        else:
            overflow_rows.append(row)

    geo = [row for row in eligible_rows if row["benchmark"] == "geo_bench"]
    cseo = [row for row in eligible_rows if row["benchmark"] == "cseo_bench"]
    selected_geo = rng.sample(geo, args.geo_samples)
    selected_cseo = choose_balanced(cseo, args.cseo_samples, rng)
    target_rng = random.Random(args.seed + 1)
    for row in selected_cseo:
        row["target_document_index"] = target_rng.randrange(len(row["documents"]))
    rows = selected_geo + selected_cseo
    rng.shuffle(rows)
    manifest = []
    for index, row in enumerate(rows):
        row_key = f"{row['benchmark']}__{row['domain']}__{row['instance_id']}"
        record = {
            "manifest_index": index,
            "sample_id": row_key,
            "benchmark": row["benchmark"],
            "domain": row["domain"],
            "instance_id": row["instance_id"],
            "query": row["query"],
            "tags": row.get("tags", []),
            "candidate_document_count": len(row["documents"]),
            "target_document_index": row["target_document_index"],
            "target_document_sha256": sha256_text(row["documents"][row["target_document_index"]]),
            "candidate_document_sha256": [sha256_text(document) for document in row["documents"]],
            "baseline_input_tokens": row["baseline_input_tokens"],
            "input_budget": args.input_budget,
            "baseline_budget_eligible": True,
            "calibration_excluded": False,
        }
        manifest.append(record)

    if len([r for r in manifest if r["benchmark"] == "geo_bench"]) != args.geo_samples:
        raise ValueError("GEO sample quota changed after baseline input eligibility filtering")
    if len([r for r in manifest if r["benchmark"] == "cseo_bench"]) != args.cseo_samples:
        raise ValueError("C-SEO sample quota changed after baseline input eligibility filtering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "manifest_id": "necessity_pilot_v1",
        "seed": args.seed,
        "target_selection_seed": args.seed + 1,
        "geo_requested": args.geo_samples,
        "cseo_requested": args.cseo_samples,
        "geo_selected": sum(r["benchmark"] == "geo_bench" for r in manifest),
        "cseo_selected": sum(r["benchmark"] == "cseo_bench" for r in manifest),
        "eligible_pool_counts": {"geo_bench": len(geo), "cseo_bench": len(cseo)},
        "excluded_baseline_overflow_count": len(overflow_rows),
        "calibration_exclusions": sorted("/".join(item) for item in CALIBRATION_EXCLUSIONS),
        "model_metadata": model_metadata,
        "input_budget": args.input_budget,
        "candidate_order_policy": "official dataset order",
        "target_policy": {
            "geo_bench": "official sugg_idx",
            "cseo_bench": "one official-protocol random target document per query",
        },
        "manifest_path": str(manifest_path),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("geo_selected", "cseo_selected", "input_budget")}, ensure_ascii=False))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {args.output_dir / 'manifest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
