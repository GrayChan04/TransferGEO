"""Audit tokenizer-specific benchmark input lengths without loading model weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

from answer_prompt import build_cseo_messages, build_geo_messages
from model_loader import DEFAULT_MODEL_ROOT, MODEL_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEO_PATH = (
    PROJECT_ROOT / "data" / "raw" / "benchmarks" / "geo_bench" / "test.jsonl"
)
DEFAULT_CSEO_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "benchmarks" / "cseo_bench" / "data"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "length_audit"

DEFAULT_INPUT_BUDGETS = (
    4_096,
    8_192,
    16_384,
    24_576,
    28_672,
    30_720,
    31_744,
    32_768,
)


@dataclass(frozen=True)
class BenchmarkInstance:
    benchmark: str
    domain: str
    instance_id: str
    query: str
    documents: tuple[str, ...]

    def messages(self) -> list[dict[str, str]]:
        if self.benchmark == "geo_bench":
            return build_geo_messages(self.query, self.documents)
        if self.benchmark == "cseo_bench":
            return build_cseo_messages(self.domain, self.query, self.documents)
        raise ValueError(f"Unsupported benchmark: {self.benchmark}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit full answer-generation input lengths under each model's official "
            "chat template."
        )
    )
    parser.add_argument("--geo-path", type=Path, default=DEFAULT_GEO_PATH)
    parser.add_argument("--cseo-root", type=Path, default=DEFAULT_CSEO_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument(
        "--geo-source-field",
        choices=("cleaned_text", "raw_text"),
        default="cleaned_text",
        help="GEO-Bench source field. The official dataset describes cleaned_text as cleaned HTML.",
    )
    parser.add_argument(
        "--input-budgets",
        nargs="+",
        type=int,
        default=list(DEFAULT_INPUT_BUDGETS),
        help="Candidate input-token budgets for overflow-rate reporting.",
    )
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Diagnostic-only cap: GEO total and each C-SEO domain. Omit for the full audit.",
    )
    return parser.parse_args()


def load_geo_instances(
    path: Path,
    *,
    source_field: str,
    limit: int | None,
) -> list[BenchmarkInstance]:
    instances: list[BenchmarkInstance] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            documents = tuple(source[source_field] for source in row["sources"])
            instances.append(
                BenchmarkInstance(
                    benchmark="geo_bench",
                    domain="all",
                    instance_id=str(index),
                    query=row["query"],
                    documents=documents,
                )
            )
    if not instances:
        raise ValueError(f"No GEO-Bench instances loaded from {path}")
    return instances


def load_cseo_instances(
    root: Path,
    *,
    limit_per_domain: int | None,
) -> list[BenchmarkInstance]:
    instances: list[BenchmarkInstance] = []
    parquet_paths = sorted(root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No C-SEO parquet files found under {root}")

    for path in parquet_paths:
        domain = path.name.split("-", maxsplit=1)[0]
        table = pq.read_table(path, columns=["query_id", "query", "document"])
        grouped: OrderedDict[int, dict[str, Any]] = OrderedDict()
        for query_id, query, document in zip(
            table.column("query_id").to_pylist(),
            table.column("query").to_pylist(),
            table.column("document").to_pylist(),
            strict=True,
        ):
            entry = grouped.setdefault(
                int(query_id),
                {"query": query, "documents": []},
            )
            if entry["query"] != query:
                raise ValueError(
                    f"Inconsistent query strings for {domain} query_id={query_id}"
                )
            if len(entry["documents"]) < 10:
                entry["documents"].append(document)

        selected = list(grouped.items())
        if limit_per_domain is not None:
            selected = selected[:limit_per_domain]
        for query_id, entry in selected:
            instances.append(
                BenchmarkInstance(
                    benchmark="cseo_bench",
                    domain=domain,
                    instance_id=str(query_id),
                    query=entry["query"],
                    documents=tuple(entry["documents"]),
                )
            )

    if not instances:
        raise ValueError(f"No C-SEO instances loaded from {root}")
    return instances


def model_context_limit(config: Any, tokenizer: Any) -> int | None:
    candidates = [
        getattr(config, "max_position_embeddings", None),
        getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    finite = [
        int(value)
        for value in candidates
        if isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 < value < 10**9
    ]
    return min(finite) if finite else None


def count_chat_tokens(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    disable_thinking: bool,
) -> int:
    template_kwargs: dict[str, Any] = {}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **template_kwargs,
    )
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def percentile(sorted_values: Sequence[int], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def describe(values: Iterable[int], budgets: Sequence[int]) -> dict[str, Any]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    return {
        "count": count,
        "min": sorted_values[0],
        "p50": round(percentile(sorted_values, 0.50), 2),
        "p90": round(percentile(sorted_values, 0.90), 2),
        "p95": round(percentile(sorted_values, 0.95), 2),
        "p99": round(percentile(sorted_values, 0.99), 2),
        "max": sorted_values[-1],
        "overflow": {
            str(budget): {
                "count": sum(value > budget for value in sorted_values),
                "rate": round(
                    sum(value > budget for value in sorted_values) / count,
                    6,
                ),
            }
            for budget in budgets
        },
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    budgets: Sequence[int],
    model_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["domain"], row["model"])].append(
            int(row["input_tokens"])
        )

    benchmark_summary: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        model_summary = {}
        for model in sorted({row["model"] for row in benchmark_rows}):
            values = [
                int(row["input_tokens"])
                for row in benchmark_rows
                if row["model"] == model
            ]
            model_summary[model] = describe(values, budgets)

        domain_summary = {}
        for domain in sorted({row["domain"] for row in benchmark_rows}):
            domain_summary[domain] = {
                model: describe(values, budgets)
                for (row_benchmark, row_domain, model), values in sorted(grouped.items())
                if row_benchmark == benchmark and row_domain == domain
            }

        unique_instances = {
            (row["domain"], row["instance_id"]) for row in benchmark_rows
        }
        benchmark_summary[benchmark] = {
            "instances": len(unique_instances),
            "models": model_summary,
            "domains": domain_summary,
        }

    return {
        "audit_name": "benchmark_answer_input_length",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "token_count_definition": (
            "Full answer-generation messages after the model-native chat template, "
            "including the generation prompt and with Qwen thinking disabled."
        ),
        "candidate_input_budgets": list(budgets),
        "model_metadata": model_metadata,
        "benchmarks": benchmark_summary,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "domain",
        "instance_id",
        "query",
        "document_count",
        "document_chars",
        "empty_document_count",
        "model",
        "model_id",
        "model_context_limit",
        "input_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    budgets = sorted(set(args.input_budgets))
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("All input budgets must be positive")

    geo_instances = load_geo_instances(
        args.geo_path,
        source_field=args.geo_source_field,
        limit=args.limit_per_group,
    )
    cseo_instances = load_cseo_instances(
        args.cseo_root,
        limit_per_domain=args.limit_per_group,
    )
    instances = [*geo_instances, *cseo_instances]
    print(
        f"Loaded {len(geo_instances)} GEO-Bench and {len(cseo_instances)} "
        "C-SEO Bench instances.",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        model_path = spec.path(args.model_root)
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
        context_limit = model_context_limit(config, tokenizer)
        model_metadata[model_name] = {
            "model_id": spec.model_id,
            "model_path": str(model_path),
            "model_context_limit": context_limit,
            "tokenizer_model_max_length": int(tokenizer.model_max_length),
            "thinking_disabled": spec.disable_thinking,
        }
        print(
            f"Auditing {model_name}: {len(instances)} inputs "
            f"(context limit: {context_limit}).",
            flush=True,
        )

        for index, instance in enumerate(instances, start=1):
            input_tokens = count_chat_tokens(
                tokenizer,
                instance.messages(),
                disable_thinking=spec.disable_thinking,
            )
            rows.append(
                {
                    "benchmark": instance.benchmark,
                    "domain": instance.domain,
                    "instance_id": instance.instance_id,
                    "query": instance.query.replace("\r", " ").replace("\n", " "),
                    "document_count": len(instance.documents),
                    "document_chars": sum(len(document) for document in instance.documents),
                    "empty_document_count": sum(
                        not document.strip() for document in instance.documents
                    ),
                    "model": model_name,
                    "model_id": spec.model_id,
                    "model_context_limit": context_limit,
                    "input_tokens": input_tokens,
                }
            )
            if index % 500 == 0 or index == len(instances):
                print(
                    f"  {model_name}: {index}/{len(instances)} complete.",
                    flush=True,
                )
        del tokenizer, config

    summary = summarize_rows(
        rows,
        budgets=budgets,
        model_metadata=model_metadata,
    )
    summary["inputs"] = {
        "geo_path": str(args.geo_path),
        "geo_source_field": args.geo_source_field,
        "cseo_root": str(args.cseo_root),
        "limit_per_group": args.limit_per_group,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_instance_lengths.csv"
    summary_path = args.output_dir / "input_length_summary.json"
    write_csv(csv_path, rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
