"""Run or CPU-validate the query-invisible literature-method rewrite stage.

Formal GPU execution intentionally requires explicit method IDs, token budgets,
failure policy and output directory.  ``--validate-only`` loads no model and
writes no result files; it checks the catalog, the frozen 400-sample data join,
and every selected method's complete orchestration path with canned outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from create_necessity_manifest import load_cseo, load_geo
from literature_method_runner import (
    BackendGeneration,
    DeterministicDryRunBackend,
    LiteratureMethodCatalog,
    LiteratureMethodExecutor,
    StageBudgets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/processed/necessity_pilot_v1/sample_manifest.jsonl"
DEFAULT_GEO_PATH = ROOT / "data/raw/benchmarks/geo_bench/test.jsonl"
DEFAULT_CSEO_ROOT = ROOT / "data/raw/benchmarks/cseo_bench/data"
DEFAULT_MODEL_ROOT = ROOT / "data/raw/models"
OPTIMIZER_MANIFEST_FIELDS = (
    "sample_id",
    "benchmark",
    "domain",
    "instance_id",
    "target_document_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--geo-path", type=Path, default=DEFAULT_GEO_PATH)
    parser.add_argument("--cseo-root", type=Path, default=DEFAULT_CSEO_ROOT)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "prompts/necessity_pilot/literature_methods_v1",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--intermediate-max-new-tokens", type=int)
    parser.add_argument("--if-geo-aggregation-max-new-tokens", type=int)
    parser.add_argument("--rewrite-growth-ratio", type=float)
    parser.add_argument("--rewrite-growth-slack-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--failure-policy", choices=("stop", "continue"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json-format-repair", action="store_true",
                        help="Opt in to logged, format-only JSON repair; never complete content.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--device-map-strategy", choices=("balanced", "none"), default="balanced"
    )
    return parser.parse_args()


def load_inputs(
    manifest_path: Path,
    geo_path: Path,
    cseo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    raw_manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = []
    for row in raw_manifest:
        missing = [field for field in OPTIMIZER_MANIFEST_FIELDS if field not in row]
        if missing:
            raise KeyError(f"Manifest row is missing optimizer fields: {missing}")
        manifest.append({field: row[field] for field in OPTIMIZER_MANIFEST_FIELDS})
    source_rows = load_geo(geo_path) + load_cseo(cseo_root)
    source_map = {
        f"{row['benchmark']}__{row['domain']}__{row['instance_id']}": {
            "documents": list(row["documents"])
        }
        for row in source_rows
    }
    seen: set[str] = set()
    for item in manifest:
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen.add(sample_id)
        if sample_id not in source_map:
            raise KeyError(f"Manifest sample missing from benchmark data: {sample_id}")
        target_index = int(item["target_document_index"])
        documents = source_map[sample_id]["documents"]
        if not 0 <= target_index < len(documents):
            raise IndexError(
                f"{sample_id}: target_document_index {target_index} is out of range"
            )
        if not str(documents[target_index]).strip():
            raise ValueError(f"{sample_id}: target document is empty")
    return manifest, source_map


def validate_runtime_args(
    args: argparse.Namespace,
    catalog: LiteratureMethodCatalog,
) -> tuple[str, ...]:
    methods = tuple(args.methods or catalog.method_ids)
    for method_id in methods:
        catalog.get(method_id)
    if args.validate_only:
        return methods
    missing = []
    for field in (
        "experiment_id",
        "intermediate_max_new_tokens",
        "rewrite_growth_ratio",
        "rewrite_growth_slack_tokens",
        "output_dir",
        "failure_policy",
    ):
        if getattr(args, field) is None:
            missing.append(field)
    if args.methods is None:
        missing.append("methods")
    if missing:
        raise ValueError(
            "Formal execution requires explicit values for: " + ", ".join(missing)
        )
    return methods


def select_manifest_rows(
    manifest: Sequence[dict[str, Any]],
    sample_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if sample_ids is None:
        return list(manifest)
    requested = list(sample_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("sample_ids must be unique")
    by_id = {str(row["sample_id"]): row for row in manifest}
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise KeyError(f"Requested sample_ids are absent from manifest: {missing}")
    return [by_id[sample_id] for sample_id in requested]


class GlmChatBackend:
    """Local Transformers GLM backend, loaded only for formal GPU execution."""

    def __init__(
        self,
        *,
        model_root: Path,
        device: str,
        dtype: str,
        attn_implementation: str,
        device_map_strategy: str | None,
        max_memory: Mapping[int | str, int | str] | None = None,
        repetition_penalty: float | None = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_root = model_root
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device_map_strategy = device_map_strategy
        self.max_memory = max_memory
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.loaded: Any = None

    def __enter__(self) -> "GlmChatBackend":
        from model_loader import load_model

        self.loaded = load_model(
            "glm",
            model_root=self.model_root,
            device=self.device,
            dtype_name=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map_strategy=self.device_map_strategy,
            max_memory=self.max_memory,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.loaded is not None:
            from model_loader import release_model

            release_model(self.loaded)
            self.loaded = None

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        stage_id: str,
        max_new_tokens: int,
        seed: int,
    ) -> BackendGeneration:
        if self.loaded is None:
            raise RuntimeError("GLM backend is not loaded")
        from model_loader import generate_answer

        encoded = self.loaded.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        input_tokens = (
            int(input_ids.shape[-1])
            if hasattr(input_ids, "shape")
            else len(input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids)
        )
        context_limit = getattr(
            self.loaded.model.config, "max_position_embeddings", None
        )
        if not isinstance(context_limit, int) or context_limit <= 0:
            raise RuntimeError("GLM config lacks a valid max_position_embeddings")
        if input_tokens + max_new_tokens > context_limit:
            raise ValueError(
                f"{stage_id}: input_tokens={input_tokens} plus "
                f"max_new_tokens={max_new_tokens} exceeds context_limit={context_limit}"
            )
        generated = generate_answer(
            self.loaded,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            **({'temperature': self.temperature, 'top_p': self.top_p}
               if self.do_sample else {}),
            seed=seed,
            **({'repetition_penalty': self.repetition_penalty}
               if self.repetition_penalty is not None else {}),
        )
        if generated.input_tokens != input_tokens:
            raise RuntimeError(
                f"{stage_id}: request token count changed between validation and generation"
            )
        return BackendGeneration(
            answer=generated.answer,
            raw_answer=generated.raw_answer,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            model_id=self.loaded.spec.model_id,
        )

    def count_text_tokens(self, text: str) -> int:
        if self.loaded is None:
            raise RuntimeError("GLM backend is not loaded")
        if not text.strip():
            raise ValueError("text must not be empty")
        encoded = self.loaded.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)


def run_cpu_validation(
    *,
    catalog: LiteratureMethodCatalog,
    methods: Sequence[str],
    manifest: Sequence[dict[str, Any]],
    source_map: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    first = manifest[0]
    source = source_map[str(first["sample_id"])]
    document = str(source["documents"][int(first["target_document_index"])])
    per_method: dict[str, Any] = {}
    for method_id in methods:
        backend = DeterministicDryRunBackend()
        executor = LiteratureMethodExecutor(
            catalog=catalog,
            backend=backend,
            budgets=StageBudgets(
                intermediate_max_new_tokens=1,
                rewrite_growth_ratio=1.25,
                rewrite_growth_slack_tokens=256,
            ),
        )
        result = executor.run(document, method_id=method_id, seed=0)
        per_method[method_id] = {
            "expected_calls": catalog.get(method_id).calls_per_document,
            "observed_calls": len(result.stage_records),
            "query_visible_to_optimizer": result.query_visible_to_optimizer,
            "realized_document_nonempty": bool(result.realized_document.strip()),
        }
    return {
        "mode": "validate_only_no_model",
        "catalog_id": catalog.catalog_id,
        "manifest_count": len(manifest),
        "selected_sample_ids": [str(row["sample_id"]) for row in manifest],
        "methods": per_method,
        "writes_results": False,
        "loads_gpu_model": False,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def build_condition_error_payload(
    *,
    experiment_id: str | None,
    sample_id: str,
    method_id: str,
    executor: LiteratureMethodExecutor,
    exc: Exception,
) -> dict[str, Any]:
    """Build a lossless diagnostic without repairing a failed model output."""

    completed = executor.completed_generation_records
    last = completed[-1].to_dict() if completed else None
    return {
        "experiment_id": experiment_id,
        "sample_id": sample_id,
        "method_id": method_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "completed_generation_call_count": len(completed),
        "last_completed_stage_record": last,
        "completed_stage_records": [record.to_dict() for record in completed],
        "diagnostic_only_no_output_repair": True,
    }


def run_formal(
    *,
    args: argparse.Namespace,
    catalog: LiteratureMethodCatalog,
    methods: Sequence[str],
    manifest: Sequence[dict[str, Any]],
    source_map: Mapping[str, dict[str, Any]],
) -> int:
    assert args.output_dir is not None
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty {args.output_dir}; use --resume explicitly"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "realized_documents.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    completed: set[tuple[str, str]] = set()
    if args.resume and output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["sample_id"]), str(row["method_id"])))

    budgets = StageBudgets(
        intermediate_max_new_tokens=args.intermediate_max_new_tokens,
        if_geo_aggregation_max_new_tokens=args.if_geo_aggregation_max_new_tokens,
        rewrite_growth_ratio=args.rewrite_growth_ratio,
        rewrite_growth_slack_tokens=args.rewrite_growth_slack_tokens,
    )
    strategy = None if args.device_map_strategy == "none" else "balanced"
    new_count = 0
    error_count = 0
    with GlmChatBackend(
        model_root=args.model_root,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map_strategy=strategy,
    ) as backend:
        executor = LiteratureMethodExecutor(
            catalog=catalog,
            backend=backend,
            budgets=budgets,
            json_format_repair=getattr(args, "json_format_repair", False),
        )
        for item in manifest:
            sample_id = str(item["sample_id"])
            source = source_map[sample_id]
            target_index = int(item["target_document_index"])
            document = str(source["documents"][target_index])
            for method_id in methods:
                key = (sample_id, method_id)
                if key in completed:
                    continue
                try:
                    result = executor.run(document, method_id=method_id, seed=args.seed)
                    _append_jsonl(
                        output_path,
                        {
                            "experiment_id": args.experiment_id,
                            "sample_id": sample_id,
                            "benchmark": item["benchmark"],
                            "domain": item["domain"],
                            "instance_id": item["instance_id"],
                            "target_document_index": target_index,
                            "method_id": method_id,
                            "source_text": document,
                            "rewritten_text": result.realized_document,
                            "method_run": result.to_dict(),
                        },
                    )
                    completed.add(key)
                    new_count += 1
                except Exception as exc:
                    error_count += 1
                    _append_jsonl(
                        errors_path,
                        build_condition_error_payload(
                            experiment_id=args.experiment_id,
                            sample_id=sample_id,
                            method_id=method_id,
                            executor=executor,
                            exc=exc,
                        ),
                    )
                    if args.failure_policy == "stop":
                        raise

    metrics = {
        "experiment_id": args.experiment_id,
        "catalog_id": catalog.catalog_id,
        "manifest_count": len(manifest),
        "selected_sample_ids": [str(row["sample_id"]) for row in manifest],
        "method_ids": list(methods),
        "expected_conditions": len(manifest) * len(methods),
        "completed_conditions": len(completed),
        "new_conditions_this_run": new_count,
        "error_count_this_run": error_count,
        "query_visible_to_optimizer": False,
        "generation_seed": args.seed,
        "json_format_repair": getattr(args, "json_format_repair", False),
        "intermediate_max_new_tokens": args.intermediate_max_new_tokens,
        "if_geo_aggregation_max_new_tokens": args.if_geo_aggregation_max_new_tokens,
        "rewrite_growth_ratio": args.rewrite_growth_ratio,
        "rewrite_growth_slack_tokens": args.rewrite_growth_slack_tokens,
        "failure_policy": args.failure_policy,
        "checkpointed": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if error_count == 0 else 1


def main() -> int:
    args = parse_args()
    catalog = LiteratureMethodCatalog(args.catalog_root)
    methods = validate_runtime_args(args, catalog)
    manifest, source_map = load_inputs(args.manifest, args.geo_path, args.cseo_root)
    manifest = select_manifest_rows(manifest, args.sample_ids)
    if args.validate_only:
        print(
            json.dumps(
                run_cpu_validation(
                    catalog=catalog,
                    methods=methods,
                    manifest=manifest,
                    source_map=source_map,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return run_formal(
        args=args,
        catalog=catalog,
        methods=methods,
        manifest=manifest,
        source_map=source_map,
    )


if __name__ == "__main__":
    raise SystemExit(main())
