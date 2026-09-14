"""Query-invisible execution core for literature-derived GEO methods.

The module is deliberately independent from the frozen custom Prompt Catalog
v12 path.  It owns prompt validation, deterministic stage serialization and
method-specific post-processing, while the model-facing backend is injected.
This makes the complete orchestration testable without loading a GPU model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_ROOT = (
    PROJECT_ROOT / "prompts" / "necessity_pilot" / "literature_methods_v1"
)
DOCUMENT_PLACEHOLDER = "{{DOCUMENT}}"
RUNTIME_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
FORBIDDEN_BENCHMARK_QUERY_PLACEHOLDERS = (
    "{{QUERY}}",
    "{{BENCHMARK_QUERY}}",
    "{{QUERY_KEYWORDS}}",
)
SINGLE_JSON_FENCE_PATTERN = re.compile(
    r"\A```json[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class BackendGeneration:
    """One backend call returned to the method executor."""

    answer: str
    raw_answer: str
    input_tokens: int
    output_tokens: int
    model_id: str


class GenerationBackend(Protocol):
    """Minimal model-facing interface used by every literature method."""

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        stage_id: str,
        max_new_tokens: int,
        seed: int,
    ) -> BackendGeneration: ...

    def count_text_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class StageBudgets:
    """Explicit generation limits; formal values must be frozen before GPU use."""

    intermediate_max_new_tokens: int
    rewrite_growth_ratio: float
    rewrite_growth_slack_tokens: int
    if_geo_aggregation_max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.if_geo_aggregation_max_new_tokens is not None and self.if_geo_aggregation_max_new_tokens <= 0:
            raise ValueError("if_geo_aggregation_max_new_tokens must be positive")
        if self.intermediate_max_new_tokens <= 0:
            raise ValueError("intermediate_max_new_tokens must be positive")
        if self.rewrite_growth_ratio < 1.0:
            raise ValueError("rewrite_growth_ratio must be at least 1.0")
        if self.rewrite_growth_slack_tokens < 0:
            raise ValueError("rewrite_growth_slack_tokens must be non-negative")

    def rewrite_max_new_tokens(self, source_tokens: int) -> int:
        if source_tokens <= 0:
            raise ValueError("source_tokens must be positive")
        return (
            math.ceil(source_tokens * self.rewrite_growth_ratio)
            + self.rewrite_growth_slack_tokens
        )


@dataclass(frozen=True)
class StageRecord:
    stage_id: str
    messages_sha256: str
    input_tokens: int
    output_tokens: int
    max_new_tokens: int
    model_id: str
    output_sha256: str
    output: str
    raw_output: str
    serialization_normalized: bool = False
    format_repair: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodRunResult:
    method_id: str
    comparison_tier: str
    source_document_sha256: str
    source_document_tokens: int
    rewrite_max_new_tokens: int
    realized_document_sha256: str
    realized_document: str
    generation_seed: int
    query_visible_to_optimizer: bool
    stage_records: tuple[StageRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage_records"] = [record.to_dict() for record in self.stage_records]
        return payload


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    comparison_tier: str
    calls_per_document: int
    payload: dict[str, Any]


class LiteratureMethodCatalog:
    """Validate and expose the versioned literature-method catalog."""

    def __init__(self, root: Path = DEFAULT_CATALOG_ROOT) -> None:
        self.root = Path(root).resolve()
        self.catalog_path = self.root / "method_catalog.json"
        self.hash_path = self.root / "prompt_sha256.txt"
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Method catalog not found: {self.catalog_path}")
        if not self.hash_path.is_file():
            raise FileNotFoundError(f"Prompt hash manifest not found: {self.hash_path}")

        self.payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if self.payload.get("query_visible_to_optimizer") is not False:
            raise ValueError("Literature method catalog must be query-invisible")
        self.catalog_id = str(self.payload["catalog_id"])
        self.catalog_version = int(self.payload["catalog_version"])
        self.rewriter_model_id = str(self.payload["rewriter_model_id"])

        self._methods: dict[str, MethodSpec] = {}
        for entry in self.payload["methods"]:
            method_id = str(entry["method_id"])
            if method_id in self._methods:
                raise ValueError(f"Duplicate method_id: {method_id}")
            calls = int(entry["calls_per_document"])
            if calls <= 0:
                raise ValueError(f"{method_id}: calls_per_document must be positive")
            self._methods[method_id] = MethodSpec(
                method_id=method_id,
                comparison_tier=str(entry["comparison_tier"]),
                calls_per_document=calls,
                payload=dict(entry),
            )
        self._validate_prompt_hashes()
        self._validate_method_paths()

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(self._methods)

    def get(self, method_id: str) -> MethodSpec:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown method_id {method_id!r}; choose one of: "
                + ", ".join(self.method_ids)
            ) from exc

    def resolve_prompt_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Prompt path escapes catalog root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path

    def read_prompt(self, relative_path: str) -> str:
        return self.resolve_prompt_path(relative_path).read_text(encoding="utf-8")

    def _read_hash_manifest(self) -> dict[str, str]:
        observed: dict[str, str] = {}
        for line_number, line in enumerate(
            self.hash_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid prompt hash line {line_number}: {line!r}"
                )
            digest, relative_path = parts
            if relative_path in observed:
                raise ValueError(f"Duplicate prompt hash path: {relative_path}")
            observed[relative_path] = digest
        return observed

    def _validate_prompt_hashes(self) -> None:
        expected = self._read_hash_manifest()
        actual_paths = sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*.txt")
            if path.resolve() != self.hash_path.resolve()
        )
        if sorted(expected) != actual_paths:
            missing = sorted(set(actual_paths) - set(expected))
            extra = sorted(set(expected) - set(actual_paths))
            raise ValueError(
                f"Prompt hash manifest mismatch; missing={missing}, extra={extra}"
            )
        for relative_path, expected_digest in expected.items():
            text = self.read_prompt(relative_path)
            actual_digest = sha256_text(text)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"Prompt hash mismatch for {relative_path}: "
                    f"{actual_digest} != {expected_digest}"
                )
            for placeholder in FORBIDDEN_BENCHMARK_QUERY_PLACEHOLDERS:
                if placeholder in text:
                    raise ValueError(
                        f"{relative_path}: forbidden benchmark-query placeholder "
                        f"{placeholder}"
                    )

    def _validate_method_paths(self) -> None:
        path_fields = (
            "system_prompt_path",
            "user_prompt_path",
        )
        list_path_fields = (
            "stage_prompt_paths",
            "system_prompt_paths",
            "stage_user_template_paths",
        )
        for spec in self._methods.values():
            for field in path_fields:
                value = spec.payload.get(field)
                if value is not None:
                    self.resolve_prompt_path(str(value))
            for field in list_path_fields:
                for value in spec.payload.get(field, []):
                    self.resolve_prompt_path(str(value))


def render_template(template: str, **values: str) -> str:
    """Render explicit ``{{NAME}}`` fields and reject unresolved runtime data."""

    rendered = template
    for name, value in values.items():
        placeholder = "{{" + name + "}}"
        if rendered.count(placeholder) != 1:
            raise ValueError(
                f"Expected exactly one {placeholder}, found {rendered.count(placeholder)}"
            )
        if not str(value).strip():
            raise ValueError(f"Runtime value {name} must not be empty")
        rendered = rendered.replace(placeholder, str(value))
    unresolved = RUNTIME_PLACEHOLDER_PATTERN.findall(rendered)
    if unresolved:
        raise ValueError(f"Unresolved runtime placeholders: {sorted(set(unresolved))}")
    return rendered


def build_messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be empty")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty when provided")
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _extract_last_fenced_block_or_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model returned an empty output")
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if blocks and blocks[-1].strip():
        return blocks[-1].strip()
    return stripped


def _strip_output_label(text: str, labels: Sequence[str]) -> str:
    value = _extract_last_fenced_block_or_text(text)
    for label in labels:
        match = re.match(rf"^\s*{re.escape(label)}\s*:\s*", value, flags=re.IGNORECASE)
        if match:
            value = value[match.end() :].strip()
            break
    if not value:
        raise ValueError("Output became empty after removing its protocol label")
    return value


def parse_json_strict_or_single_json_fence(
    text: str, *, stage_id: str
) -> tuple[Any, bool]:
    """Parse bare JSON or one exact outer ``json`` fence.

    Whitespace outside the payload is ignored.  Prose, unlabeled fences,
    multiple blocks and substring extraction remain unsupported.  The boolean
    records whether the unique outer serialization wrapper was removed.
    """

    stripped = text.strip()
    match = SINGLE_JSON_FENCE_PATTERN.fullmatch(stripped)
    candidate = match.group("body") if match is not None else stripped
    serialization_normalized = match is not None
    try:
        return json.loads(candidate), serialization_normalized
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{stage_id}: expected bare JSON or one json fenced block with no "
            "surrounding text; no semantic repair or retry is allowed"
        ) from exc


class LiteratureMethodExecutor:
    """Run one literature method without access to the benchmark query."""

    def __init__(
        self,
        *,
        catalog: LiteratureMethodCatalog,
        backend: GenerationBackend,
        budgets: StageBudgets,
        json_format_repair: bool = False,
    ) -> None:
        self.catalog = catalog
        self.backend = backend
        self.budgets = budgets
        self.json_format_repair = json_format_repair
        self._stage_records: list[StageRecord] = []
        self._rewrite_max_new_tokens: int | None = None

    @property
    def completed_generation_records(self) -> tuple[StageRecord, ...]:
        """Expose generated stage outputs for failure diagnostics.

        A record means the model call completed.  It does not imply that the
        returned text passed the method-specific parser or schema checks.
        """

        return tuple(self._stage_records)

    def _call(
        self,
        *,
        stage_id: str,
        system_prompt: str | None,
        user_prompt: str,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        messages = build_messages(system_prompt, user_prompt)
        serialized = canonical_json(messages)
        generation = self.backend.generate(
            messages,
            stage_id=stage_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if not generation.answer.strip():
            raise ValueError(f"{stage_id}: backend returned an empty answer")
        self._stage_records.append(
            StageRecord(
                stage_id=stage_id,
                messages_sha256=sha256_text(serialized),
                input_tokens=int(generation.input_tokens),
                output_tokens=int(generation.output_tokens),
                max_new_tokens=max_new_tokens,
                model_id=generation.model_id,
                output_sha256=sha256_text(generation.answer),
                output=generation.answer,
                raw_output=generation.raw_answer,
                serialization_normalized=False,
            )
        )
        return generation.answer

    def _parse_json_stage(self, text: str, *, stage_id: str) -> Any:
        repair_audit = None
        try:
            payload, normalized = parse_json_strict_or_single_json_fence(
                text, stage_id=stage_id
            )
        except ValueError:
            if not self.json_format_repair:
                raise
            from json_format_repair import parse_json_format_only
            payload, repair_audit = parse_json_format_only(text)
            normalized = True
        if not self._stage_records or self._stage_records[-1].stage_id != stage_id:
            raise RuntimeError(
                f"{stage_id}: JSON parsing is not aligned with the latest model call"
            )
        if normalized:
            self._stage_records[-1] = replace(
                self._stage_records[-1], serialization_normalized=True,
                format_repair=repair_audit,
            )
        return payload

    def run(self, document: str, *, method_id: str, seed: int = 0) -> MethodRunResult:
        """Execute a method using only the document and generated intermediates."""

        if not document.strip():
            raise ValueError("document must not be empty")
        spec = self.catalog.get(method_id)
        self._stage_records = []
        source_tokens = self.backend.count_text_tokens(document)
        self._rewrite_max_new_tokens = self.budgets.rewrite_max_new_tokens(
            source_tokens
        )
        if method_id.startswith("geo.") or method_id.startswith("cseo."):
            realized = self._run_fixed(document, spec=spec, seed=seed)
        elif method_id == "raid_gseo.full_pipeline":
            realized = self._run_raid(document, spec=spec, seed=seed)
        elif method_id == "if_geo.full_pipeline":
            realized = self._run_if_geo(document, spec=spec, seed=seed)
        else:
            raise NotImplementedError(f"No executor is registered for {method_id}")
        if len(self._stage_records) != spec.calls_per_document:
            raise RuntimeError(
                f"{method_id}: expected {spec.calls_per_document} calls, "
                f"observed {len(self._stage_records)}"
            )
        return MethodRunResult(
            method_id=method_id,
            comparison_tier=spec.comparison_tier,
            source_document_sha256=sha256_text(document),
            source_document_tokens=source_tokens,
            rewrite_max_new_tokens=self._require_rewrite_max_new_tokens(),
            realized_document_sha256=sha256_text(realized),
            realized_document=realized,
            generation_seed=seed,
            query_visible_to_optimizer=False,
            stage_records=tuple(self._stage_records),
        )

    def _require_rewrite_max_new_tokens(self) -> int:
        if self._rewrite_max_new_tokens is None:
            raise RuntimeError("rewrite output budget has not been initialized")
        return self._rewrite_max_new_tokens

    def _run_fixed(self, document: str, *, spec: MethodSpec, seed: int) -> str:
        system_path = spec.payload.get("system_prompt_path")
        system_prompt = (
            self.catalog.read_prompt(str(system_path))
            if system_path is not None
            else None
        )
        user_template = self.catalog.read_prompt(str(spec.payload["user_prompt_path"]))
        user_prompt = render_template(user_template, DOCUMENT=document)
        output = self._call(
            stage_id=f"{spec.method_id}.rewrite",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self._require_rewrite_max_new_tokens(),
            seed=seed,
        )
        cleaned = _extract_last_fenced_block_or_text(output)
        if spec.method_id == "cseo.llm_guidance":
            return cleaned + "\n\n" + document
        return _strip_output_label(cleaned, ("Updated Output", "Optimized Source"))

    def _run_raid(self, document: str, *, spec: MethodSpec, seed: int) -> str:
        system_prompt = self.catalog.read_prompt(str(spec.payload["system_prompt_path"]))
        paths = [str(path) for path in spec.payload["stage_prompt_paths"]]
        summary_prompt = render_template(
            self.catalog.read_prompt(paths[0]), DOCUMENT=document
        )
        summary_raw = self._call(
            stage_id="raid_gseo.summary",
            system_prompt=system_prompt,
            user_prompt=summary_prompt,
            max_new_tokens=self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        summary = _strip_output_label(summary_raw, ("Summary",))

        intent_prompt = render_template(
            self.catalog.read_prompt(paths[1]), DOCUMENT=document, SUMMARY=summary
        )
        intent_raw = self._call(
            stage_id="raid_gseo.generalized_intent",
            system_prompt=system_prompt,
            user_prompt=intent_prompt,
            max_new_tokens=self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        intent = _strip_output_label(intent_raw, ("Generalized Intent",))

        rewrite_prompt = render_template(
            self.catalog.read_prompt(paths[2]),
            DOCUMENT=document,
            SUMMARY=summary,
            GENERALIZED_INTENT=intent,
        )
        rewrite_raw = self._call(
            stage_id="raid_gseo.rewrite",
            system_prompt=system_prompt,
            user_prompt=rewrite_prompt,
            max_new_tokens=self._require_rewrite_max_new_tokens(),
            seed=seed,
        )
        return _strip_output_label(rewrite_raw, ("Optimized Source",))

    def _run_if_geo(self, document: str, *, spec: MethodSpec, seed: int) -> str:
        system_paths = [str(path) for path in spec.payload["system_prompt_paths"]]
        user_paths = [
            str(path) for path in spec.payload["stage_user_template_paths"]
        ]
        if len(system_paths) != 6 or len(user_paths) != 6:
            raise ValueError("IF-GEO requires six aligned system/user stage templates")
        params = spec.payload["fixed_parameters_from_paper"]
        num_queries = int(params["num_queries"])
        suggestions_num = int(params["suggestions_num"])

        query_system = self.catalog.read_prompt(system_paths[0]).replace(
            "{num_queries}", str(num_queries)
        )
        query_user = render_template(
            self.catalog.read_prompt(user_paths[0]), DOCUMENT=document
        )
        query_raw = self._call(
            stage_id="if_geo.query_mining",
            system_prompt=query_system,
            user_prompt=query_user,
            max_new_tokens=self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        query_payload = self._parse_json_stage(
            query_raw, stage_id="if_geo.query_mining"
        )
        queries = self._validate_queries(query_payload, expected=num_queries)

        grouped: list[dict[str, Any]] = []
        request_system = self.catalog.read_prompt(system_paths[1]).replace(
            "{suggestions_num}", str(suggestions_num)
        )
        for query_index, query in enumerate(queries):
            request_user = render_template(
                self.catalog.read_prompt(user_paths[1]),
                LATENT_QUERY=query["query"],
                DOCUMENT=document,
            )
            request_raw = self._call(
                stage_id=f"if_geo.query_request_{query_index + 1}",
                system_prompt=request_system,
                user_prompt=request_user,
                max_new_tokens=self.budgets.intermediate_max_new_tokens,
                seed=seed,
            )
            request_payload = self._parse_json_stage(
                request_raw,
                stage_id=f"if_geo.query_request_{query_index + 1}",
            )
            suggestions = self._validate_suggestions(
                request_payload, maximum=suggestions_num
            )
            grouped.append(
                {
                    "query": query["query"],
                    "probability": query["probability"],
                    "suggestions": [
                        item for item in suggestions if item["necessity"] >= 60
                    ],
                }
            )

        dedup_user = render_template(
            self.catalog.read_prompt(user_paths[2]),
            DOCUMENT=document,
            GROUPED_SUGGESTIONS_JSON=json.dumps(
                grouped, ensure_ascii=False, indent=2
            ),
        )
        dedup_raw = self._call(
            stage_id="if_geo.prioritization_deduplication",
            system_prompt=self.catalog.read_prompt(system_paths[2]),
            user_prompt=dedup_user,
            max_new_tokens=self.budgets.if_geo_aggregation_max_new_tokens or self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        deduplicated = self._parse_json_stage(
            dedup_raw, stage_id="if_geo.prioritization_deduplication"
        )
        if not isinstance(deduplicated, list):
            raise ValueError("IF-GEO deduplication output must be a JSON list")

        conflict_user = render_template(
            self.catalog.read_prompt(user_paths[3]),
            DOCUMENT=document,
            DEDUPLICATED_SUGGESTIONS_JSON=json.dumps(
                deduplicated, ensure_ascii=False, indent=2
            ),
        )
        conflict_raw = self._call(
            stage_id="if_geo.conflict_resolution",
            system_prompt=self.catalog.read_prompt(system_paths[3]),
            user_prompt=conflict_user,
            max_new_tokens=self.budgets.if_geo_aggregation_max_new_tokens or self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        resolved = self._parse_json_stage(
            conflict_raw, stage_id="if_geo.conflict_resolution"
        )
        if not isinstance(resolved, list):
            raise ValueError("IF-GEO conflict-resolution output must be a JSON list")

        blueprint_user = render_template(
            self.catalog.read_prompt(user_paths[4]),
            DOCUMENT=document,
            RESOLVED_SUGGESTIONS_JSON=json.dumps(
                resolved, ensure_ascii=False, indent=2
            ),
        )
        blueprint_raw = self._call(
            stage_id="if_geo.blueprint_construction",
            system_prompt=self.catalog.read_prompt(system_paths[4]),
            user_prompt=blueprint_user,
            max_new_tokens=self.budgets.if_geo_aggregation_max_new_tokens or self.budgets.intermediate_max_new_tokens,
            seed=seed,
        )
        blueprint = self._parse_json_stage(
            blueprint_raw, stage_id="if_geo.blueprint_construction"
        )
        if not isinstance(blueprint, dict) or not isinstance(
            blueprint.get("revision_blueprint"), list
        ):
            raise ValueError(
                "IF-GEO blueprint output must contain a revision_blueprint list"
            )

        revision_user = render_template(
            self.catalog.read_prompt(user_paths[5]),
            DOCUMENT=document,
            REVISION_BLUEPRINT_JSON=json.dumps(
                blueprint, ensure_ascii=False, indent=2
            ),
        )
        revision_raw = self._call(
            stage_id="if_geo.blueprint_guided_revision",
            system_prompt=self.catalog.read_prompt(system_paths[5]),
            user_prompt=revision_user,
            max_new_tokens=self._require_rewrite_max_new_tokens(),
            seed=seed,
        )
        return _extract_last_fenced_block_or_text(revision_raw)

    @staticmethod
    def _validate_queries(payload: Any, *, expected: int) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
            raise ValueError("IF-GEO query mining must return an object with queries")
        queries = payload["queries"]
        if len(queries) != expected:
            raise ValueError(
                f"IF-GEO query mining must return exactly {expected} queries"
            )
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in queries:
            if not isinstance(item, dict):
                raise ValueError("Every IF-GEO query must be a JSON object")
            query = str(item.get("query", "")).strip()
            probability = item.get("probability")
            if not query or query in seen:
                raise ValueError("IF-GEO queries must be non-empty and unique")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 100:
                raise ValueError("IF-GEO query probability must be between 0 and 100")
            seen.add(query)
            normalized.append({"query": query, "probability": probability})
        return normalized

    @staticmethod
    def _validate_suggestions(payload: Any, *, maximum: int) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(
            payload.get("suggestions"), list
        ):
            raise ValueError(
                "IF-GEO request generation must return an object with suggestions"
            )
        suggestions = payload["suggestions"]
        if len(suggestions) > maximum:
            raise ValueError(
                f"IF-GEO request generation returned more than {maximum} suggestions"
            )
        normalized: list[dict[str, Any]] = []
        for item in suggestions:
            if not isinstance(item, dict):
                raise ValueError("Every IF-GEO suggestion must be a JSON object")
            excerpt = str(item.get("excerpt", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            necessity = item.get("necessity")
            if not excerpt or not suggestion:
                raise ValueError("IF-GEO suggestions require excerpt and suggestion")
            if not isinstance(necessity, (int, float)) or not 0 <= necessity <= 100:
                raise ValueError("IF-GEO necessity must be between 0 and 100")
            normalized.append(
                {
                    "excerpt": excerpt,
                    "suggestion": suggestion,
                    "necessity": necessity,
                }
            )
        return normalized


class DeterministicDryRunBackend:
    """Canned backend used only to validate all orchestration paths without GPU."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[dict[str, str]]]] = []

    def count_text_tokens(self, text: str) -> int:
        if not text.strip():
            raise ValueError("text must not be empty")
        return max(1, len(text.split()))

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        stage_id: str,
        max_new_tokens: int,
        seed: int,
    ) -> BackendGeneration:
        self.calls.append((stage_id, list(messages)))
        if stage_id == "if_geo.query_mining":
            answer = json.dumps(
                {
                    "queries": [
                        {"query": f"latent query {index}", "probability": 90 - index}
                        for index in range(1, 6)
                    ]
                }
            )
        elif stage_id.startswith("if_geo.query_request_"):
            answer = json.dumps(
                {
                    "suggestions": [
                        {
                            "excerpt": "Source sentence",
                            "suggestion": "Clarify the source sentence",
                            "necessity": 80,
                        },
                        {
                            "excerpt": "Minor phrase",
                            "suggestion": "Polish the minor phrase",
                            "necessity": 40,
                        },
                    ]
                }
            )
        elif stage_id == "if_geo.prioritization_deduplication":
            answer = json.dumps(
                [
                    {
                        "id": "suggest_1",
                        "topic": "clarity",
                        "excerpt": "Source sentence",
                        "suggestion": "Clarify the source sentence",
                        "necessity": 80,
                    }
                ]
            )
        elif stage_id == "if_geo.conflict_resolution":
            answer = json.dumps(
                [
                    {
                        "id": "suggest_1",
                        "excerpt": "Source sentence",
                        "suggestion": "Clarify the source sentence",
                    }
                ]
            )
        elif stage_id == "if_geo.blueprint_construction":
            answer = json.dumps(
                {
                    "revision_blueprint": [
                        {
                            "section_name": "Body",
                            "target_location": "Source sentence",
                            "modification_intent": "clarity",
                            "directives": ["Clarify the source sentence"],
                            "format_note": "Keep as text paragraphs.",
                        }
                    ]
                }
            )
        elif stage_id == "raid_gseo.summary":
            answer = "Summary: dry-run summary"
        elif stage_id == "raid_gseo.generalized_intent":
            answer = "Generalized Intent: dry-run intent"
        elif stage_id == "raid_gseo.rewrite":
            answer = "Optimized Source: dry-run RAID document"
        elif stage_id.endswith(".rewrite"):
            answer = "dry-run fixed-method output"
        else:
            answer = "dry-run IF-GEO document"
        return BackendGeneration(
            answer=answer,
            raw_answer=answer,
            input_tokens=0,
            output_tokens=0,
            model_id="dry-run/no-model",
        )
