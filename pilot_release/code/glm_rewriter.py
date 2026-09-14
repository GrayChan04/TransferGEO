"""Reusable deterministic GLM document rewriting for TransferGEO.

This module owns the model-facing rewrite path. Experiment runners should
import :class:`GlmRewriter` instead of duplicating GLM loading, prompt
rendering, context checks, or generation logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model_loader import (
    DEFAULT_MODEL_ROOT,
    LoadedModel,
    generate_answer,
    load_model,
    release_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_CATALOG_ROOT = (
    PROJECT_ROOT / "prompts" / "necessity_pilot" / "v1"
)
DEFAULT_DOCUMENT_PLACEHOLDER = "{{DOCUMENT}}"


def sha256_text(text: str) -> str:
    """Return the UTF-8 SHA-256 digest of text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_count(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


@dataclass(frozen=True)
class PromptSpec:
    """One prompt entry loaded from a versioned prompt catalog."""

    prompt_id: str
    category_name: str
    variant_name: str
    path: Path
    shared_shell_group: str
    selection_eligible: bool
    template_sha256: str


class PromptCatalog:
    """Load and render a versioned TransferGEO prompt catalog."""

    def __init__(self, root: Path = DEFAULT_PROMPT_CATALOG_ROOT) -> None:
        self.root = Path(root).resolve()
        self.catalog_path = self.root / "prompt_catalog.json"
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Prompt catalog not found: {self.catalog_path}")

        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.catalog_id = str(catalog["catalog_id"])
        self.catalog_version = int(catalog["catalog_version"])
        self.document_placeholder = str(
            catalog.get("document_placeholder", DEFAULT_DOCUMENT_PLACEHOLDER)
        )
        self.rewriter_model_id = str(catalog["rewriter_model_id"])
        self.decoding = dict(catalog["decoding"])

        if self.document_placeholder != DEFAULT_DOCUMENT_PLACEHOLDER:
            raise ValueError(
                "Unsupported document placeholder: "
                f"{self.document_placeholder!r}"
            )
        if self.decoding.get("do_sample") is not False:
            raise ValueError("The frozen prompt catalog must use do_sample=False")
        if self.decoding.get("generations_per_document_variant") != 1:
            raise ValueError(
                "The frozen prompt catalog must use one generation per condition"
            )
        if self.decoding.get("silent_retry") is not False:
            raise ValueError("The frozen prompt catalog must prohibit silent retries")

        self._entries: dict[str, Mapping[str, Any]] = {}
        for entry in catalog["prompts"]:
            prompt_id = str(entry["prompt_id"])
            if prompt_id in self._entries:
                raise ValueError(f"Duplicate prompt_id in catalog: {prompt_id}")
            self._entries[prompt_id] = entry

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        """Prompt IDs in the frozen catalog order."""

        return tuple(self._entries)

    def get(self, prompt_id: str) -> PromptSpec:
        """Return prompt metadata and validate its executable template."""

        try:
            entry = self._entries[prompt_id]
        except KeyError as exc:
            supported = ", ".join(self.prompt_ids)
            raise KeyError(
                f"Unknown prompt_id {prompt_id!r}; choose one of: {supported}"
            ) from exc

        path = (self.root / str(entry["path"])).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Prompt path escapes catalog root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        template = path.read_text(encoding="utf-8")
        count = template.count(self.document_placeholder)
        if count != 1:
            raise ValueError(
                f"{path}: expected exactly one {self.document_placeholder}, "
                f"found {count}"
            )
        return PromptSpec(
            prompt_id=prompt_id,
            category_name=str(entry["category_name"]),
            variant_name=str(entry["variant_name"]),
            path=path,
            shared_shell_group=str(entry["shared_shell_group"]),
            selection_eligible=bool(entry["selection_eligible"]),
            template_sha256=sha256_text(template),
        )

    def read_template(self, prompt_id: str) -> str:
        """Read the exact template for one catalog prompt."""

        return self.get(prompt_id).path.read_text(encoding="utf-8")

    def render(self, prompt_id: str, document: str) -> tuple[PromptSpec, str]:
        """Substitute one document into one catalog prompt."""

        if not document.strip():
            raise ValueError("Document must not be empty")
        spec = self.get(prompt_id)
        template = spec.path.read_text(encoding="utf-8")
        rendered = template.replace(self.document_placeholder, document)
        return spec, rendered


@dataclass(frozen=True)
class RewriteResult:
    """One deterministic GLM rewrite and its complete provenance."""

    model_id: str
    prompt_id: str
    category_name: str
    variant_name: str
    prompt_path: str
    prompt_template_sha256: str
    rendered_request_sha256: str
    document_sha256: str
    output_sha256: str
    source_tokens: int
    input_tokens: int
    output_tokens: int
    max_new_tokens: int
    context_limit: int
    do_sample: bool
    seed: int
    answer: str
    raw_answer: str

    @property
    def cap_hit(self) -> bool:
        return self.output_tokens >= self.max_new_tokens

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cap_hit"] = self.cap_hit
        return payload


class GlmRewriter:
    """Reusable deterministic GLM-4 document rewriter.

    The class accepts complete rendered rewrite instructions as a single
    ``user`` message. It never adds a system prompt or a query field.
    """

    def __init__(
        self,
        *,
        model_root: Path = DEFAULT_MODEL_ROOT,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device_map_strategy: str | None = "balanced",
        max_memory: Mapping[int | str, int | str] | None = None,
    ) -> None:
        self.model_root = Path(model_root)
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device_map_strategy = device_map_strategy
        self.max_memory = max_memory
        self.loaded: LoadedModel | None = None

    def load(self) -> "GlmRewriter":
        """Load the frozen local GLM checkpoint once."""

        if self.loaded is not None:
            return self
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

    def close(self) -> None:
        """Release model references and CUDA cache."""

        if self.loaded is not None:
            release_model(self.loaded)
            self.loaded = None

    def __enter__(self) -> "GlmRewriter":
        return self.load()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_loaded(self) -> LoadedModel:
        if self.loaded is None:
            raise RuntimeError("GlmRewriter is not loaded; call load() first")
        return self.loaded

    @property
    def context_limit(self) -> int:
        loaded = self._require_loaded()
        value = getattr(loaded.model.config, "max_position_embeddings", None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "GLM config does not expose a valid max_position_embeddings"
            )
        return value

    def count_document_tokens(self, document: str) -> int:
        """Count source-document tokens without chat-template overhead."""

        loaded = self._require_loaded()
        encoded = loaded.tokenizer(
            document,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return _token_count(encoded["input_ids"])

    def count_request_tokens(self, rendered_prompt: str) -> int:
        """Count a single-user-message request under the GLM chat template."""

        loaded = self._require_loaded()
        encoded = loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        return _token_count(encoded["input_ids"])

    @staticmethod
    def dynamic_output_cap(
        source_tokens: int,
        *,
        growth_ratio: float = 1.25,
        growth_slack_tokens: int = 256,
    ) -> int:
        """Allow controlled growth while bounding runaway rewrites."""

        if source_tokens <= 0:
            raise ValueError("source_tokens must be positive")
        if growth_ratio < 1.0:
            raise ValueError("growth_ratio must be at least 1.0")
        if growth_slack_tokens < 0:
            raise ValueError("growth_slack_tokens must be non-negative")
        return math.ceil(source_tokens * growth_ratio) + growth_slack_tokens

    def rewrite(
        self,
        document: str,
        *,
        prompt_id: str,
        catalog: PromptCatalog,
        max_new_tokens: int | None = None,
        growth_ratio: float = 1.25,
        growth_slack_tokens: int = 256,
        seed: int = 0,
    ) -> RewriteResult:
        """Rewrite one document with one frozen catalog prompt."""

        spec, rendered_prompt = catalog.render(prompt_id, document)
        return self._rewrite_rendered_prompt(
            document,
            rendered_prompt=rendered_prompt,
            prompt_id=spec.prompt_id,
            category_name=spec.category_name,
            variant_name=spec.variant_name,
            prompt_path=str(spec.path),
            prompt_template_sha256=spec.template_sha256,
            max_new_tokens=max_new_tokens,
            growth_ratio=growth_ratio,
            growth_slack_tokens=growth_slack_tokens,
            seed=seed,
        )

    def rewrite_with_template(
        self,
        document: str,
        *,
        prompt_template: str,
        prompt_id: str = "custom_prompt",
        category_name: str = "Custom Prompt",
        variant_name: str = "Custom Prompt",
        prompt_path: str = "",
        document_placeholder: str = DEFAULT_DOCUMENT_PLACEHOLDER,
        max_new_tokens: int | None = None,
        growth_ratio: float = 1.25,
        growth_slack_tokens: int = 256,
        seed: int = 0,
    ) -> RewriteResult:
        """Rewrite with an arbitrary template for future adapted prompts.

        The template must contain exactly one ``{{DOCUMENT}}`` placeholder by
        default. This keeps the same query-invisible runtime contract while
        allowing later prompt-transfer methods to supply adapted templates that
        are not part of the frozen necessity-pilot catalog.
        """

        if not document.strip():
            raise ValueError("Document must not be empty")
        count = prompt_template.count(document_placeholder)
        if count != 1:
            raise ValueError(
                f"Expected exactly one {document_placeholder}, found {count}"
            )
        rendered_prompt = prompt_template.replace(document_placeholder, document)
        return self._rewrite_rendered_prompt(
            document,
            rendered_prompt=rendered_prompt,
            prompt_id=prompt_id,
            category_name=category_name,
            variant_name=variant_name,
            prompt_path=prompt_path,
            prompt_template_sha256=sha256_text(prompt_template),
            max_new_tokens=max_new_tokens,
            growth_ratio=growth_ratio,
            growth_slack_tokens=growth_slack_tokens,
            seed=seed,
        )

    def _rewrite_rendered_prompt(
        self,
        document: str,
        *,
        rendered_prompt: str,
        prompt_id: str,
        category_name: str,
        variant_name: str,
        prompt_path: str,
        prompt_template_sha256: str,
        max_new_tokens: int | None,
        growth_ratio: float,
        growth_slack_tokens: int,
        seed: int,
    ) -> RewriteResult:
        """Generate from a fully rendered single-user-message rewrite prompt."""

        loaded = self._require_loaded()
        source_tokens = self.count_document_tokens(document)
        input_tokens = self.count_request_tokens(rendered_prompt)
        resolved_max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else self.dynamic_output_cap(
                source_tokens,
                growth_ratio=growth_ratio,
                growth_slack_tokens=growth_slack_tokens,
            )
        )
        if resolved_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if input_tokens + resolved_max_new_tokens > self.context_limit:
            raise ValueError(
                "Rewrite would exceed the GLM context window: "
                f"input_tokens={input_tokens}, "
                f"max_new_tokens={resolved_max_new_tokens}, "
                f"context_limit={self.context_limit}"
            )

        generation = generate_answer(
            loaded,
            [{"role": "user", "content": rendered_prompt}],
            max_new_tokens=resolved_max_new_tokens,
            do_sample=False,
            seed=seed,
        )
        if generation.input_tokens != input_tokens:
            raise RuntimeError(
                "Request token count changed between validation and generation: "
                f"{input_tokens} != {generation.input_tokens}"
            )

        return RewriteResult(
            model_id=loaded.spec.model_id,
            prompt_id=prompt_id,
            category_name=category_name,
            variant_name=variant_name,
            prompt_path=prompt_path,
            prompt_template_sha256=prompt_template_sha256,
            rendered_request_sha256=sha256_text(rendered_prompt),
            document_sha256=sha256_text(document),
            output_sha256=sha256_text(generation.answer),
            source_tokens=source_tokens,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            max_new_tokens=resolved_max_new_tokens,
            context_limit=self.context_limit,
            do_sample=False,
            seed=seed,
            answer=generation.answer,
            raw_answer=generation.raw_answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite one document with the reusable TransferGEO GLM path."
    )
    parser.add_argument("--document-file", type=Path, required=True)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt-id")
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="Custom template containing exactly one {{DOCUMENT}} placeholder.",
    )
    parser.add_argument(
        "--prompt-catalog-root",
        type=Path,
        default=DEFAULT_PROMPT_CATALOG_ROOT,
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--device-map-strategy",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--growth-ratio", type=float, default=1.25)
    parser.add_argument("--growth-slack-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = args.document_file.read_text(encoding="utf-8")
    strategy = (
        None if args.device_map_strategy == "none" else args.device_map_strategy
    )
    with GlmRewriter(
        model_root=args.model_root,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map_strategy=strategy,
    ) as rewriter:
        if args.prompt_file is None:
            catalog = PromptCatalog(args.prompt_catalog_root)
            result = rewriter.rewrite(
                document,
                prompt_id=args.prompt_id,
                catalog=catalog,
                max_new_tokens=args.max_new_tokens,
                growth_ratio=args.growth_ratio,
                growth_slack_tokens=args.growth_slack_tokens,
                seed=args.seed,
            )
        else:
            prompt_template = args.prompt_file.read_text(encoding="utf-8")
            result = rewriter.rewrite_with_template(
                document,
                prompt_template=prompt_template,
                prompt_id=args.prompt_file.stem,
                category_name="Custom Prompt",
                variant_name=args.prompt_file.stem,
                prompt_path=str(args.prompt_file.resolve()),
                max_new_tokens=args.max_new_tokens,
                growth_ratio=args.growth_ratio,
                growth_slack_tokens=args.growth_slack_tokens,
                seed=args.seed,
            )

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(result.answer + "\n", encoding="utf-8")
    if args.metadata_file is not None:
        args.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_file.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.output_file is None:
        print(result.answer)
    summary = result.to_dict()
    summary.pop("answer")
    summary.pop("raw_answer")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
