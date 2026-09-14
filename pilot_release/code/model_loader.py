"""Unified local Transformers loading and generation for TransferGEO models."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "data" / "raw" / "models"

ModelFamily = Literal["qwen", "llama", "mistral", "glm"]
ModelTask = Literal["causal_lm", "image_text_to_text"]


@dataclass(frozen=True)
class ModelSpec:
    """Static configuration for one frozen local model."""

    key: ModelFamily
    model_id: str
    local_dir: str
    task: ModelTask
    disable_thinking: bool = False

    def path(self, model_root: Path = DEFAULT_MODEL_ROOT) -> Path:
        return model_root / self.local_dir


MODEL_SPECS: dict[ModelFamily, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        model_id="Qwen/Qwen3.5-9B",
        local_dir="Qwen3.5-9B",
        task="image_text_to_text",
        disable_thinking=True,
    ),
    "llama": ModelSpec(
        key="llama",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir="Meta-Llama-3.1-8B-Instruct",
        task="causal_lm",
    ),
    "mistral": ModelSpec(
        key="mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        local_dir="Mistral-7B-Instruct-v0.3",
        task="causal_lm",
    ),
}

REWRITER_MODEL_SPECS: dict[ModelFamily, ModelSpec] = {
    "glm": ModelSpec(
        key="glm",
        model_id="ZhipuAI/GLM-4-9B-0414",
        local_dir="GLM-4-9B-0414",
        task="causal_lm",
    ),
}

ALL_MODEL_SPECS: dict[ModelFamily, ModelSpec] = {
    **MODEL_SPECS,
    **REWRITER_MODEL_SPECS,
}

MODEL_ALIASES: dict[str, ModelFamily] = {
    "qwen": "qwen",
    "qwen3.5": "qwen",
    "qwen3.5-9b": "qwen",
    "llama": "llama",
    "llama3.1": "llama",
    "llama-3.1-8b-instruct": "llama",
    "mistral": "mistral",
    "mistral-7b-instruct-v0.3": "mistral",
    "glm": "glm",
    "glm4": "glm",
    "glm-4-9b": "glm",
    "glm-4-9b-0414": "glm",
}


@dataclass
class LoadedModel:
    """A tokenizer/model pair loaded from a frozen local checkpoint."""

    spec: ModelSpec
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    device_map: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Decoded answer and token accounting from one generation."""

    answer: str
    raw_answer: str
    input_tokens: int
    output_tokens: int


def get_model_spec(model_name: str) -> ModelSpec:
    """Resolve a short model name or alias to a frozen model specification."""

    normalized = model_name.strip().lower()
    try:
        return ALL_MODEL_SPECS[MODEL_ALIASES[normalized]]
    except KeyError as exc:
        supported = ", ".join(ALL_MODEL_SPECS)
        raise ValueError(f"Unknown model {model_name!r}; choose one of: {supported}") from exc


def validate_model_path(spec: ModelSpec, model_root: Path = DEFAULT_MODEL_ROOT) -> Path:
    """Check that the minimum local checkpoint files exist."""

    model_path = spec.path(model_root)
    required = (model_path / "config.json", model_path / "tokenizer_config.json")
    missing = [path for path in required if not path.is_file()]
    has_weights = (model_path / "model.safetensors").is_file() or (
        model_path / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append(model_path / "model.safetensors[.index.json]")
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Incomplete local checkpoint for {spec.key}: {formatted}")
    return model_path


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a CLI dtype name to a torch dtype."""

    normalized = dtype_name.strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _validate_device(device: str, dtype: torch.dtype) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {resolved}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested, but the visible CUDA device does not support BF16")
    return resolved


def load_model(
    model_name: str,
    *,
    model_root: Path = DEFAULT_MODEL_ROOT,
    device: str = "cuda:0",
    dtype_name: str = "bfloat16",
    attn_implementation: str = "sdpa",
    device_map_strategy: str | None = None,
    max_memory: Mapping[int | str, int | str] | None = None,
) -> LoadedModel:
    """Load one frozen checkpoint and its fast tokenizer without network access.

    ``device_map_strategy=None`` preserves the original whole-model-on-one-device
    behavior. A Transformers/Accelerate strategy such as ``"balanced"`` enables
    explicit model sharding across the CUDA devices visible to the process.
    """

    spec = get_model_spec(model_name)
    model_path = validate_model_path(spec, model_root)
    dtype = resolve_dtype(dtype_name)
    if device_map_strategy is None:
        resolved_device = _validate_device(device, dtype)
    else:
        if device_map_strategy != "balanced":
            raise ValueError(
                "Only the explicitly validated sharding strategy 'balanced' is supported"
            )
        resolved_device = _validate_device("cuda:0", dtype)
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "Balanced sharding requires at least two visible CUDA devices"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_class = (
        AutoModelForImageTextToText
        if spec.task == "image_text_to_text"
        else AutoModelForCausalLM
    )
    loader_device_map: str | dict[str, str]
    if device_map_strategy is None:
        loader_device_map = {"": str(resolved_device)}
    else:
        loader_device_map = device_map_strategy

    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=loader_device_map,
        max_memory=dict(max_memory) if max_memory is not None else None,
        attn_implementation=attn_implementation,
    )
    model.eval()

    device_map = getattr(model, "hf_device_map", None)
    if device_map_strategy is not None:
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            raise RuntimeError("Could not locate model input embeddings after sharded load")
        resolved_device = input_embeddings.weight.device
        if resolved_device.type != "cuda":
            raise RuntimeError(
                f"Input embeddings were placed on {resolved_device}, expected CUDA"
            )

    return LoadedModel(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=dtype,
        device_map=dict(device_map) if device_map is not None else None,
    )


def render_chat(
    loaded: LoadedModel,
    messages: Sequence[dict[str, str]],
) -> dict[str, torch.Tensor]:
    """Render a model-native chat template and return tensors on the model device."""

    template_kwargs: dict[str, Any] = {}
    if loaded.spec.disable_thinking:
        template_kwargs["enable_thinking"] = False

    encoded = loaded.tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    )
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}
    return {key: value.to(loaded.device) for key, value in encoded.items()}


def generation_eos_override(loaded: LoadedModel) -> dict[str, Any]:
    """Keep model-native stop tokens; use tokenizer only as a fallback."""
    config = getattr(loaded.model, "generation_config", None)
    if getattr(config, "eos_token_id", None) is not None:
        # Omitting the override lets Transformers retain the complete list.
        return {}
    eos = loaded.tokenizer.eos_token_id
    return {"eos_token_id": eos} if eos is not None else {}


def generate_answer(
    loaded: LoadedModel,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    repetition_penalty: float | None = None,
) -> GenerationResult:
    """Generate one answer from model-native chat messages."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if repetition_penalty is not None and not 0 < repetition_penalty < float('inf'):
        raise ValueError("repetition_penalty must be finite and positive")

    inputs = render_chat(loaded, messages)
    input_tokens = int(inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": loaded.tokenizer.pad_token_id,
    }
    generation_kwargs.update(generation_eos_override(loaded))
    if repetition_penalty is not None:
        generation_kwargs['repetition_penalty'] = repetition_penalty
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    torch.manual_seed(seed)
    if loaded.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        generated = loaded.model.generate(**inputs, **generation_kwargs)

    generated_only = generated[:, input_tokens:]
    answer = loaded.tokenizer.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    raw_answer = loaded.tokenizer.batch_decode(
        generated_only,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return GenerationResult(
        answer=answer,
        raw_answer=raw_answer,
        input_tokens=input_tokens,
        output_tokens=int(generated_only.shape[-1]),
    )


def release_model(loaded: LoadedModel) -> None:
    """Release model references and clear the CUDA allocator cache."""

    loaded.model = None
    loaded.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
