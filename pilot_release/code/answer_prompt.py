"""Frozen answer-generation prompts for TransferGEO benchmark inputs."""

from __future__ import annotations

from collections.abc import Mapping, Iterator, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_OFFICIAL_V1 = "benchmark_official_v1"
UNIFIED_ANSWER_V1 = "unified_answer_v1"
UNIFIED_ANSWER_V2 = "unified_answer_v2"
UNIFIED_ANSWER_V3 = "unified_answer_v3"
UNIFIED_ANSWER_V4 = "unified_answer_v4"
UNIFIED_ANSWER_V5 = "unified_answer_v5"
UNIFIED_ANSWER_V6 = "unified_answer_v6"
UNIFIED_ANSWER_V7 = "unified_answer_v7"
UNIFIED_ANSWER_V8 = "unified_answer_v8"
UNIFIED_ANSWER_V9 = "unified_answer_v9"
UNIFIED_ANSWER_PROTOCOLS = (
    UNIFIED_ANSWER_V1,
    UNIFIED_ANSWER_V2,
    UNIFIED_ANSWER_V3,
    UNIFIED_ANSWER_V4,
    UNIFIED_ANSWER_V5,
    UNIFIED_ANSWER_V6,
    UNIFIED_ANSWER_V7,
    UNIFIED_ANSWER_V8,
    UNIFIED_ANSWER_V9,
)
ANSWER_PROMPT_PROTOCOLS = (BENCHMARK_OFFICIAL_V1, *UNIFIED_ANSWER_PROTOCOLS)

UNIFIED_ANSWER_PROMPT_ROOTS = {
    UNIFIED_ANSWER_V1: ROOT / "prompts/answer_generation/unified_v1",
    UNIFIED_ANSWER_V2: ROOT / "prompts/answer_generation/unified_v2",
    UNIFIED_ANSWER_V3: ROOT / "prompts/answer_generation/unified_v3",
    UNIFIED_ANSWER_V4: ROOT / "prompts/answer_generation/unified_v4",
    UNIFIED_ANSWER_V5: ROOT / "prompts/answer_generation/unified_v5",
    UNIFIED_ANSWER_V6: ROOT / "prompts/answer_generation/unified_v6",
    UNIFIED_ANSWER_V7: ROOT / "prompts/answer_generation/unified_v7",
    UNIFIED_ANSWER_V8: ROOT / "prompts/answer_generation/unified_v8",
    UNIFIED_ANSWER_V9: ROOT / "prompts/answer_generation/unified_v9",
}
class _LazyPromptMapping(Mapping[str, str]):
    """Read only the selected version, retaining its text for this process."""

    def __init__(self, filename: str):
        self.filename = filename
        self._loaded: dict[str, str] = {}

    def __getitem__(self, protocol: str) -> str:
        # Preserve KeyError for an unknown mapping key; never fall back to v7.
        root = UNIFIED_ANSWER_PROMPT_ROOTS[protocol]
        if protocol not in self._loaded:
            path = root / self.filename
            try:
                value = path.read_text(encoding="utf-8").strip()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Selected answer prompt {protocol!r} requires {path}"
                ) from exc
            self._loaded[protocol] = value
        return self._loaded[protocol]

    def __iter__(self) -> Iterator[str]:
        return iter(UNIFIED_ANSWER_PROMPT_ROOTS)

    def __len__(self) -> int:
        return len(UNIFIED_ANSWER_PROMPT_ROOTS)

    def __contains__(self, protocol: object) -> bool:
        return protocol in UNIFIED_ANSWER_PROMPT_ROOTS


UNIFIED_ANSWER_SYSTEM_PROMPTS = _LazyPromptMapping("system_prompt.txt")
UNIFIED_ANSWER_USER_PROMPTS = _LazyPromptMapping("user_prompt.txt")

# Backward-compatible aliases: these remain frozen to unified_answer_v1.
UNIFIED_ANSWER_PROMPT_ROOT = UNIFIED_ANSWER_PROMPT_ROOTS[UNIFIED_ANSWER_V1]


def __getattr__(name: str) -> str:
    """Resolve legacy v1 string aliases only when explicitly requested."""
    if name == "UNIFIED_ANSWER_SYSTEM_PROMPT":
        return UNIFIED_ANSWER_SYSTEM_PROMPTS[UNIFIED_ANSWER_V1]
    if name == "UNIFIED_ANSWER_USER_PROMPT":
        return UNIFIED_ANSWER_USER_PROMPTS[UNIFIED_ANSWER_V1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_answer_prompt_files(protocol: str) -> dict[str, Path]:
    """Return prompt files for a file-backed protocol without silent fallback."""

    if protocol in UNIFIED_ANSWER_PROMPT_ROOTS:
        prompt_root = UNIFIED_ANSWER_PROMPT_ROOTS[protocol]
        return {
            "system": prompt_root / "system_prompt.txt",
            "user": prompt_root / "user_prompt.txt",
        }
    if protocol == BENCHMARK_OFFICIAL_V1:
        return {}
    supported = ", ".join(ANSWER_PROMPT_PROTOCOLS)
    raise ValueError(
        f"Unsupported answer prompt protocol {protocol!r}; choose: {supported}"
    )


def build_geo_messages(query: str, sources: Sequence[str]) -> list[dict[str, str]]:
    """Legacy official templates are deliberately not redistributed."""
    raise ValueError("Official benchmark templates are not included; explicitly use unified_answer_v7.")


def build_cseo_messages(domain: str, query: str, documents: Sequence[str]) -> list[dict[str, str]]:
    """Legacy official templates are deliberately not redistributed."""
    raise ValueError("Official benchmark templates are not included; explicitly use unified_answer_v7.")


def build_unified_answer_messages(
    query: str,
    sources: Sequence[str],
    *,
    protocol: str = UNIFIED_ANSWER_V1,
) -> list[dict[str, str]]:
    """Build one versioned benchmark-independent answer-prompt protocol."""

    if protocol not in UNIFIED_ANSWER_PROTOCOLS:
        supported = ", ".join(UNIFIED_ANSWER_PROTOCOLS)
        raise ValueError(
            f"Unsupported unified answer prompt protocol {protocol!r}; "
            f"choose: {supported}"
        )

    source_text = "\n\n".join(
        f"Source {index}:\n{source}"
        for index, source in enumerate(sources, start=1)
    )
    valid_citation_identifiers = ", ".join(
        f"[{index}]" for index in range(1, len(sources) + 1)
    )
    return [
        {"role": "system", "content": UNIFIED_ANSWER_SYSTEM_PROMPTS[protocol]},
        {
            "role": "user",
            "content": UNIFIED_ANSWER_USER_PROMPTS[protocol].format(
                query=query,
                sources=source_text,
                valid_citation_identifiers=valid_citation_identifiers,
            ),
        },
    ]


def build_answer_messages(
    *,
    protocol: str,
    benchmark: str,
    domain: str,
    query: str,
    sources: Sequence[str],
) -> list[dict[str, str]]:
    """Dispatch to a frozen answer-prompt protocol without implicit fallback."""

    if protocol in UNIFIED_ANSWER_PROTOCOLS:
        return build_unified_answer_messages(query, sources, protocol=protocol)
    if protocol == BENCHMARK_OFFICIAL_V1:
        if benchmark == "geo_bench":
            return build_geo_messages(query, sources)
        if benchmark == "cseo_bench":
            return build_cseo_messages(domain, query, sources)
        raise ValueError(f"Unsupported benchmark for official prompts: {benchmark!r}")
    supported = ", ".join(ANSWER_PROMPT_PROTOCOLS)
    raise ValueError(
        f"Unsupported answer prompt protocol {protocol!r}; choose: {supported}"
    )
