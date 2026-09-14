"""Official-code-compatible GEO objective impression metrics.

This module is a clean-room, dependency-minimal port of the objective metric
logic published by the GEO authors in ``GEO-optim/GEO/src/utils.py``.  The
published implementation has been unchanged since commit
``4b78a82f0a3b5b0bc239ef803a4339ed3dcef39b``; the source blob is still
``5a82f342e7be0760ae2b5c530fba8bb5c37e3410`` at the audited repository HEAD
``c9e985f2bc4b539a01e8e9d226ff2a3d8d29a888``.

Compatibility choices intentionally preserved:

* paragraph split on exactly two newlines;
* NLTK ``sent_tokenize`` and ``word_tokenize``;
* count only tokens whose string length is greater than two;
* the authors' citation regular expression;
* divide a sentence contribution by every parsed citation occurrence,
  including out-of-range occurrences;
* ignore positive out-of-range source indices;
* retain the legacy Python negative-index behavior for citation ``[0]``;
* normalize each source-score vector to sum to one;
* optionally reproduce the authors' uniform fallback when the score sum is
  zero, or use a zero fallback only for a clearly labelled sensitivity view.

No LLM, GPU, network access, answer repair, or answer filtering occurs here.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

import nltk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NLTK_DATA_DIR = ROOT / "data/raw/nltk_data"
EXPECTED_NLTK_VERSION = "3.10.2"
PUNKT_RESOURCE = "tokenizers/punkt_tab/english"
PUNKT_ARCHIVE = "tokenizers/punkt_tab.zip"
EXPECTED_PUNKT_ARCHIVE_SHA256 = (
    "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
)

OFFICIAL_REPOSITORY = "https://github.com/GEO-optim/GEO"
OFFICIAL_INITIAL_METRIC_COMMIT = "4b78a82f0a3b5b0bc239ef803a4339ed3dcef39b"
OFFICIAL_AUDITED_HEAD = "c9e985f2bc4b539a01e8e9d226ff2a3d8d29a888"
OFFICIAL_UTILS_BLOB = "5a82f342e7be0760ae2b5c530fba8bb5c37e3410"
OFFICIAL_UTILS_SHA256 = "569baf8060d5a4a343da09f367cc6c9a746db19b0a700a0f70b42f0aea42bcea"
OFFICIAL_CITATION_PATTERN = re.compile(r"\[[^\w\s]*\d+[^\w\s]*\]")

FallbackPolicy = Literal["official_uniform", "zero"]
METRIC_NAMES = ("word", "position", "overall")


@dataclass(frozen=True)
class ParsedSentence:
    """One sentence after the exact preprocessing used by official GEO code."""

    text: str
    tokens: tuple[str, ...]
    citations: tuple[int, ...]

    @property
    def eligible_word_count(self) -> int:
        """Official ``get_num_words``: tokens with string length > 2."""

        return sum(len(token) > 2 for token in self.tokens)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def ensure_official_tokenizer(
    nltk_data_dir: Path = DEFAULT_NLTK_DATA_DIR,
    *,
    require_version: str = EXPECTED_NLTK_VERSION,
) -> dict[str, Any]:
    """Make the pinned local Punkt resource available and return provenance."""

    if nltk.__version__ != require_version:
        raise RuntimeError(
            f"Expected nltk=={require_version}, found nltk=={nltk.__version__}. "
            "Install requirements/evaluation.txt in the transfergeo environment."
        )
    resolved = nltk_data_dir.resolve()
    if str(resolved) not in nltk.data.path:
        nltk.data.path.insert(0, str(resolved))
    try:
        resource_path = nltk.data.find(PUNKT_RESOURCE)
    except LookupError as exc:
        raise RuntimeError(
            "Missing pinned punkt_tab resource. Download it with NLTK_DATA set "
            f"to {resolved}."
        ) from exc
    archive = resolved / PUNKT_ARCHIVE
    archive_sha256 = sha256_file(archive) if archive.is_file() else None
    return {
        "nltk_version": nltk.__version__,
        "nltk_data_dir": str(resolved),
        "punkt_resource": str(resource_path),
        "punkt_archive": str(archive),
        "punkt_archive_sha256": archive_sha256,
        "expected_punkt_archive_sha256": EXPECTED_PUNKT_ARCHIVE_SHA256,
        "punkt_archive_hash_matches": archive_sha256
        == EXPECTED_PUNKT_ARCHIVE_SHA256,
    }


def extract_official_citations(sentence: str) -> tuple[int, ...]:
    """Apply the authors' regex and retain only the first integer per match."""

    return tuple(
        int(re.findall(r"\d+", citation)[0])
        for citation in OFFICIAL_CITATION_PATTERN.findall(sentence)
    )


def parse_answer_official(
    answer: str,
    *,
    nltk_data_dir: Path = DEFAULT_NLTK_DATA_DIR,
) -> list[ParsedSentence]:
    """Reproduce ``extract_citations_new`` without importing its OpenAI stack."""

    ensure_official_tokenizer(nltk_data_dir)
    paragraphs = re.split(r"\n\n", answer)
    sentences: list[ParsedSentence] = []
    for paragraph in paragraphs:
        for sentence in nltk.sent_tokenize(paragraph):
            # Calling word_tokenize with its default preserve_line=False matches
            # the public implementation even though the input is already a sentence.
            tokens = tuple(nltk.word_tokenize(sentence))
            sentences.append(
                ParsedSentence(
                    text=sentence,
                    tokens=tokens,
                    citations=extract_official_citations(sentence),
                )
            )
    return sentences


def _normalize(scores: list[float], fallback_policy: FallbackPolicy) -> tuple[list[float], bool]:
    denominator = sum(scores)
    if denominator != 0:
        return [score / denominator for score in scores], False
    if fallback_policy == "official_uniform":
        return [1.0 / len(scores) for _ in scores], True
    if fallback_policy == "zero":
        return [0.0 for _ in scores], True
    raise ValueError(f"Unsupported fallback policy: {fallback_policy}")


def score_parsed_answer(
    sentences: Iterable[ParsedSentence],
    *,
    source_count: int,
    fallback_policy: FallbackPolicy = "official_uniform",
) -> dict[str, Any]:
    """Compute normalized Word, Position, and Overall vectors.

    ``official_uniform`` is the primary reproduction. ``zero`` changes only
    zero-sum fallback behavior and is reserved for sensitivity analysis.
    """

    if source_count <= 0:
        raise ValueError(f"source_count must be positive, got {source_count}")
    sentence_list = list(sentences)
    raw = {name: [0.0] * source_count for name in METRIC_NAMES}
    citation_occurrences = 0
    in_range_occurrences = 0
    out_of_range_occurrences = 0
    zero_citation_occurrences = 0
    eligible_word_total = 0
    sentence_count = len(sentence_list)

    for index, sentence in enumerate(sentence_list):
        eligible_words = sentence.eligible_word_count
        eligible_word_total += eligible_words
        citations = sentence.citations
        if not citations:
            continue
        citation_occurrences += len(citations)
        decay = (
            math.exp(-index / (sentence_count - 1)) if sentence_count > 1 else 1.0
        )
        contributions = {
            "word": eligible_words / len(citations),
            "position": decay / len(citations),
            "overall": eligible_words * decay / len(citations),
        }
        for citation in citations:
            if 1 <= citation <= source_count:
                in_range_occurrences += 1
            elif citation == 0:
                # The official code indexes scores[citation - 1], so [0]
                # credits the final source. Preserve and expose that behavior.
                zero_citation_occurrences += 1
            else:
                out_of_range_occurrences += 1
            try:
                source_index = citation - 1
                for metric_name, contribution in contributions.items():
                    raw[metric_name][source_index] += contribution
            except IndexError:
                # Exact official behavior for positive source indices > n.
                continue

    normalized: dict[str, list[float]] = {}
    fallback_used: dict[str, bool] = {}
    raw_sums: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        raw_sums[metric_name] = sum(raw[metric_name])
        normalized[metric_name], fallback_used[metric_name] = _normalize(
            raw[metric_name], fallback_policy
        )

    return {
        "fallback_policy": fallback_policy,
        "source_count": source_count,
        "sentence_count": sentence_count,
        "eligible_word_count": eligible_word_total,
        "citation_occurrence_count": citation_occurrences,
        "in_range_citation_occurrence_count": in_range_occurrences,
        "out_of_range_citation_occurrence_count": out_of_range_occurrences,
        "zero_citation_occurrence_count": zero_citation_occurrences,
        "has_in_range_citation": in_range_occurrences > 0,
        "raw": raw,
        "raw_sums": raw_sums,
        "normalized": normalized,
        "percent": {
            name: [100.0 * value for value in normalized[name]]
            for name in METRIC_NAMES
        },
        "fallback_used": fallback_used,
    }


def score_answer(
    answer: str,
    *,
    source_count: int,
    fallback_policy: FallbackPolicy = "official_uniform",
    nltk_data_dir: Path = DEFAULT_NLTK_DATA_DIR,
) -> dict[str, Any]:
    parsed = parse_answer_official(answer, nltk_data_dir=nltk_data_dir)
    return score_parsed_answer(
        parsed,
        source_count=source_count,
        fallback_policy=fallback_policy,
    )


def target_scores(score: dict[str, Any], target_document_index: int) -> dict[str, Any]:
    """Extract the zero-based target source's normalized and percent scores."""

    source_count = int(score["source_count"])
    if not 0 <= target_document_index < source_count:
        raise IndexError(
            f"target_document_index={target_document_index} outside {source_count} sources"
        )
    return {
        name: float(score["normalized"][name][target_document_index])
        for name in METRIC_NAMES
    } | {
        f"{name}_percent": float(score["percent"][name][target_document_index])
        for name in METRIC_NAMES
    }


def paired_improvement(baseline: float, optimized: float) -> dict[str, float | None]:
    """Return paired delta and paper-defined relative improvement.

    Relative improvement is undefined for a zero baseline. No epsilon or
    imputation is introduced.
    """

    baseline = float(baseline)
    optimized = float(optimized)
    return {
        "baseline": baseline,
        "optimized": optimized,
        "delta": optimized - baseline,
        "relative_percent": (
            (optimized - baseline) / baseline * 100.0 if baseline != 0 else None
        ),
    }


def implementation_provenance(
    nltk_data_dir: Path = DEFAULT_NLTK_DATA_DIR,
) -> dict[str, Any]:
    return {
        "protocol": "geo_official_code_compatible_v1",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_initial_metric_commit": OFFICIAL_INITIAL_METRIC_COMMIT,
        "official_audited_head": OFFICIAL_AUDITED_HEAD,
        "official_utils_blob": OFFICIAL_UTILS_BLOB,
        "official_utils_sha256": OFFICIAL_UTILS_SHA256,
        "official_source_lines": "src/utils.py:24-77",
        "tokenizer": ensure_official_tokenizer(nltk_data_dir),
        "primary_fallback_policy": "official_uniform",
        "sensitivity_fallback_policy": "zero",
        "relative_improvement_zero_baseline_policy": "null_without_epsilon",
    }
