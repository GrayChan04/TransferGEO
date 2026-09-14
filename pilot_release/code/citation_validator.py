"""Deterministic citation-format diagnostics for GEO/C-SEO answers.

This module never edits, retries, or filters an answer. It only records whether
the answer can be scored by the official numeric-source metrics and preserves
the raw answer for sensitivity analysis.
"""

from __future__ import annotations

import re
from typing import Any


NUMERIC_CITATION = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
NAMED_BRACKET = re.compile(r"\[([^\]\d][^\]]*)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
V2_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\"'“‘])")
V2_SEPARATE_CITATIONS_BEFORE_PUNCTUATION = re.compile(
    r"(?:\[\d+\])+\s*[.!?。！？](?:[\"'”’)]*)$"
)
V2_COMBINED_OR_RANGED_CITATION = re.compile(
    r"\[\s*\d+(?:\s*[-,]\s*\d+)+\s*\]"
)
V2_NAMED_SOURCE_CITATION = re.compile(
    r"(?:\[\s*Source\s+\d+\s*\]|\(\s*Source\s+\d+\s*\)|\bSource\s+\d+\b)",
    re.IGNORECASE,
)
V4_SEPARATE_NUMERIC_CITATION = re.compile(r"\[(\d+)\]")
V4_CITATION_LIKE = re.compile(
    r"(?:"
    r"\[\s*(?:Source\s+)?\d[^\]\n]*\]"
    r"|\(\s*Source\s+\d+\s*\)"
    r"|\bSource\s+\d+\b"
    r")",
    re.IGNORECASE,
)
V4_FINAL_CITATION_BLOCK = re.compile(
    r"(?P<block>(?:\[\d+\])+)(?P<spacing>\s*)"
    r"(?P<punctuation>[.!?。！？])(?:[\"'”’)]*)$"
)
V6_FINAL_CITATION_BLOCK = re.compile(
    r"(?P<block>(?:\[\d+\][ \t]*)+)(?P<punctuation>[.!?。！？])"
    r"(?:[\"'”’)]*)$"
)
V4_REFERENCE_SECTION_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:references|sources|bibliography|参考文献|资料来源|来源)\s*:?\s*$"
)
V4_RESTATED_SOURCE_MAPPING = re.compile(
    r"(?im)^\s*(?:[-*+]\s*)?(?:Source|来源)\s+\d+\s*:\s*\S+"
)
V6_BRACKET_GROUP = re.compile(r"\[[^\[\]\n]*\]")
V6_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+")
V6_MARKDOWN_TABLE_ROW = re.compile(r"^\s*(?:[^|\n]*\|){2,}")
V6_REFERENCE_SECTION_LINE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:references|sources|bibliography|参考文献|资料来源|来源)"
    r"(?:\s*(?:\[\d+\])+)?\s*[:：]?\s*[.!?。！？]?\s*$"
)
V6_LEXICAL_CONTENT = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
V6_MARKDOWN_EMPHASIS_RUN = re.compile(r"(?<!\\)(\*{1,3}|_{1,3})")


def _expand_numeric(value: str) -> list[int]:
    values: list[int] = []
    for part in re.split(r"\s*[,\-]\s*", value):
        if part.isdigit():
            values.append(int(part))
    return values


def validate_citations(answer: str, *, source_count: int) -> dict[str, Any]:
    """Return citation diagnostics without changing ``answer``."""

    matches = NUMERIC_CITATION.findall(answer)
    citations = [number for match in matches for number in _expand_numeric(match)]
    named = [value.strip() for value in NAMED_BRACKET.findall(answer) if value.strip()]
    invalid = sorted({value for value in citations if not 1 <= value <= source_count})
    sentences = [part.strip() for part in SENTENCE_END.split(answer) if part.strip()]
    citationless = [sentence for sentence in sentences if not NUMERIC_CITATION.search(sentence)]
    return {
        "source_count": source_count,
        "numeric_citations": citations,
        "named_bracket_citations": named,
        "numeric_citation_count": len(citations),
        "has_numeric_citation": bool(citations),
        "has_named_bracket_citation": bool(named),
        "invalid_numeric_citations": invalid,
        "citationless_sentence_count": len(citationless),
        "citation_format_scorable": (
            bool(citations)
            and not invalid
            and not named
            and not citationless
        ),
        "raw_answer_preserved": True,
    }


def _v2_sentence_units(answer: str) -> list[str]:
    """Split prose for the frozen v2 format gate without external tokenizers."""

    units: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for language_segment in re.split(r"(?<=[。！？])", stripped):
            units.extend(
                part.strip()
                for part in V2_SENTENCE_END.split(language_segment.strip())
                if part.strip()
            )
    return units


def _v6_sentence_units(answer: str) -> list[str]:
    """Split v6 output with the tokenizer used by the official GEO metric.

    Each non-empty output line is handled separately so headings, list items,
    and fragments cannot disappear inside adjacent prose.  Chinese terminal
    punctuation is split explicitly because the pinned English Punkt model is
    used for the English benchmark protocol.  A narrow post-processing rule
    rejoins a Punkt split made inside a Markdown-emphasized title, for example
    ``*Am I The Only Sane One Working Here?*``; the question mark in that title
    is not a sentence boundary.
    """

    import nltk

    from geo_objective_metrics import ensure_official_tokenizer

    ensure_official_tokenizer()
    units: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for language_segment in re.split(r"(?<=[。！？])", stripped):
            segment = language_segment.strip()
            if not segment:
                continue
            tokenized = [
                sentence.strip()
                for sentence in nltk.sent_tokenize(segment)
                if sentence.strip()
            ]
            for sentence in tokenized:
                if units:
                    delimiter_counts: dict[str, int] = {}
                    for delimiter in V6_MARKDOWN_EMPHASIS_RUN.findall(units[-1]):
                        delimiter_counts[delimiter] = (
                            delimiter_counts.get(delimiter, 0) + 1
                        )
                    open_delimiters = {
                        delimiter
                        for delimiter, count in delimiter_counts.items()
                        if count % 2
                    }
                    sentence_delimiters = set(
                        V6_MARKDOWN_EMPHASIS_RUN.findall(sentence)
                    )
                    closing_delimiter = next(
                        (
                            delimiter
                            for delimiter in sorted(
                                open_delimiters,
                                key=len,
                                reverse=True,
                            )
                            if delimiter in sentence_delimiters
                        ),
                        None,
                    )
                    if closing_delimiter is not None:
                        separator = (
                            ""
                            if sentence.startswith(
                                (closing_delimiter, ":", ";", ",")
                            )
                            else " "
                        )
                        units[-1] += separator + sentence
                        continue
                units.append(sentence)
    return units


def validate_v2_citation_contract(
    answer: str,
    *,
    source_count: int,
) -> dict[str, Any]:
    """Validate only the deterministic syntax rules frozen for unified v2.

    This does not judge whether a cited source semantically supports a claim.
    It never edits the answer and remains separate from the historical
    ``validate_citations`` payload so existing stored diagnostics stay stable.
    """

    if source_count <= 0:
        raise ValueError("source_count must be positive")
    base = validate_citations(answer, source_count=source_count)
    sentences = _v2_sentence_units(answer)
    sentence_compliance = [
        bool(V2_SEPARATE_CITATIONS_BEFORE_PUNCTUATION.search(sentence))
        for sentence in sentences
    ]
    combined_or_ranged = V2_COMBINED_OR_RANGED_CITATION.findall(answer)
    named_source = V2_NAMED_SOURCE_CITATION.findall(answer)
    invalid = list(base["invalid_numeric_citations"])
    noncompliant = [
        sentence
        for sentence, compliant in zip(sentences, sentence_compliance, strict=True)
        if not compliant
    ]
    compliant = (
        bool(sentences)
        and all(sentence_compliance)
        and not invalid
        and not named_source
        and not combined_or_ranged
    )
    return {
        "source_count": source_count,
        "sentence_count": len(sentences),
        "compliant_sentence_count": sum(sentence_compliance),
        "noncompliant_sentence_count": len(noncompliant),
        "noncompliant_sentence_first_5": noncompliant[:5],
        "invalid_numeric_citations": invalid,
        "named_source_citations": named_source,
        "combined_or_ranged_citations": combined_or_ranged,
        "contract_compliant": compliant,
        "semantic_support_checked": False,
        "raw_answer_preserved": True,
    }


def validate_v4_optional_citation_contract(
    answer: str,
    *,
    source_count: int,
) -> dict[str, Any]:
    """Validate the optional-but-well-formed citation contract for unified v4.

    An uncited sentence is format-compliant. If a sentence contains anything
    that looks like a numeric source citation, every citation must be a
    separate in-range ``[k]`` token in one contiguous block immediately before
    the sentence-final punctuation. Semantic source support is deliberately
    outside this deterministic validator.
    """

    if source_count <= 0:
        raise ValueError("source_count must be positive")

    sentences = _v2_sentence_units(answer)
    numeric_citations = [
        int(match.group(1))
        for match in V4_SEPARATE_NUMERIC_CITATION.finditer(answer)
    ]
    invalid_numeric_citations = sorted(
        {value for value in numeric_citations if not 1 <= value <= source_count}
    )
    named_source_citations = V2_NAMED_SOURCE_CITATION.findall(answer)
    combined_or_ranged_citations = V2_COMBINED_OR_RANGED_CITATION.findall(answer)

    citation_bearing_sentences: list[str] = []
    uncited_sentences: list[str] = []
    compliant_citation_bearing_sentences: list[str] = []
    malformed_citation_sentences: list[str] = []

    for sentence in sentences:
        citation_like = list(V4_CITATION_LIKE.finditer(sentence))
        if not citation_like:
            uncited_sentences.append(sentence)
            continue

        citation_bearing_sentences.append(sentence)
        final_block = V4_FINAL_CITATION_BLOCK.search(sentence)
        separate_tokens = list(V4_SEPARATE_NUMERIC_CITATION.finditer(sentence))
        all_tokens_in_final_block = False
        if final_block is not None and separate_tokens:
            block_start, block_end = final_block.span("block")
            all_tokens_in_final_block = all(
                token.start() >= block_start and token.end() <= block_end
                for token in separate_tokens
            )
        sentence_numbers = [int(token.group(1)) for token in separate_tokens]
        all_numbers_in_range = bool(sentence_numbers) and all(
            1 <= value <= source_count for value in sentence_numbers
        )
        has_named_source = bool(V2_NAMED_SOURCE_CITATION.search(sentence))
        has_combined_or_ranged = bool(
            V2_COMBINED_OR_RANGED_CITATION.search(sentence)
        )
        citation_like_is_fully_tokenized = len(citation_like) == len(separate_tokens)
        sentence_compliant = (
            final_block is not None
            and all_tokens_in_final_block
            and all_numbers_in_range
            and citation_like_is_fully_tokenized
            and not has_named_source
            and not has_combined_or_ranged
        )
        if sentence_compliant:
            compliant_citation_bearing_sentences.append(sentence)
        else:
            malformed_citation_sentences.append(sentence)

    separate_reference_section_detected = bool(
        V4_REFERENCE_SECTION_HEADING.search(answer)
    )
    restated_source_mapping_detected = bool(
        V4_RESTATED_SOURCE_MAPPING.search(answer)
    )
    contract_compliant = (
        bool(sentences)
        and not malformed_citation_sentences
        and not invalid_numeric_citations
        and not named_source_citations
        and not combined_or_ranged_citations
        and not separate_reference_section_detected
        and not restated_source_mapping_detected
    )
    return {
        "source_count": source_count,
        "sentence_count": len(sentences),
        "citation_bearing_sentence_count": len(citation_bearing_sentences),
        "uncited_sentence_count": len(uncited_sentences),
        "compliant_citation_bearing_sentence_count": len(
            compliant_citation_bearing_sentences
        ),
        "malformed_citation_sentence_count": len(malformed_citation_sentences),
        "malformed_citation_sentence_first_5": malformed_citation_sentences[:5],
        "numeric_citations": numeric_citations,
        "invalid_numeric_citations": invalid_numeric_citations,
        "named_source_citations": named_source_citations,
        "combined_or_ranged_citations": combined_or_ranged_citations,
        "separate_reference_section_detected": (
            separate_reference_section_detected
        ),
        "restated_source_mapping_detected": restated_source_mapping_detected,
        "citationless_answer": (
            bool(sentences) and not citation_bearing_sentences
        ),
        "has_in_range_numeric_citation": any(
            1 <= value <= source_count for value in numeric_citations
        ),
        "contract_compliant": contract_compliant,
        "semantic_support_checked": False,
        "raw_answer_preserved": True,
    }


def validate_v6_strict_citation_contract(
    answer: str,
    *,
    source_count: int,
) -> dict[str, Any]:
    """Validate the strict, repair-oriented citation syntax frozen for v6.

    Every non-empty sentence unit must end with one final sequence of exact,
    canonical, in-range ``[k]`` tokens before terminal punctuation. Horizontal
    whitespace between citation tokens and before punctuation is allowed.
    Unlike the historical v2 gate, this rejects an otherwise legal numeric
    citation appearing earlier in a sentence.  It also rejects every other use
    of square brackets.  Semantic source support remains a separate manual or
    model-based evaluation and the raw answer is never edited here.
    """

    if source_count <= 0:
        raise ValueError("source_count must be positive")

    sentences = _v6_sentence_units(answer)
    numeric_citations = [
        int(match.group(1))
        for match in V4_SEPARATE_NUMERIC_CITATION.finditer(answer)
    ]
    invalid_numeric_citations = sorted(
        {value for value in numeric_citations if not 1 <= value <= source_count}
    )
    noncanonical_numeric_citations: list[str] = []
    malformed_bracket_groups: list[str] = []
    compliant_sentences: list[str] = []
    noncompliant_sentences: list[str] = []
    uncited_sentences: list[str] = []

    for sentence in sentences:
        bracket_groups = list(V6_BRACKET_GROUP.finditer(sentence))
        separate_tokens = list(V4_SEPARATE_NUMERIC_CITATION.finditer(sentence))
        token_spans = {token.span() for token in separate_tokens}
        group_spans = {group.span() for group in bracket_groups}
        residual = V6_BRACKET_GROUP.sub("", sentence)
        has_stray_bracket = "[" in residual or "]" in residual

        for token in separate_tokens:
            canonical = f"[{int(token.group(1))}]"
            if token.group(0) != canonical:
                noncanonical_numeric_citations.append(token.group(0))
        malformed_groups_here = [
            group.group(0)
            for group in bracket_groups
            if group.span() not in token_spans
        ]
        malformed_bracket_groups.extend(malformed_groups_here)

        if not separate_tokens:
            uncited_sentences.append(sentence)

        final_block = V6_FINAL_CITATION_BLOCK.search(sentence)
        all_tokens_in_final_block = False
        if final_block is not None and separate_tokens:
            block_start, block_end = final_block.span("block")
            all_tokens_in_final_block = all(
                token.start() >= block_start and token.end() <= block_end
                for token in separate_tokens
            )
        all_numbers_in_range = bool(separate_tokens) and all(
            1 <= int(token.group(1)) <= source_count
            for token in separate_tokens
        )
        all_tokens_canonical = all(
            token.group(0) == f"[{int(token.group(1))}]"
            for token in separate_tokens
        )
        text_without_brackets = V6_BRACKET_GROUP.sub("", sentence)
        has_lexical_content = bool(
            V6_LEXICAL_CONTENT.search(text_without_brackets)
        )
        markdown_heading = bool(V6_MARKDOWN_HEADING.search(sentence))
        markdown_table_row = bool(V6_MARKDOWN_TABLE_ROW.search(sentence))
        sentence_compliant = (
            final_block is not None
            and all_tokens_in_final_block
            and all_numbers_in_range
            and all_tokens_canonical
            and group_spans == token_spans
            and not has_stray_bracket
            and has_lexical_content
            and not markdown_heading
            and not markdown_table_row
        )
        if sentence_compliant:
            compliant_sentences.append(sentence)
        else:
            noncompliant_sentences.append(sentence)

    separate_reference_section_detected = bool(
        V4_REFERENCE_SECTION_HEADING.search(answer)
        or V6_REFERENCE_SECTION_LINE.search(answer)
    )
    restated_source_mapping_detected = bool(
        V4_RESTATED_SOURCE_MAPPING.search(answer)
    )
    contract_compliant = (
        bool(sentences)
        and not noncompliant_sentences
        and not invalid_numeric_citations
        and not noncanonical_numeric_citations
        and not malformed_bracket_groups
        and not separate_reference_section_detected
        and not restated_source_mapping_detected
    )
    return {
        "source_count": source_count,
        "sentence_count": len(sentences),
        "compliant_sentence_count": len(compliant_sentences),
        "noncompliant_sentence_count": len(noncompliant_sentences),
        "noncompliant_sentence_first_5": noncompliant_sentences[:5],
        "uncited_sentence_count": len(uncited_sentences),
        "uncited_sentence_first_5": uncited_sentences[:5],
        "numeric_citations": numeric_citations,
        "invalid_numeric_citations": invalid_numeric_citations,
        "noncanonical_numeric_citations": sorted(
            set(noncanonical_numeric_citations)
        ),
        "malformed_bracket_groups": sorted(set(malformed_bracket_groups)),
        "separate_reference_section_detected": (
            separate_reference_section_detected
        ),
        "restated_source_mapping_detected": restated_source_mapping_detected,
        "citationless_answer": bool(sentences) and not numeric_citations,
        "contract_compliant": contract_compliant,
        "semantic_support_checked": False,
        "raw_answer_preserved": True,
    }
