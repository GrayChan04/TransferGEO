"""Deterministic citation canonicalization for visibility-scoring copies.

Raw generated answers are never changed on disk. This module only creates a
derived scoring view and never judges whether a source supports a claim.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


REFERENCE_HEADING_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:references|sources|bibliography|citations|参考文献|资料来源|来源)"
    r"\s*[:：]?\s*$",
    re.IGNORECASE,
)
REFERENCE_HEADING_START = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:references|sources|bibliography|citations|参考文献|资料来源|来源)"
    r"\s*[:：]",
    re.IGNORECASE,
)
DIVIDER_LINE = re.compile(r"^\s*[-_*=—–]{3,}\s*$")
NUMERIC_BRACKET = re.compile(
    r"\[\s*(?P<body>\d+(?:\s*(?:,|-)\s*\d+)*)\s*\]"
)
NAMED_SOURCE_WRAPPER = re.compile(
    r"(?:"
    r"\[\s*Sources?\s*(?P<bracket>\d+)\s*\]"
    r"|\(\s*Sources?\s*(?P<paren>\d+)\s*\)"
    r")",
    re.IGNORECASE,
)
PAREN_NAMED_SOURCE_LIST = re.compile(
    r"\(\s*Sources?\s+\d+"
    r"(?:\s*[,，]\s*Sources?\s+\d+)+\s*\)",
    re.IGNORECASE,
)
PLAIN_SOURCE_NUMBER = re.compile(
    r"\b(?P<label>Sources?)\s+(?P<number>\d+)\b",
    re.IGNORECASE,
)
CANONICAL_CLUSTER = re.compile(
    r"(?P<cluster>\[\d+\](?:(?:[ \t]*,[ \t]*|[ \t]*)\[\d+\])*)"
)
POST_PUNCTUATION_CLUSTER = re.compile(
    r"(?P<punctuation>[.!?。！？])"
    r"(?P<closers>[\"'”’）)]*)"
    r"[ \t]*"
    r"(?P<cluster>(?:\[\d+\])+)"
    r"(?=\s|$)"
)
SPACE_BEFORE_PUNCTUATION = re.compile(
    r"(?P<cluster>(?:\[\d+\])+)[ \t]+(?P<punctuation>[.!?。！？])"
)
MALFORMED_NUMERIC_BRACKET = re.compile(r"\[[^\]\n]*\d+[^\]\n]*\]")
CANONICAL_TOKEN = re.compile(r"\[\d+\]")
LEFTOVER_NAMED_SOURCE = re.compile(r"\bSources?\s*\d+\b", re.IGNORECASE)
PURE_CITATION_PARAGRAPH = re.compile(
    r"\s*(?:\[\s*\d+(?:\s*(?:,|-)\s*\d+)*\s*\]"
    r"(?:\s*[,;，；]?\s*)?)+[.!?。！？]?\s*"
)
PLACEHOLDER_REFERENCE_LINE = re.compile(
    r"\s*(?:[-*]\s*)?"
    r"\[\s*\d+\s*\]\s*"
    r"Sources?\s*(?:\[\s*\d+\s*\]|\d+)"
    r"[.!?。！？]?\s*",
    re.IGNORECASE,
)
UNUSED_SOURCES_NOTE = re.compile(
    r"(?:\n\s*\n)?"
    r"\[\s*Sources?\s+not\s+used\s+in\s+the\s+answer\s*:[^\]\n]*\]"
    r"\s*$",
    re.IGNORECASE,
)
ORPHANED_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+(?=[.!?。！？])")


def _citation_numbers(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def strip_unheaded_bibliography(answer: str) -> tuple[str, str | None]:
    """Strip a separated terminal numbered bibliography, never body prose.

    Require multiple entries and bibliographic evidence in every entry.
    A list occupying the entire answer is left intact for manual review.
    """
    entry = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]+)?\[\d+\][ \t]+(?=\S)")
    evidence = re.compile(r"https?://|\bdoi\b|\b(?:18|19|20)\d{2}\b|\(n\.d\.\)|^Sources?\s*\[\d+\]\s*:", re.I)
    starts = list(entry.finditer(answer))
    for i, start in enumerate(starts):
        if len(starts) - i < 2:
            break
        prefix = answer[:start.start()]
        if not prefix.strip() or not re.search(r"\n[ \t]*\n[ \t]*$", prefix):
            continue
        chunks = [answer[m.end():starts[j+1].start() if j+1 < len(starts) else len(answer)]
                  for j, m in enumerate(starts) if j >= i]
        if all(evidence.search(chunk) for chunk in chunks):
            return prefix.rstrip(), answer[start.start():].strip()
    return answer, None


def _strip_trailing_reference_section(answer: str) -> tuple[str, dict[str, Any] | None]:
    """Remove successive eligible tails using the same existing single-tail rule."""
    original = answer
    removals = []
    while True:
        cleaned, removal = _strip_trailing_reference_section_once(answer)
        if removal is None:
            break
        if len(cleaned) >= len(answer) or not answer.startswith(cleaned):
            raise ValueError('Reference cleanup must strictly remove a suffix')
        answer = cleaned
        removals.append(removal)
    if not removals:
        return answer, None
    if len(removals) == 1:
        return answer, removals[0]
    removed = original[len(answer):].strip()
    return answer, {'type': 'successive_reference_tails', 'removed_text': removed,
                    'removed_citation_numbers': _citation_numbers(removed),
                    'individual_removals': removals}


def _strip_trailing_reference_section_once(answer: str) -> tuple[str, dict[str, Any] | None]:
    """Remove only an explicitly separated trailing reference-list section."""

    unused_note = UNUSED_SOURCES_NOTE.search(answer)
    if unused_note is not None:
        prefix = answer[: unused_note.start()].rstrip()
        if prefix:
            removed = unused_note.group(0).strip()
            return prefix, {
                "type": "explicit_unused_sources_note",
                "removed_text": removed,
                "removed_citation_numbers": _citation_numbers(removed),
            }

    lines = answer.splitlines(keepends=True)
    heading_indices = [
        index
        for index, line in enumerate(lines)
        if REFERENCE_HEADING_LINE.fullmatch(line.strip())
        or REFERENCE_HEADING_START.match(line.strip())
    ]
    if heading_indices:
        index = heading_indices[-1]
        prefix = "".join(lines[:index]).rstrip()
        removed = "".join(lines[index:]).strip()
        if prefix and removed:
            prefix_lines = prefix.splitlines(keepends=True)
            while prefix_lines and (
                not prefix_lines[-1].strip()
                or DIVIDER_LINE.fullmatch(prefix_lines[-1].strip())
            ):
                prefix_lines.pop()
            prefix = "".join(prefix_lines).rstrip()
            if prefix:
                return prefix, {
                    "type": "explicit_reference_section",
                    "removed_text": removed,
                    "removed_citation_numbers": _citation_numbers(removed),
                }

    paragraph_matches = list(re.finditer(r"\n\s*\n", answer))
    if paragraph_matches:
        boundary = paragraph_matches[-1]
        tail = answer[boundary.end() :].strip()
        tail_lines = [line for line in tail.splitlines() if line.strip()]
        citation_only_tail = bool(
            tail
            and (
                PURE_CITATION_PARAGRAPH.fullmatch(tail)
                or (
                    tail_lines
                    and all(
                        PLACEHOLDER_REFERENCE_LINE.fullmatch(line)
                        for line in tail_lines
                    )
                )
            )
        )
        if citation_only_tail:
            prefix = answer[: boundary.start()].rstrip()
            prefix_lines = prefix.splitlines(keepends=True)
            while prefix_lines and DIVIDER_LINE.fullmatch(prefix_lines[-1].strip()):
                prefix_lines.pop()
            prefix = "".join(prefix_lines).rstrip()
            if prefix:
                return prefix, {
                    "type": "standalone_citation_paragraph",
                    "removed_text": tail,
                    "removed_citation_numbers": _citation_numbers(tail),
                }

    cleaned, removed = strip_unheaded_bibliography(answer)
    if removed is not None:
        return cleaned, {
            "type": "unheaded_numbered_bibliography",
            "removed_text": removed,
            "removed_citation_numbers": _citation_numbers(removed),
        }
    return answer, None


def _expand_numeric_body(body: str) -> tuple[list[int], list[str]]:
    values: list[int] = []
    invalid_ranges: list[str] = []
    for part in re.split(r"\s*,\s*", body.strip()):
        if part.isdigit():
            values.append(int(part))
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match is None:
            invalid_ranges.append(part)
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if start > end:
            invalid_ranges.append(part)
            continue
        values.extend(range(start, end + 1))
    return values, invalid_ranges


def canonicalize_citations(answer: str, *, source_count: int) -> dict[str, Any]:
    """Return a citation-normalized scoring copy and an action log."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if source_count <= 0:
        raise ValueError("source_count must be positive")

    raw_answer = answer
    counters: Counter[str] = Counter()
    details: dict[str, list[Any]] = {
        "invalid_or_out_of_range_numbers": [],
        "invalid_ranges": [],
        "removed_reference_sections": [],
    }

    def record_removed_section(removed_section: dict[str, Any] | None) -> None:
        if removed_section is None:
            return
        counters["remove_trailing_reference_list"] += 1
        counters[f"remove_{removed_section['type']}"] += 1
        details["removed_reference_sections"].append(removed_section)

    answer, removed_section = _strip_trailing_reference_section(answer)
    record_removed_section(removed_section)

    def normalized_label(value: int) -> str:
        if not 1 <= value <= source_count:
            counters["remove_invalid_or_out_of_range_label"] += 1
            details["invalid_or_out_of_range_numbers"].append(value)
            return ""
        return f"[{value}]"

    def replace_named_list(match: re.Match[str]) -> str:
        counters["convert_named_source_list"] += 1
        return "".join(normalized_label(value) for value in _citation_numbers(match.group(0)))

    answer = PAREN_NAMED_SOURCE_LIST.sub(replace_named_list, answer)

    def replace_named_wrapper(match: re.Match[str]) -> str:
        value = next(int(group) for group in match.groups() if group is not None)
        counters["convert_named_source_label"] += 1
        return normalized_label(value)

    answer = NAMED_SOURCE_WRAPPER.sub(replace_named_wrapper, answer)

    def replace_plain_source_number(match: re.Match[str]) -> str:
        counters["convert_plain_source_number"] += 1
        return f"{match.group('label')} {normalized_label(int(match.group('number')))}".rstrip()

    # Keep the word "Source" because it can be a grammatical subject. Only
    # canonicalize its numeric label; deleting the word would change prose.
    answer = PLAIN_SOURCE_NUMBER.sub(replace_plain_source_number, answer)

    def replace_numeric(match: re.Match[str]) -> str:
        body = match.group("body")
        values, invalid_ranges = _expand_numeric_body(body)
        if "," in body:
            counters["split_combined_citation"] += 1
        if "-" in body:
            counters["expand_ranged_citation"] += 1
        if invalid_ranges:
            counters["remove_invalid_range"] += len(invalid_ranges)
            details["invalid_ranges"].extend(invalid_ranges)

        valid: list[int] = []
        seen: set[int] = set()
        for value in values:
            if not 1 <= value <= source_count:
                counters["remove_invalid_or_out_of_range_label"] += 1
                details["invalid_or_out_of_range_numbers"].append(value)
            elif value in seen:
                counters["remove_duplicate_label"] += 1
            else:
                seen.add(value)
                valid.append(value)
        return "".join(f"[{value}]" for value in valid)

    answer = NUMERIC_BRACKET.sub(replace_numeric, answer)

    def compact_cluster(match: re.Match[str]) -> str:
        cluster = match.group("cluster")
        values = [int(value) for value in re.findall(r"\d+", cluster)]
        unique: list[int] = []
        seen: set[int] = set()
        for value in values:
            if value in seen:
                counters["remove_duplicate_label"] += 1
            else:
                seen.add(value)
                unique.append(value)
        canonical = "".join(f"[{value}]" for value in unique)
        if canonical != cluster:
            counters["compact_separate_citation_cluster"] += 1
        if match.start() > 0:
            previous = match.string[match.start() - 1]
            if previous.isalnum() or previous in ")】}”’\"'":
                canonical = " " + canonical
                counters["insert_space_before_citation_cluster"] += 1
        return canonical

    def compact_all_clusters(value: str) -> str:
        return CANONICAL_CLUSTER.sub(compact_cluster, value)

    answer = compact_all_clusters(answer)

    def move_post_punctuation(match: re.Match[str]) -> str:
        counters["move_post_punctuation_citation"] += 1
        return (
            f" {match.group('cluster')}"
            f"{match.group('punctuation')}{match.group('closers')}"
        )

    previous = None
    while answer != previous:
        previous = answer
        answer = POST_PUNCTUATION_CLUSTER.sub(move_post_punctuation, answer)

    # Moving a citation can create a newly adjacent cluster such as
    # ``[1] [2]``. Compact it in the same pass so the operation is idempotent.
    answer = compact_all_clusters(answer)

    def remove_space_before_punctuation(match: re.Match[str]) -> str:
        counters["remove_space_before_final_punctuation"] += 1
        return match.group("cluster") + match.group("punctuation")

    answer = SPACE_BEFORE_PUNCTUATION.sub(remove_space_before_punctuation, answer)

    if counters["remove_invalid_or_out_of_range_label"]:
        answer, removed_space_count = ORPHANED_SPACE_BEFORE_PUNCTUATION.subn(
            "", answer
        )
        if removed_space_count:
            counters["remove_orphaned_space_before_punctuation"] += (
                removed_space_count
            )

    # Some model-written reference lists only become citation-only paragraphs
    # after labels such as ``Source 1`` have been normalized to ``[1]``.
    answer, removed_section = _strip_trailing_reference_section(answer)
    record_removed_section(removed_section)
    answer = answer.rstrip()

    unresolved: list[dict[str, str]] = []
    for match in MALFORMED_NUMERIC_BRACKET.finditer(answer):
        if CANONICAL_TOKEN.fullmatch(match.group(0)) is None:
            unresolved.append(
                {"type": "malformed_numeric_bracket", "text": match.group(0)}
            )
    for match in LEFTOVER_NAMED_SOURCE.finditer(answer):
        unresolved.append({"type": "leftover_named_source", "text": match.group(0)})
    for line in answer.splitlines():
        if REFERENCE_HEADING_LINE.fullmatch(line.strip()):
            unresolved.append(
                {"type": "leftover_reference_heading", "text": line.strip()}
            )

    return {
        "answer": answer,
        "changed": answer != raw_answer,
        "action_counts": dict(sorted(counters.items())),
        "details": {key: value for key, value in details.items() if value},
        "unresolved": unresolved,
        "semantic_support_checked": False,
        "raw_answer_preserved": True,
        "source_count": source_count,
    }
