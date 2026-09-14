"""Recompute citation diagnostics for saved answers without changing them.

The command writes a separate post-hoc diagnostic file and summary.  It never
edits the input JSONL, retries a model, or substitutes a repaired answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from citation_validator import (
    validate_v2_citation_contract,
    validate_v4_optional_citation_contract,
)


KEY_FIELDS = (
    "experiment_id",
    "model",
    "sample_id",
    "benchmark",
    "domain",
    "prompt_id",
    "seed",
    "answer_prompt_protocol",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_count_from_row(row: dict[str, Any]) -> int:
    observed = {
        int(diagnostics["source_count"])
        for field in (
            "citation_diagnostics",
            "v2_citation_contract",
            "v4_citation_contract",
            "v6_citation_contract",
        )
        if isinstance((diagnostics := row.get(field)), dict)
        and diagnostics.get("source_count") is not None
    }
    if len(observed) != 1:
        raise ValueError(
            "Saved answer must contain one consistent source_count across its "
            f"stored diagnostics; observed={sorted(observed)}"
        )
    return observed.pop()


def recompute_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        answer = str(row.get("answer") or "")
        source_count = source_count_from_row(row)
        stored_v2 = row.get("v2_citation_contract")
        stored_v4 = row.get("v4_citation_contract")
        recomputed_v2 = validate_v2_citation_contract(
            answer,
            source_count=source_count,
        )
        recomputed_v4 = validate_v4_optional_citation_contract(
            answer,
            source_count=source_count,
        )
        output.append(
            {
                **{field: row.get(field) for field in KEY_FIELDS},
                "source_count": source_count,
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "stored_v2_citation_contract": stored_v2,
                "recomputed_v2_citation_contract": recomputed_v2,
                "stored_v4_citation_contract": stored_v4,
                "recomputed_v4_citation_contract": recomputed_v4,
                "v2_diagnostic_changed": stored_v2 != recomputed_v2,
                "v4_diagnostic_changed": stored_v4 != recomputed_v4,
                "input_answer_edited": False,
            }
        )
    return output


def record_key(row: dict[str, Any]) -> list[Any]:
    return [
        row.get("model"),
        row.get("sample_id"),
        row.get("prompt_id"),
        row.get("seed"),
    ]


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    input_path: Path,
    input_sha256: str,
) -> dict[str, Any]:
    stored_v4 = [row["stored_v4_citation_contract"] for row in rows]
    recomputed_v4 = [row["recomputed_v4_citation_contract"] for row in rows]
    changed_v4 = [row for row in rows if row["v4_diagnostic_changed"]]
    named_source = [
        row
        for row in rows
        if row["recomputed_v4_citation_contract"]["named_source_citations"]
    ]
    return {
        "input_answers_path": str(input_path),
        "input_answers_sha256": input_sha256,
        "answer_count": len(rows),
        "input_answers_edited": False,
        "model_generation_rerun": False,
        "stored_v4_contract_compliant_count": sum(
            bool(item.get("contract_compliant")) for item in stored_v4
        ),
        "recomputed_v4_contract_compliant_count": sum(
            bool(item.get("contract_compliant")) for item in recomputed_v4
        ),
        "stored_v4_citationless_answer_count": sum(
            bool(item.get("citationless_answer")) for item in stored_v4
        ),
        "recomputed_v4_citationless_answer_count": sum(
            bool(item.get("citationless_answer")) for item in recomputed_v4
        ),
        "recomputed_v4_malformed_citation_answer_count": sum(
            int(item.get("malformed_citation_sentence_count", 0)) > 0
            for item in recomputed_v4
        ),
        "named_source_form_answer_count": len(named_source),
        "v4_diagnostic_changed_count": len(changed_v4),
        "v4_diagnostic_changed_keys": [record_key(row) for row in changed_v4],
        "named_source_form_keys": [record_key(row) for row in named_source],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.answers.is_file():
        raise FileNotFoundError(args.answers)
    resolved = {
        args.answers.resolve(),
        args.output_jsonl.resolve(),
        args.metrics_json.resolve(),
    }
    if len(resolved) != 3:
        raise ValueError("Input and output paths must be distinct")
    for output in (args.output_jsonl, args.metrics_json):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rows = recompute_rows(read_jsonl(args.answers))
    input_sha256 = sha256_file(args.answers)
    summary = build_summary(
        rows,
        input_path=args.answers,
        input_sha256=input_sha256,
    )
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.metrics_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
