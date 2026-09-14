"""Run one checkpointable shard of the frozen RAID subjective judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from model_loader import generate_answer, load_model, release_model


ROOT = Path(__file__).resolve().parents[1]
EXACT_SCORE = re.compile(r"^\s*(?:\[([0-5])\]|([0-5]))\s*$")
LABELED_SCORE = re.compile(
    r"(?i)(?:final\s+)?(?:score|rating|answer|评分|得分)"
    r"(?:\s*\(\s*0\s*[-–]\s*5\s*\))?"
    r"\s*(?:is|为|[:=：])?\s*\[?\s*([0-5])\s*\]?"
)
FINAL_LABELED_SCORE = re.compile(
    r"(?i)(?:final\s+(?:score|rating|answer)|最终(?:评分|得分|答案))"
    r"\s*(?:is|为|[:=：])?\s*\[?\s*([0-5])\s*\]?"
)
RATIO_SCORE = re.compile(r"(?i)(?<!\d)([0-5])\s*(?:/|out\s+of)\s*5(?!\d)")
LEADING_SCORE = re.compile(
    r"^\s*(?:\[([0-5])\]|([0-5]))(?=\s|[.,;:!?)\-–—]|$)"
)
CONCLUDING_SCORE = re.compile(
    r"(?i)(?:therefore|thus|hence|so|最终|因此|故)\s*"
    r"(?:(?:the\s+)?(?:score|rating|answer)\s*)?"
    r"(?:is|为|[:=：])?\s*\[?\s*([0-5])\s*\]?"
)
LAST_LINE_SCORE = re.compile(
    r"(?:^|\n)\s*(?:\[([0-5])\]|([0-5]))\s*[.!。！]?\s*$"
)
AMBIGUITY_CUE = re.compile(r"(?i)\b(?:or|either|could|maybe|between)\b|或|可能")
ANY_SCORE_DIGIT = re.compile(r"(?<!\d)([0-5])(?!\d)")
CITATION_INDEX = re.compile(r"\[(\d+)\]")
DIMENSIONS = (
    "diversity_detailed",
    "follow_detailed",
    "influence_detailed",
    "relevance_detailed",
    "subjcount_detailed",
    "subjpos_detailed",
    "uniqueness_detailed",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answer_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["seed"]),
    )


def trial_key(row: dict[str, Any]) -> tuple[str, str, str, int, str, int]:
    return (
        str(row["model"]),
        str(row["sample_id"]),
        str(row["prompt_id"]),
        int(row["answer_seed"]),
        str(row["dimension"]),
        int(row["judge_seed"]),
    )


def shard_for(row: dict[str, Any], num_shards: int) -> int:
    canonical = json.dumps(answer_key(row), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def select_shard_answers(
    all_answers: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
    limit_answers: int | None = None,
) -> list[dict[str, Any]]:
    """Apply the one authoritative deterministic shard/diagnostic selection."""

    selected = [
        row for row in all_answers if shard_for(row, num_shards) == shard_index
    ]
    if limit_answers is not None:
        selected = selected[:limit_answers]
    return selected


def unique_score(values: list[str]) -> int | None:
    parsed = {int(value) for value in values}
    return next(iter(parsed)) if len(parsed) == 1 else None


def parse_score_with_method(raw: str) -> tuple[int | None, str]:
    """Extract one unambiguous 0-5 score while recording how it was found."""

    exact = EXACT_SCORE.match(raw)
    if exact:
        method = "exact_bracketed" if exact.group(1) is not None else "exact_bare"
        return int(exact.group(1) or exact.group(2)), method

    final_labeled_values = FINAL_LABELED_SCORE.findall(raw)
    if final_labeled_values:
        score = unique_score(final_labeled_values)
        return (
            score,
            "final_labeled" if score is not None else "ambiguous_final_labeled",
        )

    labeled_values = LABELED_SCORE.findall(raw)
    if labeled_values:
        if AMBIGUITY_CUE.search(raw):
            return None, "ambiguous_language"
        score = unique_score(labeled_values)
        return score, "labeled" if score is not None else "ambiguous_labeled"

    ratio_values = RATIO_SCORE.findall(raw)
    if ratio_values:
        score = unique_score(ratio_values)
        return score, "ratio" if score is not None else "ambiguous_ratio"

    concluding_values = CONCLUDING_SCORE.findall(raw)
    if concluding_values:
        score = unique_score(concluding_values)
        return score, "concluding" if score is not None else "ambiguous_concluding"

    last_line = LAST_LINE_SCORE.search(raw)
    if last_line:
        return int(last_line.group(1) or last_line.group(2)), "last_line"

    if AMBIGUITY_CUE.search(raw):
        return None, "ambiguous_language"

    leading = LEADING_SCORE.search(raw)
    if leading:
        return int(leading.group(1) or leading.group(2)), "leading"

    score = unique_score(ANY_SCORE_DIGIT.findall(raw))
    if score is not None:
        return score, "unique_numeric_candidate"
    return None, "ambiguous_or_missing"


def parse_score(raw: str) -> int | None:
    return parse_score_with_method(raw)[0]


def extract_citation_indices(answer_text: str) -> set[int]:
    """Return indices written in the answer protocol's exact ``[index]`` form."""

    return {int(value) for value in CITATION_INDEX.findall(answer_text)}


def target_source_is_cited(
    answer: dict[str, Any], sample_metadata: dict[str, Any]
) -> bool:
    """Whether the generated answer visibly cites the frozen target source."""

    answer_text = answer.get("answer")
    if not isinstance(answer_text, str):
        raise ValueError(
            f"Missing answer text for sample_id={answer.get('sample_id')}"
        )
    target_citation_index = int(sample_metadata["target_citation_index"])
    return target_citation_index in extract_citation_indices(answer_text)


def load_target_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    target_map: dict[str, dict[str, Any]] = {}
    for row in manifest:
        sample_id = str(row["sample_id"])
        if sample_id in target_map:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing or empty query in manifest: {sample_id}")
        target_document_index = int(row["target_document_index"])
        candidate_count = int(row["candidate_document_count"])
        if not 0 <= target_document_index < candidate_count:
            raise ValueError(
                f"Invalid target_document_index={target_document_index} for "
                f"{sample_id} with {candidate_count} candidates"
            )
        target_map[sample_id] = {
            "query": query,
            "target_document_index": target_document_index,
            "target_citation_index": target_document_index + 1,
            "candidate_document_count": candidate_count,
        }
    return target_map


def resolve_answer_query(
    answer: dict[str, Any], sample_metadata: dict[str, Any]
) -> tuple[str, str]:
    """Use a verified answer query, or recover it from the frozen manifest."""

    sample_id = str(answer["sample_id"])
    manifest_query = sample_metadata.get("query")
    if not isinstance(manifest_query, str) or not manifest_query.strip():
        raise ValueError(f"Missing or empty manifest query for sample_id={sample_id}")

    answer_query = answer.get("query")
    if answer_query is None:
        return manifest_query, "manifest_fallback"
    if not isinstance(answer_query, str) or not answer_query.strip():
        raise ValueError(f"Empty answer query for sample_id={sample_id}")
    if answer_query != manifest_query:
        raise ValueError(
            f"Query mismatch for sample_id={sample_id}: answer and manifest differ"
        )
    return answer_query, "answer_verified"


def render_targeted_prompt(
    template: str,
    *,
    query: str,
    answer: str,
    target_citation_index: int,
) -> str:
    """Retarget the archived RAID Source [1] rubric without editing its file."""

    if target_citation_index <= 0:
        raise ValueError("target_citation_index must be positive")
    source_mentions = template.count("Source [1]")
    if source_mentions <= 0:
        raise ValueError("Archived RAID prompt does not contain Source [1]")
    targeted = template.replace(
        "Source [1]", f"Source [{target_citation_index}]"
    )
    rendered = targeted.format(query=query, answer=answer)
    if "{query}" in rendered or "{answer}" in rendered:
        raise ValueError("RAID prompt placeholders were not fully rendered")
    return rendered


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-root", type=Path, default=ROOT / "prompts/evaluation/raid"
    )
    parser.add_argument(
        "--model-root", type=Path, default=ROOT / "data/raw/models"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--physical-gpus", nargs="+", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--limit-answers",
        type=int,
        default=None,
        help="Diagnostic-only limit applied after deterministic shard assignment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Judge seeds must be unique: {args.seeds}")
    if args.limit_answers is not None and args.limit_answers <= 0:
        raise ValueError("limit-answers must be positive")
    if args.physical_gpus is not None and len(args.physical_gpus) != len(
        set(args.physical_gpus)
    ):
        raise ValueError("physical-gpus must be unique")

    all_answers = read_jsonl(args.input)
    keys = [answer_key(row) for row in all_answers]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate Answer keys: {len(keys) - len(set(keys))}")
    target_map = load_target_map(args.manifest)
    missing_manifest_samples = sorted(
        {str(row["sample_id"]) for row in all_answers} - set(target_map)
    )
    if missing_manifest_samples:
        raise ValueError(
            f"Answer input contains {len(missing_manifest_samples)} samples missing "
            "from manifest"
        )
    answers = select_shard_answers(
        all_answers,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit_answers=args.limit_answers,
    )
    resolved_queries: dict[tuple[str, str, str, int], str] = {}
    target_cited_by_answer_key: dict[tuple[str, str, str, int], bool] = {}
    query_source_counts = {"answer_verified": 0, "manifest_fallback": 0}
    for answer in answers:
        sample_metadata = target_map[str(answer["sample_id"])]
        query, source = resolve_answer_query(answer, sample_metadata)
        resolved_queries[answer_key(answer)] = query
        target_cited_by_answer_key[answer_key(answer)] = target_source_is_cited(
            answer, sample_metadata
        )
        query_source_counts[source] += 1
    prompts = {
        name: (args.prompt_root / f"{name}.txt").read_text(encoding="utf-8")
        for name in DIMENSIONS
    }
    prompt_sha256 = {
        name: hashlib.sha256(template.encode("utf-8")).hexdigest()
        for name, template in prompts.items()
    }
    expected = len(answers) * len(DIMENSIONS) * len(args.seeds)
    target_cited_answer_count = sum(target_cited_by_answer_key.values())
    target_absent_answer_count = len(answers) - target_cited_answer_count
    trials_per_answer = len(DIMENSIONS) * len(args.seeds)
    plan = {
        "experiment_id": args.experiment_id,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "physical_gpus": args.physical_gpus,
        "all_answer_count": len(all_answers),
        "shard_answer_count": len(answers),
        "dimension_count": len(DIMENSIONS),
        "seeds": args.seeds,
        "expected_trials": expected,
        "checkpointed": True,
        "score_output_protocol": "extract one unambiguous integer score from 0-5",
        "score_normalization": "store extracted score as integer",
        "query_policy": "verify answer query; otherwise use frozen manifest query",
        "query_source_counts": query_source_counts,
        "target_source_policy": "manifest target_document_index + 1",
        "target_absence_policy": (
            "if exact [target citation index] is absent, assign 0 to all "
            "dimensions/seeds without calling GLM"
        ),
        "target_cited_answer_count": target_cited_answer_count,
        "target_absent_answer_count": target_absent_answer_count,
        "expected_glm_trials": target_cited_answer_count * trials_per_answer,
        "expected_deterministic_zero_trials": (
            target_absent_answer_count * trials_per_answer
        ),
        "selected_target_citation_indices": sorted(
            {
                target_map[str(row["sample_id"])]["target_citation_index"]
                for row in answers
            }
        ),
        "prompt_sha256": prompt_sha256,
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "trials.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    progress_path = args.output_dir / "progress.json"
    metrics_path = args.output_dir / "metrics.json"
    nonempty = [
        path
        for path in (trials_path, errors_path, progress_path, metrics_path)
        if path.exists() and path.stat().st_size
    ]
    if nonempty and not args.resume:
        raise FileExistsError(
            "Existing shard outputs require --resume: "
            + ", ".join(str(path) for path in nonempty)
        )
    existing_trials = read_jsonl(trials_path) if args.resume else []
    existing_errors = read_jsonl(errors_path) if args.resume else []
    trial_keys = [trial_key(row) for row in existing_trials]
    error_keys = [trial_key(row) for row in existing_errors]
    if len(trial_keys) != len(set(trial_keys)):
        raise ValueError("Duplicate trial keys in checkpoint")
    if len(error_keys) != len(set(error_keys)):
        raise ValueError("Duplicate error keys in checkpoint")
    trial_set = set(trial_keys)
    error_set = set(error_keys)
    if trial_set & error_set:
        raise ValueError("Trial keys overlap generation-error keys")
    accounted = trial_set | error_set
    parsed_score_count = sum(
        row.get("score") is not None for row in existing_trials
    )
    deterministic_zero_trial_count = sum(
        row.get("score_origin") == "deterministic_target_absence"
        for row in existing_trials
    )
    glm_judge_trial_count = sum(
        row.get("score_origin", "glm_judge") == "glm_judge"
        for row in existing_trials
    )

    expected_keys = {
        (*answer_key(row), dimension, judge_seed)
        for row in answers
        for dimension in DIMENSIONS
        for judge_seed in args.seeds
    }
    unexpected_checkpoint = accounted - expected_keys
    if unexpected_checkpoint:
        raise ValueError(
            f"Checkpoint contains {len(unexpected_checkpoint)} unexpected trial keys"
        )

    def progress_payload() -> dict[str, Any]:
        finished = len(accounted)
        return {
            **plan,
            "completed_trials": len(trial_set),
            "generation_error_count": len(error_set),
            "accounted_trials": finished,
            "remaining_trials": expected - finished,
            "percent_accounted": round(100.0 * finished / expected, 4)
            if expected
            else 100.0,
            "parsed_score_count": parsed_score_count,
            "unparsed_score_count": len(trial_set) - parsed_score_count,
            "glm_judge_trial_count": glm_judge_trial_count,
            "deterministic_zero_trial_count": deterministic_zero_trial_count,
        }

    atomic_json(progress_path, progress_payload())
    needs_glm = any(
        target_cited_by_answer_key[answer_key(answer)]
        and any(
            (*answer_key(answer), dimension, judge_seed) not in accounted
            for dimension in DIMENSIONS
            for judge_seed in args.seeds
        )
        for answer in answers
    )
    loaded = (
        load_model(
            "glm",
            model_root=args.model_root,
            dtype_name="bfloat16",
            attn_implementation="sdpa",
            device_map_strategy="balanced",
        )
        if needs_glm
        else None
    )
    new_trials = 0
    new_errors = 0
    try:
        with trials_path.open("a", encoding="utf-8") as trials_file, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_file:
            for answer in answers:
                sample_metadata = target_map[str(answer["sample_id"])]
                target_source_cited = target_cited_by_answer_key[answer_key(answer)]
                base = {
                    "model": answer["model"],
                    "sample_id": answer["sample_id"],
                    "prompt_id": answer["prompt_id"],
                    "answer_seed": int(answer["seed"]),
                    "benchmark": answer["benchmark"],
                    "domain": answer["domain"],
                    "target_document_index": sample_metadata[
                        "target_document_index"
                    ],
                    "target_citation_index": sample_metadata[
                        "target_citation_index"
                    ],
                    "candidate_document_count": sample_metadata[
                        "candidate_document_count"
                    ],
                    "target_source_cited": target_source_cited,
                }
                for dimension, template in prompts.items():
                    rendered = None
                    if target_source_cited:
                        rendered = render_targeted_prompt(
                            template,
                            query=resolved_queries[answer_key(answer)],
                            answer=answer["answer"],
                            target_citation_index=base["target_citation_index"],
                        )
                    for judge_seed in args.seeds:
                        key = (*answer_key(answer), dimension, judge_seed)
                        if key in accounted:
                            continue
                        try:
                            if not target_source_cited:
                                row = {
                                    **base,
                                    "dimension": dimension,
                                    "judge_seed": judge_seed,
                                    "raw_output": None,
                                    "answer": None,
                                    "score": 0,
                                    "score_parse_method": (
                                        "deterministic_target_absence"
                                    ),
                                    "score_origin": (
                                        "deterministic_target_absence"
                                    ),
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                }
                            else:
                                if loaded is None or rendered is None:
                                    raise RuntimeError(
                                        "GLM must be loaded for a cited target source"
                                    )
                                generation = generate_answer(
                                    loaded,
                                    [{"role": "user", "content": rendered}],
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    seed=judge_seed,
                                )
                                score_value, score_parse_method = (
                                    parse_score_with_method(generation.answer)
                                )
                                row = {
                                    **base,
                                    "dimension": dimension,
                                    "judge_seed": judge_seed,
                                    "raw_output": generation.raw_answer,
                                    "answer": generation.answer,
                                    "score": score_value,
                                    "score_parse_method": score_parse_method,
                                    "score_origin": "glm_judge",
                                    "input_tokens": generation.input_tokens,
                                    "output_tokens": generation.output_tokens,
                                }
                            trials_file.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                            trials_file.flush()
                            existing_trials.append(row)
                            trial_set.add(key)
                            accounted.add(key)
                            if row["score"] is not None:
                                parsed_score_count += 1
                            if row["score_origin"] == "deterministic_target_absence":
                                deterministic_zero_trial_count += 1
                            else:
                                glm_judge_trial_count += 1
                            new_trials += 1
                        except torch.cuda.OutOfMemoryError:
                            raise
                        except Exception as exc:
                            row = {
                                **base,
                                "dimension": dimension,
                                "judge_seed": judge_seed,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            errors_file.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                            errors_file.flush()
                            error_set.add(key)
                            accounted.add(key)
                            new_errors += 1
                        atomic_json(progress_path, progress_payload())
                        if len(accounted) % 100 == 0 or len(accounted) == expected:
                            print(
                                json.dumps(progress_payload(), ensure_ascii=False),
                                flush=True,
                            )
    finally:
        if loaded is not None:
            release_model(loaded)

    final_trials = read_jsonl(trials_path)
    parsed_count = sum(row.get("score") is not None for row in final_trials)
    metrics = {
        **progress_payload(),
        "new_trials_this_run": new_trials,
        "new_errors_this_run": new_errors,
        "parsed_score_count": parsed_count,
        "unparsed_score_count": len(final_trials) - parsed_count,
        "complete_accounting": len(accounted) == expected,
    }
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["complete_accounting"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
