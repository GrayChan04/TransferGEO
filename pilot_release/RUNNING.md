# Running the pilot code snapshot

Run commands from this directory. This is a code-and-configuration snapshot, not a complete dataset release. The two published tables and normalized-improvement formula are unchanged.

## What was added

Model loading and GLM rewriting, answer generation and shard aggregation, citation processing, objective scoring, subjective scoring and score merging, historical configurations, dependency declarations, and tests are included. Python sources retain historical output filenames, multilingual parsing rules and test fixtures for compatibility. Legacy analytical reports are not the current normalized-improvement tables.

`code/answer_prompt.py` is a release-specific copy: legacy official benchmark templates are omitted and their entry points raise an explicit error. Always pass `--answer-prompt-protocol unified_answer_v7`. The v7 template text and rendering are unchanged; old-version tests are not supported by this package.

Historical YAML configurations are serialized in JSON syntax (valid YAML). Commands and execution parameters are preserved. Narrative claim/notes fields are replaced with English release notes. Source-config hashes are recorded. In the judge configuration, an unquoted historical narrative contained a colon and was not valid YAML; it is omitted in the public copy, without changing scoring settings. The original local files are unchanged.

## External inputs you must supply

- Model directories listed in `code/model_loader.py`, beneath `data/raw/models/`, acquired under the applicable model terms.
- The frozen sample manifest, benchmark source files and realized rewrites at the paths named in the chosen configuration. They are not included here.
- Literature-method and RAID seven-dimension prompt files at their configured paths, from authorized sources. Their original text has not been changed or redistributed in this supplement.
- NLTK `punkt_tab` resources beneath `data/raw/nltk_data/`; the metric module records the expected version and archive hash. Installing the Python package alone is insufficient.
- Saved answer/score files if running repair, merging or final analysis instead of generating new answers.

Missing inputs must not be replaced by fabricated data or zeros. The source manifest and raw artifacts remain local; this is not a verified off-site backup of them.

## Environment

The snapshot was checked using the existing `transfergeo` Python 3.11 environment. Requirements are grouped in `requirements/hf.txt`, `requirements/data.txt`, and `requirements/evaluation.txt`; no installation is performed by this release. They describe the experiment stack, not a complete lockfile for every historical project-management tool. JSON-compatible historical configurations can be inspected with the standard library without installing YAML tooling.

## Safe checks without models or private inputs

```bash
sha256sum -c SHA256SUMS
python -m unittest discover -s tests -p 'test_public_package.py'
python -m unittest discover -s tests -p 'test_normalized_improvement_pilots.py'
python -m unittest discover -s tests -p 'test_json_format_repair.py'
python -m unittest discover -s tests -p 'test_source_content_truncation.py'
```

Do not use blanket test discovery as a self-contained package check. Other historical tests require local data, NLTK resources, omitted third-party prompts, or legacy answer protocols. Those tests are retained for users with the original inputs; their presence does not mean they all pass in this public snapshot.

## Generation and scoring entry points

Inspect each command before execution; `--help` does not load model weights:

```bash
python code/run_literature_method_rewrite_stage.py --help
python code/run_necessity_answer_model_shard.py --help
python code/run_raid_judge_shard.py --help
python code/normalized_improvement_pilots.py --help
```

The configurations in `configs/experiments/` are historical execution records, **not an automatic launch list**. They contain the original GPU assignments and output locations. Choose available devices and a new, empty output directory before launching; do not overwrite prior results. Do not automatically run every historical repair or continuation job.

For the final tables, the only normalized-improvement entry point is `code/normalized_improvement_pilots.py` with `configs/normalized_improvement_pilots_v1.json`. It requires the saved objective/subjective files and first-pilot selection file. Historical absolute-difference summaries and older relative-improvement fields are not substitutes for this calculation.

## Provenance and limitations

The GEO objective module records its source repository, audited commits and compatibility choices. These are provenance statements, not a new license for upstream materials. No blanket license is granted for omitted third-party prompts, benchmark text or model weights.

The literature objective wrapper historically ended with a failed status after saving its component scores; a subsequent independent verification checked all 28,701 records. Preserve both records. This supplement does not rewrite that history, fix every historical runner, or claim a fresh end-to-end run.

`SOURCE_PROVENANCE.json` records source hashes and differences for this supplement. These are current-source snapshots, not proof that every past run was made from a clean Git commit. No GPU generation or rescoring was performed to prepare this release.
