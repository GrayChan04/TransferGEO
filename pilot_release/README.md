# TransferGEO pilot results

This snapshot contains the two pilot tables, manuscript findings, numerical summaries, original prompt-family templates (v12), the shared answer-generation prompt (v7), and the shared statistical implementation. It is a reporting snapshot, not an end-to-end generation release.

## Contents

- `paper/sections/pilot_findings.tex`: findings, reporting equation, and limitations.
- `paper/tables/`: exactly two main LaTeX tables and their captions.
- `results/summary.csv`: 114 model/benchmark/method rows with objective and subjective normalized improvement scores. Sample counts remain in machine-readable data, not as a table column.
- `results/best_methods.csv` and `results/ranking_reversals.csv`: descriptive comparisons.
- `prompts/necessity_pilot/v12/`: ten templates (nine optimization variants and Neutral Rewrite).
- `prompts/answer_generation/unified_v7/`: exact system and user templates used for both benchmarks.
- `code/`: the common paired-score analysis implementation and its helper.
- `tests/`: tests of the common computation.
- `SHA256SUMS`: checksums for this snapshot, excluding the checksum file itself.

## Reporting rule

For each instance and answer seed, compute `10 * (optimized - original) / (original + 1)`. Average seeds 0, 1, and 2 within each instance, then average instances within a benchmark. These are normalized improvement scores, **not percentages**. Original zeros are retained. Objective base scores range from 0 to 1; subjective base scores range from 0 to 5. Their magnitudes are not directly comparable.

The prompt-family table uses a common complete subset: GEO-Bench 200 instances and C-SEO Bench 197. The literature table uses available paired instances: normally 200 per benchmark; IF-GEO has 192 GEO-Bench and 197 C-SEO instances. Missing generations are not treated as zero. Negative results and controls are retained.

## Experimental context

The answer models are Qwen3.5-9B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3. Both pilots use the v7 answer protocol, three answer seeds (0/1/2), temperature 0.7, top-p 0.9, a 2,048-token output limit, and a 20,480-token input budget. The shared rewriter checkpoint is GLM-4-9B-0414, without access to the user query. The prompt-family catalog records deterministic rewriting. Literature methods retain their distinct procedures; this snapshot does not claim that their internal generation settings are identical.

Subjective scoring uses GLM-4-9B-0414 in a separate judge role and averages seven RAID dimensions. Absence of the target source's valid separate bracketed identifier leads to zero on all seven dimensions; this is not a semantic verification of citation correctness. Final scoring uses citation-cleaned answer copies, not unprocessed generation outputs.

## Interpretation

Some method orderings reverse across answer models. However, Key-Point Enumeration is the shared best among the nine prompt-family variants, and LLM Guidance is the shared best among the five fixed literature methods. The results motivate investigating adaptation of a given strategy; they do not establish that transfer is necessary or effective. See the manuscript section for sample-coverage, truncation, decoding, and normalization limitations.

## Checking and reuse

From this directory:

```bash
sha256sum -c SHA256SUMS
python -m unittest discover -s tests -p 'test_*.py'
```

The tests require only Python's standard library and do not load a model. Include the findings section and both table files in your LaTeX manuscript using `booktabs`, `graphicx`, and `amsmath`. No PDF compilation is claimed for this snapshot.

The analysis modules require paired per-answer objective/subjective score files and a selection configuration to regenerate results; these files are not included here. This package is therefore not sufficient to reproduce generation or scoring from scratch. Raw benchmark text, answers, large scoring outputs, model weights, PDFs, private records, and third-party method/judge prompt copies pending redistribution review remain local. The original local evidence is preserved.
