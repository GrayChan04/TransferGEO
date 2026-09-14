# Paper-ready pilot table captions

Two tables, each with GEO-Bench and C-SEO Bench panels. The shared metric definition and aggregation procedure are in `paper/sections/pilot_findings.tex` (Eq. `eq:pilot-normalized-improvement`).

## Prompt-family pilot

Prompt-family pilot on GEO-Bench (200 instances) and C-SEO Bench (197 instances). Cells report objective and subjective normalized improvement scores as defined in Eq.~\ref{eq:pilot-normalized-improvement}; higher is better. Bold marks the highest mean among the nine optimization prompts for each benchmark, model, and metric, excluding Original and Neutral Rewrite.

## Literature-method pilot

Literature-method pilot on GEO-Bench and C-SEO Bench. Each method uses 200 instances per benchmark, except IF-GEO, which uses 192 and 197 successful rewrites, respectively, with matched Original scores. Cells report normalized improvement scores (Eq.~\ref{eq:pilot-normalized-improvement}). Bold marks the highest mean among the five fixed methods. Multi-step pipelines are reported separately; missing rewrites are not assigned zero.
