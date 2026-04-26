# Deepset Style Gap Analysis Design

## Goal

Explain why the fine-tuned DeBERTa classifier misses many adversarial examples from `deepset/prompt-injections`, especially attacks predicted benign with high confidence, despite performing well on internal splits, Jackhhao, Safeguard, and related data.

The output is a one-off research report at `reports/research_external/deepset_style_gap_analysis.md`. This work must not change training code or create a maintained pipeline.

## Investigation Method

Use a dynamic research loop:

1. Inspect schemas and representative examples from each available dataset and prediction artifact.
2. Run quantitative comparisons.
3. Read representative examples from the strongest contrasts.
4. Form hypotheses about what makes Deepset different.
5. Test those hypotheses with more code and example inspection.
6. Repeat until the main explanation is clear and evidence-backed.

The initial hypotheses are not expected conclusions. The analysis must actively look for unexpected differences and discard weak explanations.

## Primary Comparison

The central group is:

- Deepset adversarial examples predicted benign by DeBERTa with high confidence.

Compare that group against:

- Deepset adversarial examples correctly caught by DeBERTa.
- Jackhhao adversarial examples.
- Safeguard adversarial examples.
- Mindgard-origin adversarial examples where source attribution is available.
- Synthetic/internal adversarial examples.
- Benign examples.

If source columns do not cleanly separate Mindgard-origin rows from synthetic or other internal rows, the report must say source attribution is uncertain.

## Evidence To Gather

For each major claim, include direct supporting evidence:

- Counts and rates.
- Representative examples.
- Keyword and phrase contrasts.
- Distinctive n-grams or lexical cues.
- Attack-style heuristics for instruction hierarchy attacks, prompt injection, roleplay, harmful requests, prompt extraction, tool/data exfiltration, and document/email/webpage injection.
- DeBERTa missed-vs-caught examples and confidence distributions.
- Embedding or nearest-neighbor checks if useful and feasible from local artifacts or installed dependencies.

The confidence analysis must report how many missed Deepset attacks are high-confidence benign, what they look like, and whether confidence appears useful for routing or misleading under this out-of-distribution style.

## Report Shape

The report should include:

- Executive summary.
- Dataset comparison table.
- Verdict section covering the main supported difference, secondary differences, rejected or weak hypotheses, recommended synthetic additions, and recommended decision-engine guardrail.
- Evidence-backed comparison of Deepset missed attacks against each reference group.
- Label-definition mismatch audit.
- Gaps in current synthetic/internal adversarial data.
- Concrete synthetic data recommendations.
- Concrete routing or decision-engine recommendations.

## Constraints

- Do not change training code.
- Temporary notebooks, scripts, or shell commands are allowed during analysis.
- Do not turn this into a maintained pipeline.
- The final report must go beyond metrics and inspect actual examples.
