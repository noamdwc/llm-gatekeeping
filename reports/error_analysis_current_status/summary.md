# Error Analysis Summary

## Headline Findings

- Current main test metrics recomputed from `research_test.parquet`: accuracy=0.9580, FPR=0.1333, FNR=0.0212.
- Current combined external metrics recomputed from current external research parquets: accuracy=0.7672, FPR=0.1171, FNR=0.4486.
- Main routing is concentrated in `ml` and `deberta`; external benign false positives are concentrated in the non-ML routes, especially where DeBERTa finalizes early or where abstain/risk still outputs adversarial on benign OOD prompts.
- Historical comparison caveat: Old per-sample main-hybrid artifacts were not found in the repository. Historical old/new comparisons in this notebook therefore use: (a) exact current row-level artifacts, (b) legacy external prediction files when present, and (c) the user-supplied historical headline metrics as reference-only context.

## Top Plots Produced

- `confusion_test.png`
- `routes_test.png`
- `external_fp_composition.png`
- `threshold_sweep_test.png`
- `threshold_sweep_external_combined.png`

## Exact Recommended Next Experiments

- Tighten or band-limit the DeBERTa benign/adversarial fast path for OOD-looking prompts; many external benign FPs are finalized before the LLM can disagree.
- Preserve the DeBERTa path for the main set, because it is carrying a large block of correct decisions that would otherwise hit a worse LLM-only fallback.
- Audit abstain-to-benign rescues separately from direct DeBERTa fast-path decisions; they should likely have a stricter OOD guard than the in-domain threshold.
- Use the threshold sweep table to run at least three follow-up experiments around DeBERTa confidence 0.70: balanced, more benign-friendly, and more attack-catching operating points.
- Add a small OOD guard feature set for benign-looking external prompts: prompt length, policy-like wording, markdown/list formatting, and role/instruction heavy phrasing are already available as cheap features in this notebook.
- If the next router experiment keeps DeBERTa enabled, route borderline high-confidence external-looking prompts back to LLM rather than finalizing them directly.
