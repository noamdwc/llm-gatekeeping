# Speaker Notes (8-12 min total)

## Slide 1 — Title (0:30)
Talk track:
- "This is a prompt-security gatekeeper that runs before an application LLM."
- "The repo now has a full 4-tier hybrid pipeline: classic ML, DeBERTa binary gate, LLM escalation, and a risk model for abstain resolution."

Evidence paths:
- `README.md`
- `PRD.md`

## Slide 2 — The Problem (0:35)
Talk track:
- "The core prediction is adversarial vs benign, but the pipeline also keeps attack taxonomy for downstream analysis."
- "False negatives are security failures; false positives damage the product experience."

Evidence paths:
- `PRD.md`
- `src/evaluate.py`

## Slide 3 — Why This Is Hard (0:45)
Talk track:
- "Attackers intentionally perturb prompts, and the repo handles both unicode-heavy and NLP-style attacks."
- "The dataset is imbalanced, and the hardest problem is distribution shift across external benchmarks."
- "A real solution also has to manage latency and cost, not just benchmark accuracy."

Evidence paths:
- `README.md`
- `project_plan.md`
- `reports/research_external/research_external_deepset.md`
- `reports/research_external/research_external_safeguard.md`

## Slide 4 — Current Pipeline (0:55)
Talk track:
- "The pipeline starts with Mindgard adversarial data plus synthetic benign generation, grouped into train/val/test splits."
- "Routing is 4-tier: ML catches high-confidence unicode attacks first (46% of traffic). DeBERTa handles binary classification for the rest (37%). Only 6% of samples actually need an LLM API call. The remaining 11% are abstains resolved by a post-hoc risk model."
- "This 4-tier design reduced LLM API calls by 93%."

Evidence paths:
- `dvc.yaml`
- `configs/default.yaml`
- `src/hybrid_router.py`
- `interview_presentation/assets/pipeline_diagram.svg`

## Slide 5 — Data Construction and Coverage (0:50)
Talk track:
- "Benign data now includes synthetic benign augmentation — validated through heuristic, judge, and dedup filters."
- "The test split has 300 benign samples now, up from 137 before synthetic augmentation."
- "Splits are still grouped by `prompt_hash`, and held-out attack types support unseen-attack testing."

Evidence paths:
- `src/preprocess.py`
- `src/build_splits.py`
- `configs/default.yaml`
- `interview_presentation/assets/data_coverage.csv`

## Slide 6 — Modeling Paths (0:45)
Talk track:
- "The ML baseline is still sparse features plus logistic regression — a useful specialist for unicode attacks."
- "DeBERTa is the big change: it's now the primary binary gate in the hybrid router, not just a standalone model."
- "The LLM path is only for genuinely uncertain samples. And the risk model resolves abstain cases using trace features."

Evidence paths:
- `src/ml_classifier/ml_baseline.py`
- `src/models/deberta_classifier.py`
- `src/hybrid_router.py`
- `src/llm_classifier/llm_classifier.py`

## Slide 7 — DeBERTa as Hybrid Gate (0:50)
Talk track:
- "DeBERTa standalone gets 86% accuracy and F1 of 0.9061, but the key number is 98.67% benign recall."
- "At a confidence threshold of 0.93, high-confidence predictions are finalized without calling the LLM."
- "This integration reduced LLM API calls from ~871 to 91 on the test set — a 93% cost reduction."
- "ROC-AUC is 0.989, so the model's probability ranking is excellent."

Evidence paths:
- `reports/deberta_classifier/summary.md`
- `src/hybrid_router.py`
- `configs/default.yaml`
- `artifacts/deberta_classifier/best_checkpoint.json`

## Slide 8 — Main Results (0:55)
Talk track:
- "The headline: hybrid accuracy is now 95.8%, with FNR of only 2.1%."
- "That's a big jump from the previous 84% accuracy and 14% FNR."
- "ML is still strongest in its specialist scope — 98.9% accuracy on unicode + benign."
- "LLM alone only gets 80.7% — the pipeline is much more than its LLM component."

Evidence paths:
- `reports/research/summary_report.md`
- `reports/research/eval_report_hybrid.md`
- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_llm.md`

## Slide 9 — External Generalization (0:50)
Talk track:
- "External generalization is still the main risk, but it's no longer uniformly bad."
- "jackhhao improved to 88% accuracy with only 9% FNR — the pipeline generalizes to these attack styles."
- "deepset has a 43% FPR problem — too many benign prompts flagged as adversarial."
- "safeguard has a 53% FNR problem — many threat/coercion-style attacks are missed entirely."

Evidence paths:
- `reports/research/summary_report.md`
- `reports/research_external/research_external_deepset.md`
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`

## Slide 10 — External Benchmark Caveats (0:55)
Talk track:
- "The repo now includes a benchmark-overlap audit, which changes how I interpret external numbers."
- "`deepset` is the cleanest external benchmark."
- "`jackhhao` is partially contaminated, and `safeguard` is clearly not a clean unseen benchmark because of Mindgard-family overlap."

Evidence paths:
- `docs/2026-03-13_baseline_dataset_overlap.md`
- `configs/default.yaml`

## Slide 11 — Baseline Comparison (0:55)
Talk track:
- "The repo compares against Sentinel v2 and ProtectAI v2."
- "On external datasets, public baselines are much stronger — Sentinel v2 gets 99.8% on safeguard, 87.9% on deepset."
- "But on our test split, the baselines break: Sentinel v2 has 98% FPR — it classifies almost everything as adversarial."
- "That tells us something important: our benign distribution is unusual, and calibration is domain-specific. Our hybrid is the only model with balanced performance on our test set."

Evidence paths:
- `reports/research/summary_report.md`
- `docs/2026-03-13_baseline_dataset_overlap.md`

## Slide 12 — Error Pattern (0:45)
Talk track:
- "The dominant error is now FPR at 13.3% — 40 of 300 benign samples misclassified."
- "FNR is very low at 2.1% — we're catching 98% of attacks."
- "The clean-benign FPR is 0.0% — all validated synthetic benigns are correct. The remaining FP errors are on harder, ambiguous benign samples."
- "180 samples go through the risk model abstain path."

Evidence paths:
- `reports/research/eval_report_hybrid.md`
- `data/processed/research/research_test.parquet`

## Slide 13 — Production View (0:40)
Talk track:
- "The serving shape is concrete now: ML is instant, DeBERTa adds ~10ms, and only 6% of traffic needs an API call."
- "That's 93% cost reduction compared to routing everything through the LLM."
- "The risk model handles abstain cases with near-zero additional latency."
- "Still missing: production API, load testing, and human audit workflow."

Evidence paths:
- `src/cli/predict.py`
- `src/hybrid_router.py`
- `reports/research/eval_report_hybrid.md`

## Slide 14 — What Changed Recently (0:35)
Talk track:
- "The biggest change is the DeBERTa binary gate — it transformed the hybrid router from an ML+LLM two-stage into a 4-tier pipeline."
- "The risk model for abstain resolution is a novel addition."
- "Hybrid accuracy went from 84% to 95.8%, and LLM costs dropped 93%."
- "I also added an autoresearch framework for automated threshold optimization."

Evidence paths:
- `git log --oneline`
- `configs/default.yaml`
- `autoresearch/program.md`

## Slide 15 — Next Steps (0:35)
Talk track:
- "The top priorities are external generalization: safeguard FNR at 53% and deepset FPR at 43%."
- "Risk model threshold tuning and the autoresearch optimization loop should help."
- "Need to decide whether DeBERTa or a public baseline should anchor the low-latency path."
- "Then production evaluation: latency under load, cost modeling, and policy behavior."

Evidence paths:
- `reports/research/summary_report.md`
- `autoresearch/program.md`

## Slide 16 — Appendix (0:20)
Talk track:
- "The core interview claims are tied to current repo artifacts, not hand-maintained slide text."
- "All custom visuals in this deck are generated from repository artifacts by `make_plots.py`."

Evidence paths:
- `reports/research/summary_report.md`
- `reports/deberta_classifier/summary.md`
- `docs/2026-03-13_baseline_dataset_overlap.md`
- `interview_presentation/make_plots.py`
