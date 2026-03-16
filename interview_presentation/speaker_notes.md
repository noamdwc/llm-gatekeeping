# Speaker Notes (8-12 min total)

## Slide 1 — Title (0:30)
Talk track:
- "This is a prompt-security gatekeeper that runs before an application LLM."
- "The current repo is broader than the original deck: it now includes classic ML, LLM/hybrid routing, a DeBERTa path, and baseline benchmarking."

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
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`

## Slide 4 — Current Pipeline (0:55)
Talk track:
- "The repo now starts with Mindgard adversarial data plus optional synthetic benign generation before preprocessing."
- "After grouped splits, there are three model paths: TF-IDF ML, DeBERTa fine-tuning, and LLM/hybrid."
- "Everything is tracked as DVC stages and report artifacts."

Evidence paths:
- `dvc.yaml`
- `dvc.lock`
- `configs/default.yaml`
- `interview_presentation/assets/pipeline_diagram.svg`
- `interview_presentation/make_plots.py`

## Slide 5 — Data Construction and Coverage (0:50)
Talk track:
- "Benign data is no longer just original-prompt recovery; synthetic benign augmentation is configured into preprocessing."
- "Splits are grouped by `prompt_hash`, and held-out attack types still support unseen-attack testing."
- "External datasets differ a lot in size and balance, so aggregate metrics can hide failure modes."

Evidence paths:
- `src/preprocess.py`
- `src/build_splits.py`
- `configs/default.yaml`
- `interview_presentation/assets/data_coverage.csv`
- `interview_presentation/assets/data_coverage.png`

## Slide 6 — Modeling Paths (0:45)
Talk track:
- "The ML baseline is still sparse features plus logistic regression and remains a useful specialist."
- "The LLM path remains classifier plus judge, with hybrid routing on top."
- "The notable change is that DeBERTa is now a first-party trained model path in the repo."

Evidence paths:
- `src/ml_classifier/ml_baseline.py`
- `src/llm_classifier/llm_classifier.py`
- `src/hybrid_router.py`
- `src/cli/deberta_classifier.py`

## Slide 7 — DeBERTa Update (0:50)
Talk track:
- "Recent changes made DeBERTa much more serious: class-weighted loss, longer training, best-checkpoint persistence, and WandB logging."
- "The checked-in best checkpoint metadata shows the current best model at epoch 2 with F1 around 0.971."
- "I present this as a current capability addition, not as part of the canonical summary-report comparison yet."

Evidence paths:
- `src/models/deberta_classifier.py`
- `src/cli/deberta_classifier.py`
- `configs/default.yaml`
- `artifacts/deberta_classifier/best_checkpoint.json`
- `interview_presentation/assets/deberta_summary.png`

## Slide 8 — Main Results (0:55)
Talk track:
- "These numbers come from the canonical `reports/research/*` flow."
- "ML is still strongest in its specialist scope; hybrid is much better than LLM-only on attack recall."
- "But hybrid benign recall is still weak enough that routing/calibration remains an open problem."

Evidence paths:
- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_hybrid.md`
- `reports/research/eval_report_llm.md`
- `reports/research/summary_report.md`
- `interview_presentation/assets/main_metrics.csv`
- `interview_presentation/assets/main_metrics_comparison.png`

## Slide 9 — External Generalization (0:50)
Talk track:
- "The project’s central weakness is still external generalization."
- "Even though the current reports are better than the old deck snapshot, adversarial recall still collapses on external sets."
- "That makes external robustness the most important model-quality target."

Evidence paths:
- `reports/research/summary_report.md`
- `reports/research_external/research_external_deepset.md`
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`
- `interview_presentation/assets/external_metrics.csv`
- `interview_presentation/assets/external_generalization.png`

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
- "The repo now compares against Sentinel v2 and ProtectAI v2."
- "On checked-in artifacts, our hybrid is stronger on the main test split, but both public baselines are clearly stronger on `deepset`."
- "That is useful because it gives us a target, but the overlap caveats mean benchmark interpretation still needs care."

Evidence paths:
- `reports/research/summary_report.md`
- `docs/2026-03-13_baseline_dataset_overlap.md`
- `interview_presentation/assets/baseline_comparison.csv`
- `interview_presentation/assets/baseline_comparison.png`

## Slide 12 — Error Pattern (0:45)
Talk track:
- "On the main test split, the hybrid confusion structure shows the remaining cost of current routing."
- "The good part is that the repo already has the right merged artifacts and report plumbing for deeper diagnosis."

Evidence paths:
- `data/processed/research/research_test.parquet`
- `interview_presentation/assets/hybrid_confusion_matrix.png`
- `reports/research/summary_report.md`

## Slide 13 — Production View (0:40)
Talk track:
- "The likely deployment shape is still cheap first-stage classification plus selective escalation."
- "The research pipeline already measures many things production would care about, but there is no full API/policy stack in the repo yet."

Evidence paths:
- `PRD.md`
- `STATUS.md`
- `src/cli/predict.py`
- `src/hybrid_router.py`

## Slide 14 — What Changed Recently (0:35)
Talk track:
- "The delta from the old deck is meaningful: synthetic benign integration, DeBERTa, baseline comparisons, and overlap-aware reporting."
- "That shifts the project from a narrow prototype to a better-instrumented research platform."

Evidence paths:
- `git log --oneline`
- `configs/default.yaml`
- `dvc.yaml`
- `docs/2026-03-13_baseline_dataset_overlap.md`

## Slide 15 — Next Steps (0:35)
Talk track:
- "My next priorities would be improving benign realism, optimizing for the cleanest external benchmark first, and deciding whether DeBERTa or a public baseline should anchor the low-latency path."
- "After that, I would add latency/cost/policy evaluation and production interfaces."

Evidence paths:
- `STATUS.md`
- `reports/research/summary_report.md`
- `artifacts/deberta_classifier/best_checkpoint.json`

## Slide 16 — Appendix (0:20)
Talk track:
- "The core interview claims are tied to current repo artifacts, not hand-maintained slide text."
- "All custom visuals in this deck are generated from repository artifacts by `make_plots.py`."

Evidence paths:
- `reports/research/summary_report.md`
- `reports/research/`
- `docs/2026-03-13_baseline_dataset_overlap.md`
- `interview_presentation/make_plots.py`
