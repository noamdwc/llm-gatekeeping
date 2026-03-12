# Speaker Notes (8-12 min total)

## Slide 1 — Title (0:30)
Talk track:
- "This project is a pre-LLM security gatekeeper: classify prompts before they hit the model."
- "The core value is balancing security recall with latency/cost by combining ML and LLM routing."

Evidence paths:
- `README.md`
- `PRD.md`

## Slide 2 — The Problem (0:40)
Talk track:
- "The system predicts adversarial vs benign prompts, then attack category/type for analysis."
- "In production, false negatives are security failures; false positives hurt product usability."

Evidence paths:
- `PRD.md`
- `src/evaluate.py`

## Slide 3 — Why It Is Hard (0:50)
Talk track:
- "Attackers intentionally obfuscate prompts, especially with unicode manipulation and NLP paraphrase attacks."
- "NLP sub-types are not reliably separable, while unicode sub-types are much cleaner."
- "OOD behavior across external datasets is the hardest challenge in this repo."

Evidence paths:
- `README.md`
- `project_plan.md`
- `reports/research_external/research_external_deepset.md`
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`
- `reports/research_external/research_external_spml.md`

## Slide 4 — High-Level Solution (0:55)
Talk track:
- "The DVC pipeline is preprocess -> grouped splits -> ML predictions -> optional LLM predictions -> hybrid merge -> reports."
- "The hybrid policy is threshold-gated and routes uncertain cases to the LLM path."

Evidence paths:
- `dvc.yaml`
- `src/research.py`
- `interview_presentation/assets/pipeline_diagram.svg`
- `interview_presentation/make_plots.py`

## Slide 5 — Data (0:55)
Talk track:
- "Main processed dataset has 11,795 rows with strong class imbalance toward adversarial."
- "Splits are grouped by `prompt_hash`, and `test_unseen` is held out by attack type."
- "External datasets vary heavily in size and class balance, which stresses generalization."

Evidence paths:
- `data/processed/full_dataset.parquet`
- `data/processed/splits/train.parquet`
- `data/processed/splits/val.parquet`
- `data/processed/splits/test.parquet`
- `data/processed/splits/test_unseen.parquet`
- `configs/default.yaml`
- `src/build_splits.py`
- `interview_presentation/assets/data_coverage.csv`
- `interview_presentation/assets/data_coverage.png`

## Slide 6 — Modeling (1:00)
Talk track:
- "ML uses character n-gram TF-IDF plus handcrafted unicode features with logistic regression heads."
- "LLM path is classifier+judge; hybrid ties them together with confidence routing."
- "One nuance: LLM CLI default limit is 100, so sample coverage must be called out in results."

Evidence paths:
- `src/ml_classifier/ml_baseline.py`
- `src/llm_classifier/llm_classifier.py`
- `src/hybrid_router.py`
- `configs/default.yaml`
- `interview_presentation/assets/modeling_diagram.png`

## Slide 7 — Training/Eval Design (0:50)
Talk track:
- "Benign data is synthesized from original prompts, then deduplicated and merged with adversarial samples."
- "Evaluation tracks binary metrics, FNR/FPR, calibration, and hierarchy-level quality."
- "Baselines are ML, Hybrid, and LLM with explicit scope differences."

Evidence paths:
- `src/preprocess.py`
- `src/build_splits.py`
- `src/evaluate.py`
- `src/cli/eval_new.py`
- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_hybrid.md`
- `reports/research/eval_report_llm.md`

## Slide 8 — Main Results (1:00)
Talk track:
- "In current canonical artifacts, ML is very strong in unicode-scope evaluation."
- "Hybrid and LLM artifacts show lower adversarial recall and materially higher FNR, so policy tuning is still open."
- "I explicitly call out non-identical evaluation scope and sample size per mode."

Evidence paths:
- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_hybrid.md`
- `reports/research/eval_report_llm.md`
- `interview_presentation/assets/main_metrics.csv`
- `interview_presentation/assets/main_metrics_comparison.png`

## Slide 9 — External Results (0:55)
Talk track:
- "Generalization on external datasets is the biggest weakness in this snapshot."
- "Across several datasets, adversarial recall collapses and FNR spikes."
- "This is where product risk is highest and where data strategy must improve first."

Evidence paths:
- `reports/research_external/research_external_deepset.md`
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`
- `reports/research_external/research_external_spml.md`
- `interview_presentation/assets/external_metrics.csv`
- `interview_presentation/assets/external_generalization.png`

## Slide 10 — Error Analysis (0:55)
Talk track:
- "On the main test split, confusion structure shows many adversarial prompts classified benign under hybrid outputs."
- "External reports include many high-confidence mistakes and calibration mismatch in top confidence bins."
- "Failure examples include both benign instructions flagged as adversarial and jailbreak prompts passed as benign."

Evidence paths:
- `data/processed/research/research_test.parquet`
- `interview_presentation/assets/hybrid_confusion_matrix.png`
- `reports/research_external/research_external_jackhhao.md`
- `reports/research_external/research_external_safeguard.md`
- `reports/research_external/research_external_spml.md`

## Slide 11 — Ablations (0:45)
Talk track:
- "The repo includes a historical threshold sweep table, but not a current canonical ablation pack for latest reports."
- "I explicitly mark this as missing and propose a dedicated ablation artifact as next work."

Evidence paths:
- `project_plan.md`
- `reports/research/summary_report.md`
- `STATUS.md`

## Slide 12 — Production Plan (0:55)
Talk track:
- "Production should run ML inline and use LLM escalation only for uncertainty, with policy actions and auditability."
- "Monitoring needs drift, calibration, and escalation-rate alerts tied to retraining cadence."
- "Current repo has strong research plumbing but no production API/policy implementation yet."

Evidence paths:
- `PRD.md`
- `STATUS.md`
- `src/cli/predict.py`
- `src/hybrid_router.py`
- `dvc.yaml`

## Slide 13 — Lessons + Next Steps (0:40)
Talk track:
- "The strongest technical asset is reproducible experimentation and clear artifact lineage."
- "The largest open risk is external generalization and policy-ready calibration."
- "Top priorities: improve benign diversity, rerun strict full-coverage hybrid evaluation, and add formal ablations."

Evidence paths:
- `STATUS.md`
- `reports/research/summary_report.md`
- `reports/research_external/research_external_spml.md`
- `configs/default.yaml`

## Slide 14 — Appendix (0:20)
Talk track:
- "I keep canonical-vs-legacy report paths explicit so interview claims stay auditable."
- "All deck visuals are generated from repository artifacts by `make_plots.py`."

Evidence paths:
- `README.md`
- `reports/research/`
- `reports/research_external/`
- `interview_presentation/make_plots.py`

