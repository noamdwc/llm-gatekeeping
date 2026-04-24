# Core Flows

## 1. Preprocess The Main Dataset
Trigger:
- `python -m src.preprocess`
- DVC `preprocess` stage

Steps:
1. Load the configured Hugging Face dataset in `src/preprocess.py`
2. Add hierarchical labels: binary, category, type
3. Build a benign set from original prompts
4. Optionally merge validated synthetic benign prompts
5. Deduplicate by modified text
6. Add `prompt_hash`
7. Write `data/processed/full_dataset.parquet`

Key files:
- `configs/default.yaml`
- `src/preprocess.py`
- `src/synthetic_benign.py`
- `src/validators.py`

Output/result:
- Combined labeled parquet for downstream splitting

Failure points / debugging tips:
- Missing synthetic benign parquet when synthetic mode is enabled causes a hard failure
- Unexpected label names will surface later as category/type issues
- If row counts look odd, check deduplication and benign resampling behavior

## 2. Build Grouped Splits
Trigger:
- `python -m src.build_splits`
- DVC `build_splits` stage

Steps:
1. Load `full_dataset.parquet`
2. Separate held-out attack types and split their prompt_hashes 50/50 into `unseen_val` (monitoring) and `unseen_test` (final generalization)
3. Reserve benign hashes for the unseen splits to match main-pool adv/benign ratio
4. Shuffle remaining `prompt_hash` groups using configured seed
5. Split remaining groups into train/val/test
6. Save split parquets

Key files:
- `src/build_splits.py`
- `configs/default.yaml`

Output/result:
- `data/processed/splits/train.parquet`
- `data/processed/splits/val.parquet`
- `data/processed/splits/test.parquet`
- `data/processed/splits/unseen_val.parquet`
- `data/processed/splits/unseen_test.parquet`

Failure points / debugging tips:
- If leakage is suspected, inspect `prompt_hash` grouping
- If unseen results look too good, confirm held-out attack names in config

## 3. Train The ML Baseline And Produce ML Predictions
Trigger:
- `python -m src.ml_classifier.ml_baseline --research`
- DVC `ml_model` stage

Steps:
1. Load train/val/test split parquets
2. Filter NLP attacks out of ML training scope
3. Build char-level TF-IDF plus handcrafted features
4. Train logistic-regression models for hierarchy levels
5. Optionally calibrate binary confidence
6. Save `ml_baseline.pkl`
7. Write ML prediction parquets for configured splits

Key files:
- `src/ml_classifier/ml_baseline.py`
- `src/ml_classifier/utils.py`
- `configs/default.yaml`

Output/result:
- Trained model in `data/processed/models/ml_baseline.pkl`
- Research prediction parquets in `data/processed/predictions/`

Failure points / debugging tips:
- If training complains about a single binary class, inspect benign coverage in the training split
- If performance collapses on NLP attacks, that is expected from the design
- Confidence source can be raw or calibrated depending on artifact version

## 4. Run LLM Classification
Trigger:
- `python -m src.llm_classifier.llm_classifier --split test --research`
- DVC `llm_classifier` stage
- `python -m src.cli.predict --mode llm`

Steps:
1. Build few-shot examples from training data
2. Select provider via `LLM_PROVIDER`
3. Build classifier prompt messages
4. Call the model and parse JSON output
5. Optionally call the judge model for low-confidence cases
6. Cache raw chat responses under `.cache/llm/`
7. Persist LLM prediction parquet in research mode

Key files:
- `src/llm_classifier/llm_classifier.py`
- `src/llm_classifier/prompts.py`
- `src/embeddings.py`
- `src/llm_cache.py`
- `src/llm_provider.py`

Output/result:
- LLM predictions with confidence, evidence, and judge metadata

Failure points / debugging tips:
- Provider/API-key mismatch is the first thing to check
- Cache can mask upstream prompt/model changes if request arguments are unchanged
- Dynamic few-shot behavior depends on available embeddings and train split contents

## 5. Compute Hybrid Research Outputs
Trigger:
- `python -m src.research --split test`
- DVC `research` stage

Steps:
1. Load split parquet and ML prediction artifact
2. Optionally load LLM prediction artifact
3. Join artifacts by `sample_id`
4. Compute hybrid routing decisions
5. Mark ML fast-path, LLM escalation, or abstain
6. Compute routing diagnostics and report-ready fields
7. Save wide research parquet

Key files:
- `src/research.py`
- `src/evaluate.py`

Output/result:
- `data/processed/research/research_test.parquet`

Failure points / debugging tips:
- Strict hybrid mode requires LLM coverage for escalations
- Missing or empty LLM artifacts should be treated as pipeline-precondition issues, not silent fallbacks
- If row counts mismatch, inspect `sample_id` generation and merge assumptions

## 6. Evaluate External Datasets
Trigger:
- `python -m src.eval_external --dataset deepset --mode ml`
- `python -m src.cli.research_external --dataset deepset --mode hybrid`
- DVC external `foreach` stages

Steps:
1. Load the configured external HF dataset
2. Map dataset-specific labels to `adversarial`/`benign`
3. Rename text column to `modified_sample`
4. Run ML, or require precomputed external LLM predictions for hybrid research mode
5. Build per-dataset research parquet/report

Key files:
- `src/eval_external.py`
- `src/cli/research_external.py`
- `configs/default.yaml`
- `dvc.yaml`

Output/result:
- External reports and research parquets under `data/processed/research_external/` and `reports/research_external/`

Failure points / debugging tips:
- External label-map mismatches will silently drop rows after warning
- Hybrid external mode fails loudly if the required LLM artifact is missing
- External datasets are binary-only, so category/type outputs are synthetic placeholders

## 7. Lightweight Local Inference
Trigger:
- `./run_inference.sh --mode ml --split test`
- `echo "text" | python -m src.cli.predict --mode hybrid --pretty`

Steps:
1. Verify split/model artifacts exist
2. Load saved ML model
3. Route to ML-only, LLM-only, or hybrid code path
4. Return JSON or save an inference report

Key files:
- `run_inference.sh`
- `src/cli/predict.py`
- `src/cli/infer_split.py`
- `src/hybrid_router.py`

Output/result:
- CLI output for ad hoc inspection
- Optional Markdown inference reports

Failure points / debugging tips:
- `run_inference.sh` assumes artifacts already exist from a prior DVC run
- Hybrid and LLM modes will require valid API credentials
- ML-mode reports include NLP rows, but they are out of scope for the ML specialist; read the scope breakdown
