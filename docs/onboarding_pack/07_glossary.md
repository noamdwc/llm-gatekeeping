# Glossary

## Core Domain Terms

### Prompt injection
An adversarial prompt intended to override instructions, exfiltrate hidden context, or hijack behavior. This is the main threat class the repo studies.

### Jailbreak
A prompt designed to bypass model safety or policy restrictions. In this repo, jailbreak-like prompts are treated as adversarial.

### Adversarial
The positive class in binary detection. Usually means prompt injection or jailbreak intent.

### Benign
The negative class in binary detection. Includes ordinary prompts and synthetic benign prompts.

## Project-Specific Label Terms

### `label_binary`
Top-level label: `adversarial` or `benign`.

### `label_category`
Second-level label. Common values:
- `unicode_attack`
- `nlp_attack`
- `benign`

### `label_type`
Third-level label. For Unicode attacks this is a specific subtype. NLP attacks are mostly collapsed to `nlp_attack`.

### Unicode attack
An attack relying on Unicode or character-level perturbations such as homoglyphs or zero-width characters.

### NLP attack
An attack based on word substitutions or similar perturbations. The repo treats these as hard for the ML specialist to separate.

### Held-out attacks
Attack types listed in `configs/default.yaml` under `labels.held_out_attacks` and sent entirely to `test_unseen`.

## Pipeline Terms

### Research parquet
A wide parquet artifact containing ground truth, predictions, confidences, routing decisions, and metadata for analysis.

### DVC stage
A reproducible pipeline step defined in `dvc.yaml`.

### Frozen stage
A DVC stage intentionally not recomputed during ordinary `dvc repro`. Here, the LLM stage is frozen by default because it costs API tokens.

### `prompt_hash`
A deterministic hash of the original prompt used to keep prompt families together during split creation.

### `sample_id`
A deterministic hash of the modified sample text used to align prediction artifacts across stages.

## Model Terms

### ML baseline
The scikit-learn character-level classifier in `src/ml_classifier/ml_baseline.py`.

### Binary calibration
A post-training calibration step that adjusts binary confidence estimates, implemented with `CalibratedClassifierCV`.

### Few-shot examples
Prompt examples included in LLM messages to steer classification behavior.

### Dynamic few-shot
An embedding-based retrieval mode that selects examples from an exemplar bank rather than using only static examples.

### Exemplar bank
A stored set of embedded examples used to retrieve similar benign/attack prompt pairs for LLM prompting.

### Judge
A second LLM pass used to independently review low-confidence classifier outputs.

### Abstain
A hybrid outcome meaning the LLM result was not trusted enough. The system treats this conservatively as adversarial for binary output in some paths.

## Operational Terms

### Strict hybrid
Inferred from `src/research.py` and external research flow. Means escalated samples must have actual LLM coverage rather than silently falling back to ML.

### Routing diagnostics
Additional counts and rates showing how many samples were handled by ML, escalated to LLM, or abstained.

### External datasets
Additional Hugging Face datasets configured in `configs/default.yaml` for generalization evaluation.

### Synthetic benign
LLM-generated benign prompts used to make the classifier more robust to instruction-like but harmless text.
