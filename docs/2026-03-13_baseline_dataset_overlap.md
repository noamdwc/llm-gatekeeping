# Baseline Dataset Overlap Audit

Date: 2026-03-13

## Executive Summary

This note audits whether the two external HuggingFace baselines we added, `rogue-security/prompt-injection-jailbreak-sentinel-v2` and `protectai/deberta-v3-base-prompt-injection-v2`, are being compared against datasets that overlap with their training data or with our own corpus. The main result is that `safeguard` is not a clean external benchmark for our pipeline because the Mindgard dataset we train on explicitly states that its original prompt-injection samples were curated from Safe-Guard, and our local materialized copies contain 474 exact text overlaps with the `safeguard` external test set. `jackhhao` has a smaller but real exact overlap with our corpus, and it is also explicitly listed as one of ProtectAI v2's training datasets. `deepset` is the cleanest external benchmark in the current setup: we found no exact local overlap and no direct documented training-data overlap for either our pipeline or the baselines.

The key distinction in this report is between:

- documented provenance overlap: a model card or dataset card says one dataset was used to build or train another
- exact local overlap: identical prompt strings found in our local materialized datasets
- unknown overlap: public sources do not disclose enough information to determine overlap confidently

## What We Train On

Our pipeline is configured to load `Mindgard/evaded-prompt-injection-and-jailbreak-samples` as its adversarial source dataset in [configs/default.yaml](/Users/noamc/repos/llm-gatekeeping/configs/default.yaml#L3). The preprocessing pipeline then constructs benign examples from unique `original_sample` values from that same raw dataset in [src/preprocess.py](/Users/noamc/repos/llm-gatekeeping/src/preprocess.py#L70).

That design matters for overlap analysis:

- the adversarial side is Mindgard-derived
- the benign side is not independent; it is also seeded from Mindgard originals
- therefore any dataset family overlap with Mindgard can affect both the adversarial and benign sides of our training/eval corpus

### Local Corpus Snapshot

From the materialized local parquet files:

| Item | Count |
|------|-------|
| `full_dataset.parquet` rows | 12,082 |
| adversarial rows | 11,172 |
| benign rows | 910 |
| unique `modified_sample` strings | 12,082 |
| unique `original_sample` strings | 910 |
| benign `original_sample` values that exactly overlap adversarial `original_sample` values | 623 |

This is consistent with the earlier repo note in [reports/fp_investigation_report.md](/Users/noamc/repos/llm-gatekeeping/reports/fp_investigation_report.md#L5), which already calls out label noise caused by building benigns from pre-perturbation adversarial prompts.

## Public Provenance: Baselines vs Our Data

### Our dataset family

The Mindgard dataset card states that "The original prompt injection samples were curated from Safe-Guard-Prompt-Injection." Source: [Mindgard dataset card](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples).

That single sentence is enough to establish documented source-family overlap between our corpus and the `safeguard` external benchmark.

### ProtectAI v2

The ProtectAI model card states that the training data was assembled from multiple public datasets and explicitly lists `jackhhao/jailbreak-classification` among them in the "Datasets used to train" section. Source: [ProtectAI v2 model card](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2).

What we can say from public sources:

- documented overlap with `jackhhao`: yes
- documented overlap with Mindgard: not found
- documented overlap with Safe-Guard: not found on the public model card
- documented overlap with `deepset`: not found on the public model card

Because the model card does not name all source datasets exhaustively in the visible card text, absence of evidence here is not proof of no overlap. It is only "not publicly documented in the sources we checked."

### Sentinel v2

The Sentinel v2 model card says the model was trained on "3× more data compared to v1" and reports benchmark scores on `rogue-security/prompt-injections-benchmark`, `allenai/wildjailbreak`, `jackhhao/jailbreak-classification`, `deepset/prompt-injections`, and `xTRam1/safe-guard-prompt-injection`. Source: [Sentinel v2 model card](https://huggingface.co/rogue-security/prompt-injection-jailbreak-sentinel-v2).

What the public model card does not disclose:

- the exact training dataset names
- whether `jackhhao`, `deepset`, or `safeguard` were part of training rather than only evaluation
- whether any Mindgard-derived or Safe-Guard-derived content was included in training

So for Sentinel v2 the correct conclusion is:

- benchmark reuse is documented
- training overlap with our data is unknown from current public sources

## Exact Local Overlap: Our Corpus vs External Benchmarks

The table below compares exact string overlap between the materialized external evaluation sets in `data/processed/research_external/` and our own materialized train/val/test data.

| External dataset | External rows | Overlap with train | Overlap with val | Overlap with test | Overlap with test_unseen originals | Main interpretation |
|------------------|---------------|--------------------|------------------|-------------------|------------------------------------|--------------------|
| `deepset` | 116 | 0 | 0 | 0 | 0 | No exact local overlap found |
| `jackhhao` | 262 | 8 | 0 | 1 | 9 | Small but real exact overlap |
| `safeguard` | 2049 | 342 | 57 | 75 | 474 | Large overlap; not a clean external set |

### Notes on the overlap counts

- For `jackhhao`, the overlap is concentrated in known jailbreak-style prompts such as DUDE/JB/Burple persona prompts.
- For `safeguard`, the overlap is large and spread across train/val/test because our Mindgard-derived corpus contains many prompts sourced from Safe-Guard.
- For `deepset`, we found no exact string matches against our materialized train/val/test/full data.

## Exact Local Overlap: External Benchmarks vs Each Other

We also checked exact string overlap among the external benchmarks themselves.

| Pair | Exact shared prompts |
|------|----------------------|
| `deepset` vs `jackhhao` | 0 |
| `deepset` vs `safeguard` | 1 |
| `jackhhao` vs `safeguard` | 0 |

Interpretation:

- the external datasets are mostly distinct from each other at the exact-string level
- the contamination problem is mostly between our Mindgard-derived corpus and `safeguard`, not between the external benchmarks themselves

## Baseline-by-Baseline Assessment

### 1. ProtectAI v2

#### Documented training overlap with our datasets

| Comparison target | Public overlap status | Reason |
|-------------------|-----------------------|--------|
| Our Mindgard-derived train/val/test | Unknown | The public card does not name Mindgard or Safe-Guard as ProtectAI training sources |
| `jackhhao` external eval | Yes | `jackhhao/jailbreak-classification` is explicitly listed as a training dataset |
| `deepset` external eval | Not documented | No direct source evidence found on the card |
| `safeguard` external eval | Not documented | No direct source evidence found on the card |

#### Practical interpretation

- `jackhhao` is not a clean external generalization benchmark for ProtectAI v2
- `safeguard` may still be partially contaminated in a broader family sense, but we do not have direct public evidence that ProtectAI trained on Safe-Guard
- comparisons between ProtectAI and our pipeline on `safeguard` are doubly fragile because `safeguard` overlaps strongly with our own data

### 2. Sentinel v2

#### Documented training overlap with our datasets

| Comparison target | Public overlap status | Reason |
|-------------------|-----------------------|--------|
| Our Mindgard-derived train/val/test | Unknown | The model card does not disclose training dataset names |
| `jackhhao` external eval | Unknown | The card reports evaluation on `jackhhao`, but does not say it trained on it |
| `deepset` external eval | Unknown | Same issue |
| `safeguard` external eval | Unknown | Same issue |

#### Practical interpretation

- Sentinel v2 benchmark results on these datasets are publicly documented
- Sentinel v2 training overlap cannot be confirmed or ruled out from the public card
- the cleanest statement is that Sentinel's overlap status is unknown, not clean

## External Benchmark Quality for Our Comparisons

### `safeguard`

This should be treated as a source-overlapping benchmark, not a truly unseen external dataset, for our pipeline.

Why:

- the Mindgard dataset card explicitly says its original prompt-injection samples were curated from Safe-Guard
- our materialized data contains 474 exact string overlaps with the `safeguard` external test set

Implication:

- `safeguard` is still useful as a stress test
- it should not be used as the primary evidence of external generalization

### `jackhhao`

This is partially contaminated.

Why:

- there are 9 exact prompt overlaps with our local Mindgard-derived corpus
- ProtectAI v2 explicitly trained on `jackhhao`

Implication:

- `jackhhao` is still informative, especially for Sentinel and our pipeline
- but for ProtectAI it is partly in-domain by construction

### `deepset`

This is the cleanest external benchmark in the current setup.

Why:

- no exact local overlap found with our materialized train/val/test/full data
- no direct documented training overlap found for our pipeline, ProtectAI, or Sentinel from the sources checked

Implication:

- `deepset` should be the lead external comparison in summary tables and narrative conclusions

## Recommended Reporting Language

For future summary reports and benchmark tables:

- keep all three external datasets
- annotate `safeguard` as "source-overlapping with Mindgard; not a fully external benchmark"
- annotate `jackhhao` as "small exact overlap with our corpus; explicit training dataset for ProtectAI v2"
- annotate `deepset` as "lowest-known-overlap external benchmark in current setup"

Suggested short labels:

| Dataset | Annotation |
|---------|------------|
| `deepset` | cleanest external benchmark |
| `jackhhao` | partial overlap / ProtectAI-trained |
| `safeguard` | Mindgard-family overlap |

## Methodology

### Local overlap check

Exact overlap was computed by comparing string equality on the materialized `modified_sample` and `original_sample` columns from:

- `data/processed/full_dataset.parquet`
- `data/processed/splits/train.parquet`
- `data/processed/splits/val.parquet`
- `data/processed/splits/test.parquet`
- `data/processed/splits/test_unseen.parquet`
- `data/processed/research_external/research_external_deepset.parquet`
- `data/processed/research_external/research_external_jackhhao.parquet`
- `data/processed/research_external/research_external_safeguard.parquet`

This is a strict exact-string test only. It does not detect paraphrases, near-duplicates, formatting-normalized matches, or shared synthetic generation templates.

### Public-source review

Primary sources checked:

- ProtectAI v2 model card: <https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2>
- Sentinel v2 model card: <https://huggingface.co/rogue-security/prompt-injection-jailbreak-sentinel-v2>
- Mindgard dataset card: <https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples>
- Safe-Guard dataset card: <https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection>
- deepset dataset card: <https://huggingface.co/datasets/deepset/prompt-injections>
- jackhhao dataset card: <https://huggingface.co/datasets/jackhhao/jailbreak-classification>

### Limitations

- Sentinel v2's public card does not provide enough training-data detail to prove or disprove train/eval overlap.
- ProtectAI's visible card text names some, but not necessarily all, data sources.
- Exact-string overlap is a lower bound on contamination. Real overlap can be higher through paraphrases or derived prompts.

## Bottom Line

If we want one external benchmark to anchor baseline comparisons, use `deepset` first. Keep `jackhhao` and `safeguard` in the report, but annotate them clearly:

- `jackhhao`: partially contaminated, especially for ProtectAI
- `safeguard`: substantially contaminated for our Mindgard-derived pipeline
- Sentinel v2: training-data overlap status remains unknown from public sources
