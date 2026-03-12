# First Steps Snapshot

## 1) Top 10 Files To Rely On

| File path | Why it matters |
|---|---|
| `README.md` | Canonical project framing, hierarchy, pipeline stages, and where latest outputs live. |
| `configs/default.yaml` | Source of truth for dataset names, label schema, thresholds, and external datasets. |
| `dvc.yaml` | Reproducible stage graph and artifact lineage from raw data to reports. |
| `src/preprocess.py` | How benign data is constructed and hierarchical labels are assigned. |
| `src/build_splits.py` | Grouped split logic and held-out attack-type strategy. |
| `src/ml_classifier/ml_baseline.py` | ML feature design, training scope (unicode+benign), and calibration setup. |
| `src/llm_classifier/llm_classifier.py` | LLM classifier/judge architecture and default evaluation limit. |
| `src/research.py` | Hybrid routing behavior and report generation logic. |
| `reports/research/eval_report_ml.md` | Main ML metrics on canonical latest artifact. |
| `reports/research/eval_report_hybrid.md` | Main hybrid metrics + routing diagnostics on canonical latest artifact. |

## 2) Key Metrics + Dataset Stats (from repo artifacts)

| Scope | Rows | Adversarial | Benign | Accuracy | Adv Recall | Benign Recall | FNR | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full processed dataset | 11,795 | 11,172 | 623 | TODO | TODO | TODO | TODO | `data/processed/full_dataset.parquet` |
| Train split | 7,633 | 7,197 | 436 | TODO | TODO | TODO | TODO | `data/processed/splits/train.parquet` |
| Val split | 1,652 | 1,559 | 93 | TODO | TODO | TODO | TODO | `data/processed/splits/val.parquet` |
| Test split | 1,690 | 1,596 | 94 | TODO | TODO | TODO | TODO | `data/processed/splits/test.parquet` |
| Main ML (unicode scope) | 996 | 902 | 94 | 0.9839 | 0.9867 | 0.9574 | 0.0133 | `reports/research/eval_report_ml.md` |
| Main Hybrid | 1,690 | 1,596 | 94 | 0.6219 | 0.6034 | 0.9362 | 0.3966 | `reports/research/eval_report_hybrid.md` |
| Main LLM | 100 | 89 | 11 | 0.7100 | 0.6966 | 0.8182 | 0.3034 | `reports/research/eval_report_llm.md` |
| External deepset | 116 | 60 | 56 | 0.4483 | 0.7500 | 0.1250 | 0.2500 | `reports/research_external/research_external_deepset.md` |
| External jackhhao | 262 | 139 | 123 | 0.2672 | 0.0504 | 0.5122 | 0.9496 | `reports/research_external/research_external_jackhhao.md` |
| External safeguard | 2,049 | 648 | 1,401 | 0.3514 | 0.0324 | 0.4989 | 0.9676 | `reports/research_external/research_external_safeguard.md` |
| External spml | 15,917 | 12,541 | 3,376 | 0.1253 | 0.0494 | 0.4073 | 0.9506 | `reports/research_external/research_external_spml.md` |

## 3) Draft Slide Titles (titles only)

1. LLM Security Gatekeeper: Detect Adversarial Prompts Before They Reach Production LLMs
2. Problem: Prompt Injection/Jailbreak Detection for Real Traffic
3. Why This Is Hard: Evasion, Dataset Shift, and Cost-Latency Constraints
4. High-Level Solution: ML-First + LLM Escalation Pipeline
5. Data: Sources, Label Hierarchy, and Split Strategy
6. Modeling: Feature Stack, LLM Cascade, and Routing Logic
7. Training & Evaluation Design
8. Main Results: ML vs Hybrid vs LLM (Canonical Reports)
9. External Generalization Results: What Breaks OOD
10. Error Analysis: Failure Patterns and Calibration Gaps
11. Ablations and Sensitivity (Available + Missing)
12. Productionization Plan: SLOs, Monitoring, Retraining, Guardrails
13. Lessons Learned and Next Steps
14. Appendix: Artifact Map, Legacy Report Caveats, and Extra Tables
