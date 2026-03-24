# First Steps Snapshot

## 1) Top 10 Files To Rely On

| File path | Why it matters |
|---|---|
| `README.md` | Canonical project framing, hierarchy, pipeline stages, and where latest outputs live. |
| `configs/default.yaml` | Source of truth for dataset names, label schema, thresholds, DeBERTa gate config, and external datasets. |
| `dvc.yaml` | Reproducible stage graph and artifact lineage from raw data to reports. |
| `src/preprocess.py` | How benign data is constructed and hierarchical labels are assigned. |
| `src/build_splits.py` | Grouped split logic and held-out attack-type strategy. |
| `src/ml_classifier/ml_baseline.py` | ML feature design, training scope (unicode+benign), and calibration setup. |
| `src/hybrid_router.py` | 4-tier routing logic: ML → DeBERTa → LLM → risk model. |
| `src/models/deberta_classifier.py` | DeBERTa binary gate model architecture and training. |
| `src/llm_classifier/llm_classifier.py` | LLM classifier/judge architecture and evaluation. |
| `reports/research/summary_report.md` | Current canonical metrics, routing stats, and baseline comparison. |

## 2) Key Metrics + Dataset Stats (from repo artifacts)

| Scope | Rows | Adversarial | Benign | Accuracy | Adv Recall | Benign Recall | FNR | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full processed dataset | ~13,166 | ~11,172 | ~1,994 | - | - | - | - | `data/processed/full_dataset.parquet` |
| Train split | 8,881 | 7,486 | 1,395 | - | - | - | - | `data/processed/splits/train.parquet` |
| Val split | 1,847 | 1,548 | 299 | - | - | - | - | `data/processed/splits/val.parquet` |
| Test split | 1,618 | 1,318 | 300 | - | - | - | - | `data/processed/splits/test.parquet` |
| Main ML (unicode scope) | 1,070 | 770 | 300 | 0.9888 | 0.9844 | 1.0000 | 0.0156 | `reports/research/eval_report_ml.md` |
| Main Hybrid | 1,618 | 1,318 | 300 | 0.9580 | 0.9788 | 0.8667 | 0.0212 | `reports/research/eval_report_hybrid.md` |
| Main LLM | 1,618 | 1,318 | 300 | 0.8072 | 0.7982 | 0.8467 | 0.2018 | `reports/research/eval_report_llm.md` |
| DeBERTa (test) | 1,618 | 1,318 | 300 | 0.8597 | 0.8308 | 0.9867 | - | `reports/deberta_classifier/summary.md` |
| External deepset | 116 | 60 | 56 | 0.6121 | 0.6500 | 0.5714 | 0.3500 | `reports/research/summary_report.md` |
| External jackhhao | 262 | 139 | 123 | 0.8817 | 0.9065 | 0.8537 | 0.0935 | `reports/research/summary_report.md` |
| External safeguard | 2,049 | 648 | 1,401 | 0.7613 | 0.4660 | 0.8979 | 0.5340 | `reports/research/summary_report.md` |

## 3) Routing Breakdown (test split)

| Tier | Samples | % | Description |
|---|---:|---:|---|
| ML fast-path | 747 | 46.2% | High-confidence unicode attack detection |
| DeBERTa gate | 600 | 37.1% | Binary classification at threshold 0.93 |
| LLM escalation | 91 | 5.6% | Classifier + judge for uncertain samples |
| Abstain/Risk | 180 | 11.1% | Post-hoc resolution via risk model |

## 4) Draft Slide Titles (titles only)

1. LLM Security Gatekeeper: Detect Adversarial Prompts Before They Reach Production LLMs
2. Problem: Prompt Injection/Jailbreak Detection for Real Traffic
3. Why This Is Hard: Evasion, Dataset Shift, and Cost-Latency Constraints
4. High-Level Solution: 4-Tier Hybrid Routing Pipeline
5. Data: Sources, Label Hierarchy, Synthetic Benign, and Split Strategy
6. Modeling: ML Specialist, DeBERTa Gate, LLM Cascade, Risk Model
7. DeBERTa as Hybrid Binary Gate
8. Main Results: ML vs Hybrid vs LLM (Canonical Reports)
9. External Generalization Results: What Breaks OOD
10. External Benchmark Caveats and Overlap Audit
11. Baseline Comparison: Sentinel v2, ProtectAI v2
12. Error Analysis: FPR, Abstain, and Risk Model
13. Production View: Cost, Latency, and Routing
14. What Changed Recently
15. Next Steps
16. Appendix: Artifact Map and Canonical Sources
