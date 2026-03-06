# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|-------|------|----------|--------|-----------|-----|-----|
| ML (unicode scope) | 996 | 0.9839 | 0.9911 | 0.9184 | 0.0426 | 0.0133 |
| Hybrid | 1690 | 0.7083 | 0.8205 | 0.2212 | 0.2553 | 0.2939 |
| LLM | 1690 | 0.5349 | 0.6792 | 0.1548 | 0.2340 | 0.4787 |

- routing: abstain=44, llm=760, ml=886

## FPR Diagnostic Views (Hybrid)

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.2553 | All samples, abstain=adversarial |
| Abstain-excluded | 0.2391 | 44 abstain samples removed |
| Abstain rate | 0.0260 | 44/1690 samples |

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|
| 2427 | 0.3490 | 0.6370 | 0.3703 | 0.7450 | 0.1854 | 0.6942 | 847 | 1580 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|---------|------|-------|----------|--------|-----------|-----|-----|
| deepset | 116 | 0.5172 | 0.5259 | 0.5926 | 0.4330 | 0.6250 | 0.3333 |
| jackhhao | 262 | 0.5305 | 0.6450 | 0.6235 | 0.6643 | 0.2520 | 0.4460 |
| safeguard | 2049 | 0.3163 | 0.6423 | 0.2793 | 0.7621 | 0.1620 | 0.7809 |
