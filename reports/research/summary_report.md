# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|-------|------|----------|--------|-----------|-----|-----|
| ML (unicode scope) | 1094 | 0.9872 | 0.9926 | 0.9507 | 0.0146 | 0.0125 |
| Hybrid | 1819 | 0.7367 | 0.8375 | 0.3068 | 0.2263 | 0.2663 |
| LLM | 1819 | 0.5520 | 0.6879 | 0.2064 | 0.2263 | 0.4661 |

- routing: abstain=23, llm=868, ml=928

## FPR Diagnostic Views (Hybrid)

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.2263 | All samples, abstain=adversarial |
| Abstain-excluded | 0.2263 | 23 abstain samples removed |
| Abstain rate | 0.0126 | 23/1819 samples |

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|
| 2427 | 0.3490 | 0.7721 | 0.5220 | 0.8504 | 0.0051 | 0.6434 | 847 | 1580 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|---------|------|-------|----------|--------|-----------|-----|-----|
| deepset | 116 | 0.5172 | 0.5517 | 0.2571 | 0.6790 | 0.0179 | 0.8500 |
| jackhhao | 262 | 0.5305 | 0.7595 | 0.7097 | 0.7948 | 0.0081 | 0.4460 |
| safeguard | 2049 | 0.3163 | 0.7862 | 0.4966 | 0.8643 | 0.0043 | 0.6667 |
