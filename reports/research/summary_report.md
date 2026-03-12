# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|-------|------|----------|--------|-----------|-----|-----|
| ML (unicode scope) | 1094 | 0.9872 | 0.9926 | 0.9507 | 0.0146 | 0.0125 |
| Hybrid | 1819 | 0.8395 | 0.9085 | 0.3482 | 0.4307 | 0.1385 |
| LLM | 1819 | 0.5618 | 0.6968 | 0.2101 | 0.2263 | 0.4554 |

- routing: abstain=19, llm=872, ml=928

## FPR Diagnostic Views (Hybrid)

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.4307 | All samples, abstain=adversarial |
| Abstain-excluded | 0.4265 | 19 abstain samples removed |
| Abstain rate | 0.0104 | 19/1819 samples |

## Benign Risk Model (train-on-val, eval-on-test)

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 | 0.8134 | 0.1526 |
| Logistic (all features) | 0.7558 | 0.9102 | 0.1094 |

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|
| 2427 | 0.3490 | 0.7713 | 0.5186 | 0.8500 | 0.0044 | 0.6470 | 847 | 1580 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|---------|------|-------|----------|--------|-----------|-----|-----|
| deepset | 116 | 0.5172 | 0.5431 | 0.2319 | 0.6748 | 0.0179 | 0.8667 |
| jackhhao | 262 | 0.5305 | 0.7595 | 0.7097 | 0.7948 | 0.0081 | 0.4460 |
| safeguard | 2049 | 0.3163 | 0.7857 | 0.4937 | 0.8641 | 0.0036 | 0.6698 |
