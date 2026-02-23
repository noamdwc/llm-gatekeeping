# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FNR |
|-------|------|----------|--------|-----------|-----|
| ML (unicode scope) | 996 | 0.9829 | 0.9905 | 0.9128 | 0.0133 |
| Hybrid | 1690 | 0.6213 | 0.7506 | 0.2138 | 0.3966 |
| LLM | 100 | 0.7500 | 0.8408 | 0.4186 | 0.2584 |

- routing: llm=47, ml=1643

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-------------|----------------|
| 18344 | 0.7298 | 0.1556 | 0.0821 | 0.2182 | 0.9482 | 13388 | 4956 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FNR |
|---------|------|-------|----------|--------|-----------|-----|
| deepset | 116 | 0.5172 | 0.4483 | 0.5844 | 0.1795 | 0.2500 |
| jackhhao | 262 | 0.5305 | 0.2634 | 0.0676 | 0.3912 | 0.9496 |
| safeguard | 2049 | 0.3163 | 0.3553 | 0.0308 | 0.5170 | 0.9676 |
| spml | 15917 | 0.7879 | 0.1260 | 0.0818 | 0.1660 | 0.9506 |
