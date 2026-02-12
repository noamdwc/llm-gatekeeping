---
title: "[Short descriptive title]"
date: YYYY-MM-DD
author: "[Agent name or human author]"
type: "[research-insight | error-analysis | experiment-comparison]"
git_commit: "[output of git rev-parse HEAD]"
wandb_runs:
  - "[run_id]: [run_name]"
files_analyzed:
  - "[filepath]: lines [N-M] — [what was examined]"
tools_used:
  - "[Read | Grep | WandB SDK | Shell | WebSearch]"
status: "[draft | reviewed | final]"
---

# [Title]

## Executive Summary

<!-- 2-3 sentence overview of what was investigated and the key finding -->

## Background

<!-- Why this investigation was needed. What question was asked. -->

## Methodology

### Data Examined
<!-- What data was loaded, how many samples, which splits -->

| Dataset | Samples | Source |
|---------|---------|--------|
| test split | N | `data/processed/test.parquet` |

### Code Analyzed
<!-- Which source files and specific line ranges were examined -->

- `src/llm_classifier.py:144-165` — Binary classification prompt
- ...

### WandB Runs Referenced
<!-- Run IDs and what each run represents -->

| Run ID | Name | Key Config |
|--------|------|-----------|
| `abc123` | llm-gpt4o-mini-test | model=gpt-4o-mini, few_shot=2 |

## Findings

### Finding 1: [Title]

**Evidence:**
<!-- Specific metric values, code references with line numbers, data counts -->

**Interpretation:**
<!-- What this means for the classifier's behavior -->

### Finding 2: [Title]

**Evidence:**

**Interpretation:**

<!-- Add more findings as needed -->

## Recommendations

<!-- Prioritized, actionable suggestions. Each should reference the finding it addresses. -->

| Priority | Recommendation | Addresses | Expected Impact |
|----------|---------------|-----------|-----------------|
| High | [specific change] | Finding 1 | [expected metric improvement] |
| Medium | [specific change] | Finding 2 | [expected metric improvement] |

### Detailed Recommendations

#### 1. [Recommendation title]
<!-- Specific implementation details, which files to change, what to try -->

#### 2. [Recommendation title]

## Open Questions

<!-- Things that need further investigation or user input -->

- [ ] [Question 1]
- [ ] [Question 2]

## Appendix

### Raw Metrics
<!-- Tables, confusion matrices, or other detailed data -->

### Code Snippets
<!-- Relevant code blocks with file paths and line numbers -->
