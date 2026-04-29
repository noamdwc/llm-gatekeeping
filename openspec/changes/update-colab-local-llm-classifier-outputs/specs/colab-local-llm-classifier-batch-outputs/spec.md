## ADDED Requirements

### Requirement: Notebook defines batch output targets
The Colab local LLM classifier notebook SHALL define output targets for all requested main splits and external datasets.

#### Scenario: Main split targets are configured
- **WHEN** the notebook configuration cells are executed
- **THEN** the configured main split targets include `train`, `val`, `test`, `unseen_val`, `unseen_test`, and `safeguard_test`

#### Scenario: External dataset targets are configured
- **WHEN** the notebook configuration cells are executed
- **THEN** the configured external dataset targets include `deepset` and `jackhhao`

### Requirement: Notebook writes per-target prediction artifacts
The notebook SHALL write separate checkpoint and final classifier-only parquet artifacts for each configured main split and external dataset target.

#### Scenario: Main split output is written
- **WHEN** classification completes for a main split target
- **THEN** the notebook writes that target's final predictions under the main predictions directory with the target split name in the filename

#### Scenario: External dataset output is written
- **WHEN** classification completes for an external dataset target
- **THEN** the notebook writes that target's final predictions under the external predictions directory with the external dataset key in the filename

#### Scenario: Checkpoint output is written
- **WHEN** classification has produced at least one pending row for a target
- **THEN** the notebook writes or updates a checkpoint parquet for that same target without mixing rows from other targets

### Requirement: Notebook resumes each target independently
The notebook SHALL resume each output target from its own valid checkpoint rows.

#### Scenario: Target has an existing checkpoint
- **WHEN** a target's checkpoint parquet exists before classification starts
- **THEN** the notebook excludes valid completed `sample_id` rows from that target's pending rows

#### Scenario: Target checkpoint contains invalid rows
- **WHEN** a target's checkpoint parquet contains rows that fail prediction validation
- **THEN** the notebook ignores those rows for resume purposes and reprocesses their `sample_id` values if they are present in the target input

### Requirement: Notebook preserves classifier-only prediction schema per target
The notebook SHALL preserve the existing classifier-only prediction columns, provider metadata, model metadata, parse status, confidence, raw response text, and token logprob behavior for every generated target artifact.

#### Scenario: Target output validates
- **WHEN** the notebook writes a final parquet for any configured target
- **THEN** the final parquet contains `sample_id`, all expected `llm_*` and `clf_*` prediction columns, has `llm_stages_run` equal to `1`, and contains no `judge_*` columns

#### Scenario: External output validates without main-only ground truth
- **WHEN** the notebook writes a final parquet for an external dataset target
- **THEN** prediction validation succeeds without requiring main-split-only hierarchical ground-truth columns

### Requirement: Notebook reports batch run results
The notebook SHALL print per-target progress and a final summary of generated outputs.

#### Scenario: Target progress is reported
- **WHEN** a target starts and completes classification
- **THEN** the notebook prints the target key, pending row count, completed row count, and final output path

#### Scenario: Batch summary is reported
- **WHEN** all configured targets have been attempted
- **THEN** the notebook prints a summary listing each target and its final output path or failure status
