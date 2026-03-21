# autoresearch — LLM Classifier Optimization

You are an autonomous researcher optimizing an LLM-based adversarial prompt classifier.

## Setup

1. **Agree on a run tag** with the user (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read all in-scope files**:
   - `autoresearch/program.md` — this file (your instructions)
   - `autoresearch/prepare.py` — fixed eval harness, DO NOT MODIFY
   - `autoresearch/experiment.py` — the file you modify (all LLM knobs)
4. **Verify the data exists**: `ls data/processed/splits/val.parquet`
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
6. **Confirm and go**.

## What You're Optimizing

A two-stage LLM classifier for detecting adversarial prompts:
- **Stage 1 (classifier)**: Llama 3.1 8B via NVIDIA NIM. Predicts binary label (benign/adversarial/uncertain) with confidence and evidence.
- **Stage 2 (judge)**: Llama 3.1 70B. Conditionally reviews low-confidence predictions. Can accept or override the classifier's label.

The classifier feeds into a **hybrid router** where ML handles easy cases and LLM handles escalations. Improving LLM quality directly improves the hybrid pipeline.

## The Diagnosis (CRITICAL CONTEXT)

The current system has a **structural flaw** that you should fix first:

### Problem 1: Confidence is fake
The 8B classifier outputs exactly 3 confidence values:
- **0.90** for all benign predictions (284/1847 val samples)
- **0.88** for all adversarial predictions (1515/1847 val samples)
- **0.50** for uncertain (48/1847 val samples)

This happens because the few-shot examples use `confidence: 90` for benign and `confidence: 88` for adversarial. The model copies these values verbatim instead of producing real confidence scores.

### Problem 2: Judge triggers on everything
With `judge_confidence_threshold: 0.8` and all adversarial predictions at 0.88, only samples with conf < 0.8 trigger the judge (just the 0.50 uncertain ones). So the **current config actually avoids the worst damage**. But if you raise the threshold above 0.88, you'd send all adversarial predictions to the judge, which is catastrophic because...

### Problem 3: The 70B judge destroys accuracy (when it runs)
Historical data from when threshold was 0.9 (all adversarial hit judge):
- Judge overrides 630 adversarial→benign predictions
- Of those 630 overrides: only 27 correct, **603 wrong** (96% error rate)
- The judge is far too eager to flip adversarial→benign

### Problem 4: Classifier-only is quite good
The 8B classifier alone gets 1563 adversarial predictions with 1548 actually adversarial (99% precision for adversarial). But it only catches 1563/1548 adversarial = missing ~0. Wait — the issue is actually **benign recall**: the classifier calls 284 samples benign, but how many are correct?

### Current Baseline (val set, N=1847, threshold=0.8)
- Accuracy: 62%, Adversarial recall: 57%, Benign recall: 85%, FPR: 14.7%
- adv_f1: 0.72, benign_f1: 0.42

The subsample (N=200) baseline will differ — establish it on your first run.

## Experimentation

**What you CAN do:**
- Modify `autoresearch/experiment.py` — this is the only file you edit. Everything in it is fair game: system prompts, few-shot confidence values, judge threshold, few-shot strategy, patterns, hard-benign examples.

**What you CANNOT do:**
- Modify `autoresearch/prepare.py`. It is read-only. It contains the fixed evaluation harness.
- Modify any file under `src/`, `configs/`, or `data/`.
- Install new packages.

**Running an experiment:**
```bash
cd /Users/noamc/repos/llm-gatekeeping && /Users/noamc/miniconda3/envs/llm_gate/bin/python autoresearch/prepare.py > autoresearch/run.log 2>&1
```

**Extracting results:**
```bash
grep "^score:\|^adv_f1:\|^benign_f1:\|^fpr:\|^fnr:\|^accuracy:\|^adv_recall:\|^ben_recall:\|^gate_pass:\|^judge_rate:" autoresearch/run.log
```

If grep output is empty, the run crashed. Read `tail -n 50 autoresearch/run.log` for the error.

**The goal: maximize `score`.** The composite score is:
```
IF adv_recall >= 0.80 AND accuracy >= 0.55:
    score = 0.4 * adv_f1 + 0.4 * benign_f1 + 0.2 * (1 - FPR)
ELSE:
    score = -1  (gate failure)
```

Benign F1 has the most headroom (currently ~0.42), so improving benign classification without hurting adversarial recall gives the biggest score gains.

## Promising Experiment Directions

Ordered by expected impact:

### 1. Fix confidence calibration (highest priority)
- The few-shot confidence values (90/88) are echoed verbatim by the model
- Try: spread them out (e.g., 95/75, or vary per example)
- Try: use dramatically different values to see if the model still copies
- Try: remove confidence from few-shot responses entirely and ask for it separately
- Goal: get a real confidence distribution, not 3 point masses

### 2. Prompt engineering for benign/adversarial boundary
- The classifier's adversarial precision is great (99%) but FPR is 14.7%
- The benign definition may be too narrow — try making it more explicit
- Try: add explicit "NOT adversarial" examples in the system prompt
- Try: emphasize that security discussion/education is benign
- Try: add the concept of "assume benign unless strong evidence of active attack"

### 3. Judge prompt (if judge is triggered)
- The judge overrides adversarial→benign 96% incorrectly
- The judge prompt has too much "benign by default" language
- Try: make the judge more conservative (require overwhelming evidence to override)
- Try: add "the classifier has 99% adversarial precision, override only with very strong counter-evidence"
- Try: frame the judge as a confirmer rather than an independent reviewer

### 4. Few-shot strategy
- Try: `FEW_SHOT_MODE = "none"` to see if few-shot is helping or hurting
- Try: different N_UNICODE_EXAMPLES / N_NLP_EXAMPLES ratios
- Try: INCLUDE_HARD_BENIGN = True (adds instruction-like benign examples)
- Try: vary FEW_SHOT_EVIDENCE_MAX_CHARS

### 5. Judge threshold tuning
- Current threshold 0.8 means only uncertain (0.50) samples hit the judge
- Try lowering to 0.5 (effectively disabling judge) to see pure classifier performance
- Try raising to 0.85 to send some adversarial predictions to judge (with improved judge prompt)

### 6. Benign task override patterns
- The judge has a deterministic override that forces benign on productivity tasks
- Try expanding BENIGN_TASK_INTENT_PATTERNS
- Try tightening BYPASS_INTENT_PATTERNS

## Simplicity Criterion

All else being equal, simpler is better. A small score improvement that adds complexity (longer prompts, more patterns) is less valuable than a simplification that maintains performance. If removing something gives equal or better results, that's a great outcome.

## Logging Results

Log to `autoresearch/results.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	score	adv_f1	benign_f1	fpr	judge_rate	status	description
```

- **commit**: short git hash (7 chars)
- **score**: composite score (or -1.0 for gate failures, 0.0 for crashes)
- **adv_f1**: adversarial F1
- **benign_f1**: benign F1
- **fpr**: false positive rate
- **judge_rate**: fraction of samples sent to judge
- **status**: `keep`, `discard`, or `crash`
- **description**: short text of what this experiment tried

Example:
```
commit	score	adv_f1	benign_f1	fpr	judge_rate	status	description
a1b2c3d	0.6280	0.7200	0.4200	0.1470	0.0260	keep	baseline
b2c3d4e	0.7100	0.8100	0.5500	0.0800	0.0000	keep	disable judge (threshold 0.5)
c3d4e5f	-1.000	0.5000	0.8000	0.0200	0.0000	discard	gate fail: adv_recall dropped to 0.60
```

Do NOT commit results.tsv — leave it untracked.

## The Experiment Loop

LOOP FOREVER:

1. Look at your results so far and current experiment.py state.
2. Choose an experimental idea. Edit `experiment.py`.
3. `git commit -m "experiment: <short description>"`
4. Run: `cd /Users/noamc/repos/llm-gatekeeping && /Users/noamc/miniconda3/envs/llm_gate/bin/python autoresearch/prepare.py > autoresearch/run.log 2>&1`
5. Extract results: `grep "^score:\|^adv_f1:\|^benign_f1:\|^fpr:\|^judge_rate:" autoresearch/run.log`
6. If empty, read `tail -n 50 autoresearch/run.log` and attempt fix or skip.
7. Log results to `autoresearch/results.tsv`.
8. If score improved: keep the commit (advance the branch).
9. If score equal or worse: `git reset --soft HEAD~1` to revert experiment.py.
10. Go to step 1.

**NEVER STOP.** Do not pause to ask the human. Run autonomously until manually interrupted. If you run out of ideas, re-read the diagnosis, try combining previous near-misses, try more radical prompt changes. Each experiment takes ~7-15 minutes (API rate limited), so expect ~4-8 experiments per hour.

**Crashes**: If it's a typo or import error, fix and retry. If the idea is fundamentally broken, skip it.

**Rate limits**: The NIM API is free but rate-limited to ~30 RPM. The eval harness handles this. If you see 429 errors, the harness will retry with backoff. Don't try to work around rate limits.
