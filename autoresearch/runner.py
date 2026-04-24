#!/usr/bin/env python3
"""
autoresearch/runner.py — Orchestrator for routing optimization experiments.

Calls Claude Code to pick experiments, runs prepare.py, keeps or reverts.

Usage:
    python autoresearch/runner.py                    # 20-hour run
    python autoresearch/runner.py --max-hours 4      # 4-hour run
    python autoresearch/runner.py --dry-run           # print what would happen
"""

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENT_FILE = REPO / "autoresearch" / "experiment.py"
RESULTS_FILE = REPO / "autoresearch" / "results.tsv"
RUN_LOG = REPO / "autoresearch" / "run.log"
CTRL_LOG = REPO / "autoresearch" / "controller.log"
PYTHON = "/Users/noamc/miniconda3/envs/llm_gate/bin/python"
PREPARE = REPO / "autoresearch" / "prepare.py"

RUN_TIMEOUT = 2 * 60        # 2 min (no API calls, just local models)
CLAUDE_TIMEOUT = 5 * 60     # 5 min for Claude to pick experiment
NOTIFY_EVERY = 5

CLAUDE_PROMPT = (
    "Read autoresearch/program.md for research goals and diagnosis. "
    "Read autoresearch/results.tsv for all past experiment results. "
    "Read autoresearch/experiment.py for the current routing logic. "
    "Read autoresearch/run.log for the full output of the last eval run "
    "(includes per-dataset metrics and errors). "
    "Analyze what has been tried, what worked, and what didn't. "
    "Pick ONE next experiment. Edit autoresearch/experiment.py with your change. "
    "Output ONLY a single line describing what you changed — nothing else."
)

RESULTS_HEADER = "commit\tscore\tval_score\tdeepset_score\tjackhhao_score\tsafeguard_score\tstatus\tdescription\n"
METRIC_KEYS = ("score", "val_score", "deepset_score", "jackhhao_score", "safeguard_score")


def sh(*args, check=True):
    return subprocess.run(
        list(args), cwd=REPO, check=check, text=True, capture_output=True,
    ).stdout.strip()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with CTRL_LOG.open("a") as f:
        f.write(line + "\n")


def notify(msg):
    log(f"NOTIFY: {msg}")
    try:
        subprocess.run(
            ["claude", "-p", "--allowedTools", "mcp__telegram-notify__notify",
             f'Use the mcp__telegram-notify__notify tool. '
             f'path="autoresearch/results.tsv" message="{msg}"'],
            cwd=REPO, timeout=60, capture_output=True,
        )
    except Exception:
        pass


def parse_results():
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def best_score(rows):
    best, commit = -1.0, "none"
    for r in rows:
        if r.get("status") == "keep":
            try:
                s = float(r["score"])
                if s > best:
                    best, commit = s, r["commit"]
            except (ValueError, KeyError):
                pass
    return best, commit


def parse_metrics():
    if not RUN_LOG.exists():
        return None
    text = RUN_LOG.read_text(errors="replace")
    m = {}
    for key in METRIC_KEYS:
        match = re.search(rf"^{key}:\s*([-+]?\d+\.?\d*)$", text, re.MULTILINE)
        if match:
            m[key] = float(match.group(1))
    return m if "score" in m else None


def append_result(commit, metrics, status, desc):
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER)
    vals = [metrics.get(k, 0) for k in METRIC_KEYS]
    with RESULTS_FILE.open("a") as f:
        f.write(f"{commit}\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{status}\t{desc}\n")


def last_error():
    if not RUN_LOG.exists():
        return "no log"
    lines = RUN_LOG.read_text(errors="replace").strip().splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:200]
    return "empty log"


def experiment_changed():
    return bool(sh("git", "diff", "--", "autoresearch/experiment.py", check=False).strip())


def call_claude():
    try:
        proc = subprocess.Popen(
            ["claude", "-p", "--allowedTools", "Edit,Read",
             CLAUDE_PROMPT],
            cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        output_lines = []
        for line in proc.stdout:
            print(f"  [claude] {line}", end="", flush=True)
            output_lines.append(line.strip())
        proc.wait(timeout=CLAUDE_TIMEOUT)
        lines = [l for l in output_lines if l]
        return lines[-1][:200] if lines else None
    except subprocess.TimeoutExpired:
        proc.kill()
        log("Claude timed out")
        return None
    except Exception as e:
        log(f"Claude error: {e}")
        return None


def run_prepare():
    with RUN_LOG.open("w") as f:
        proc = subprocess.Popen(
            [PYTHON, str(PREPARE)], cwd=REPO, stdout=f, stderr=subprocess.STDOUT, text=True,
        )
        try:
            proc.wait(timeout=RUN_TIMEOUT)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)
    start = time.time()
    n, kept, discarded, crashed = 0, 0, 0, 0

    log(f"Starting orchestrator ({args.max_hours}h budget)")

    while time.time() - start < args.max_hours * 3600:
        rows = parse_results()
        bs, bc = best_score(rows)
        log(f"--- Experiment {n+1} | best={bs:.4f} ({bc}) ---")

        if args.dry_run:
            print(f"Would call: claude -p '<prompt>'")
            return

        desc = call_claude()
        if not desc:
            log("Claude produced nothing. Retry in 60s...")
            time.sleep(60)
            continue

        if not experiment_changed():
            log(f"No diff after Claude ('{desc}'). Retry in 30s...")
            time.sleep(30)
            continue

        # Commit
        sh("git", "add", "autoresearch/experiment.py")
        sh("git", "commit", "-m", f"experiment: {desc}")
        commit = sh("git", "rev-parse", "--short", "HEAD")
        log(f"Committed {commit}: {desc}")

        # Run eval
        ok = run_prepare()
        metrics = parse_metrics()

        if not ok or not metrics:
            crashed += 1
            append_result(commit, {}, "crash", desc)
            err = last_error()
            log(f"CRASH: {err}")
            sh("git", "reset", "--soft", "HEAD~1")
            sh("git", "checkout", "--", "autoresearch/experiment.py")
            notify(f"autoresearch crash!\\nExperiment: {desc}\\nError: {err[:150]}")
            n += 1
            continue

        score = metrics["score"]
        if score > bs:
            kept += 1
            append_result(commit, metrics, "keep", desc)
            log(f"KEEP score={score:.4f} (was {bs:.4f})")
            notify(f"autoresearch improvement!\\nScore: {bs:.4f} -> {score:.4f} (+{score-bs:.4f})\\nChange: {desc}")
        else:
            discarded += 1
            append_result(commit, metrics, "discard", desc)
            log(f"DISCARD score={score:.4f} (best={bs:.4f})")
            sh("git", "reset", "--soft", "HEAD~1")
            sh("git", "checkout", "--", "autoresearch/experiment.py")

        n += 1

        if n % NOTIFY_EVERY == 0:
            bs2, _ = best_score(parse_results())
            notify(f"autoresearch status\\nExperiments: {n} ({kept}K {discarded}D {crashed}C)\\nBest: {bs2:.4f}")

    log(f"Done. {n} experiments ({kept}K {discarded}D {crashed}C)")
    bs_final, _ = best_score(parse_results())
    notify(f"autoresearch finished!\\n{n} experiments in {args.max_hours}h\\nBest: {bs_final:.4f}")


if __name__ == "__main__":
    main()
