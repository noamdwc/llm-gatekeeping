"""
LLM classifier with classifier + judge pattern:
  Stage 1: Classifier — single call predicts binary + type together
  Stage 2: Judge — conditional higher-quality model call on low-confidence predictions

Usage:
    python -m src.llm_classifier [--config configs/default.yaml] [--split test] [--limit N]
    python -m src.llm_classifier --split val --research --no-wandb --target-rpm 30 --max-concurrency 2
"""

import argparse
import json
import dotenv
import random
import threading
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import wandb

from src.llm_classifier.constants import (
    _HARD_BENIGN_EXAMPLES
)
from src.embeddings import ExemplarBank
from src.llm_classifier.rate_limiter import APIRateLimiter
from src.utils import load_config, build_sample_id, SPLITS_DIR, PREDICTIONS_DIR
from src.llm_classifier.hierarchical_llm_classifier import HierarchicalLLMClassifier
from src.llm_classifier.lllm_call_checkpointing import (checkpoint_path,
                                                        build_research_row,
                                                        load_checkpoint,
                                                        append_checkpoint,
                                                        finalize_checkpoint,
                                                    )

dotenv.load_dotenv()

def _build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hierarchical LLM classifier")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", help="Which split to evaluate on")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples to classify (default: full split)"
    )
    parser.add_argument("--output", default=None, help="Output predictions CSV path")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic few-shot retrieval")
    parser.add_argument(
        "--bank-path", default=None, help="Path to exemplar bank pickle (built if not exists)"
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Save research-grade parquet with full prediction columns",
    )

    # Rate limiting / concurrency controls
    rate_group = parser.add_argument_group("rate-limiting", "API rate-limit controls")
    rate_group.add_argument(
        "--target-rpm",
        type=float,
        default=None,
        help="Target requests per minute (default: config llm.target_rpm or 60)",
    )
    rate_group.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max parallel API workers (default: config llm.max_concurrency)",
    )
    rate_group.add_argument(
        "--cooldown-on-429",
        type=float,
        default=None,
        help="Global cooldown seconds after a 429 (default: config llm.cooldown_on_429 or 15)",
    )

    # Resume / checkpoint controls
    resume_group = parser.add_argument_group("resume", "Checkpoint / resume controls")
    resume_group.add_argument(
        "--no-resume", action="store_true", help="Start fresh, ignore existing checkpoint"
    )
    resume_group.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save checkpoint every N samples (default: config llm.checkpoint_every or 200)",
    )
    return parser


# ---------------------------------------------------------------------------
# Few-shot builder
# ---------------------------------------------------------------------------
def build_few_shot_examples(df: pd.DataFrame, cfg: dict) -> tuple[list[tuple[str, str, str]], list]:
    """Build static few-shot examples as (benign_text, attack_text, attack_type) pairs.

    When cfg["llm"]["few_shot"]["include_hard_benign"] is True, prepends 3 extra
    pairs where the benign_text is a hardcoded hard-benign example (instruction-like
    but genuinely benign), paired with a randomly sampled attack text. This teaches
    the LLM to correctly label near-miss benign content.
    """
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]
    n_unicode = cfg["llm"]["few_shot"]["unicode"]
    n_nlp = cfg["llm"]["few_shot"]["nlp"]
    unicode_set = set(cfg["labels"]["unicode_attacks"])
    include_hard_benign = cfg.get("llm", {}).get("few_shot", {}).get("include_hard_benign", False)

    pairs = []
    used_ids = []
    rng = random.Random(42)

    benign_pool = df.loc[df[label_col] == "benign", text_col].tolist()

    for attack_type in cfg["labels"]["unicode_attacks"] + cfg["labels"]["nlp_attacks"]:
        n = n_unicode if attack_type in unicode_set else n_nlp
        pool = df.loc[df[label_col] == attack_type, text_col]
        if len(pool) < n:
            n = len(pool)
        if n == 0:
            continue
        attack_samples = pool.sample(n=n, random_state=42)
        used_ids.extend(attack_samples.index.tolist())
        for attack_text in attack_samples.tolist():
            benign_text = rng.choice(benign_pool) if benign_pool else ""
            pairs.append((benign_text, attack_text, attack_type))

    # Prepend hard-benign pairs when enabled (pairs hard benign text with sampled attack)
    if include_hard_benign:
        attack_pool = df.loc[df[label_col] != "benign"]
        for i, hard_benign_text in enumerate(_HARD_BENIGN_EXAMPLES):
            if len(attack_pool) > 0:
                sample = attack_pool.sample(n=1, random_state=42 + i)
                attack_text = sample[text_col].iloc[0]
                attack_type = sample[label_col].iloc[0]
                pairs.insert(i, (hard_benign_text, attack_text, attack_type))
                used_ids.extend(sample.index.tolist())

    return pairs, used_ids


def main():
    args = _build_args_parser().parse_args()
    cfg = load_config(args.config)
    llm_cfg = cfg.get("llm", {})

    # Apply CLI overrides to config
    if args.max_concurrency is not None:
        cfg["llm"]["max_concurrency"] = args.max_concurrency
    if args.target_rpm is not None:
        cfg["llm"]["target_rpm"] = args.target_rpm
    if args.cooldown_on_429 is not None:
        cfg["llm"]["cooldown_on_429"] = args.cooldown_on_429

    checkpoint_every = args.checkpoint_every or int(llm_cfg.get("checkpoint_every", 200))
    enable_resume = not args.no_resume and llm_cfg.get("resume", True)

    # Build rate limiter from resolved config
    target_rpm = float(cfg["llm"].get("target_rpm", 60))
    max_concurrency = int(cfg["llm"].get("max_concurrency", 8))
    cooldown = float(cfg["llm"].get("cooldown_on_429", 15))
    rate_limiter = APIRateLimiter(
        target_rpm=target_rpm,
        max_concurrency=max_concurrency,
        cooldown_on_429=cooldown,
    )

    print(
        f"Rate limiter: target_rpm={target_rpm}, max_concurrency={max_concurrency}, "
        f"cooldown_on_429={cooldown}s"
    )

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name=f"llm-{cfg['llm']['model']}-{args.split}{'_dynamic' if args.dynamic else ''}",
            config={
                "model": cfg["llm"]["model"],
                "split": args.split,
                "limit": args.limit,
                "few_shot_unicode": cfg["llm"]["few_shot"]["unicode"],
                "few_shot_nlp": cfg["llm"]["few_shot"]["nlp"],
                "dynamic": args.dynamic,
                "target_rpm": target_rpm,
                "max_concurrency": max_concurrency,
                "cooldown_on_429": cooldown,
            },
        )

    # Load train for few-shot, eval split for evaluation
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    df_eval = pd.read_parquet(SPLITS_DIR / f"{args.split}.parquet")

    if args.limit and args.limit < len(df_eval):
        df_eval = df_eval.sample(n=args.limit, random_state=42)

    text_col = cfg["dataset"]["text_col"]

    # Build sample_id column for resume tracking
    df_eval = df_eval.copy()
    df_eval["sample_id"] = df_eval[text_col].apply(build_sample_id)

    # Resume: load checkpoint and filter out completed samples
    completed_ids: set[str] = set()
    if enable_resume:
        completed_ids = load_checkpoint(args.split)
        if completed_ids:
            n_before = len(df_eval)
            df_eval = df_eval[~df_eval["sample_id"].isin(completed_ids)].reset_index(drop=True)
            print(
                f"Resuming: {len(completed_ids)} already done, {n_before - len(df_eval)} skipped, "
                f"{len(df_eval)} remaining"
            )
    elif not args.no_resume:
        # Starting fresh: remove stale checkpoint
        cp = checkpoint_path(args.split)
        if cp.exists():
            cp.unlink()

    if len(df_eval) == 0:
        print("All samples already completed. Use --no-resume to re-run.")
        # Finalize if research mode
        if args.research:
            out_path = str(PREDICTIONS_DIR / f"llm_predictions_{args.split}.parquet")
            finalize_checkpoint(args.split, out_path)
            print(f"Research predictions saved → {out_path}")
        return

    # Build few-shot from train (static) or exemplar bank (dynamic)
    exemplar_bank = None
    few_shot = []

    if args.dynamic:
        bank_path = args.bank_path or str(PREDICTIONS_DIR / "exemplar_bank.pkl")
        if Path(bank_path).exists():
            print(f"Loading exemplar bank from {bank_path}")
            exemplar_bank = ExemplarBank.load(bank_path)
        else:
            print("Building exemplar bank (this may take a minute)...")
            exemplar_bank = ExemplarBank.build(df_train, cfg)
            exemplar_bank.save(bank_path)
        print(f"Exemplar bank: {exemplar_bank}")
    else:
        few_shot, _ = build_few_shot_examples(df_train, cfg)
        print(f"Few-shot examples: {len(few_shot)} pairs")

    # Classify with checkpointing
    classifier = HierarchicalLLMClassifier(
        cfg,
        few_shot,
        dynamic=args.dynamic,
        exemplar_bank=exemplar_bank,
        rate_limiter=rate_limiter,
    )

    texts = df_eval[text_col].tolist()
    sample_ids = df_eval["sample_id"].tolist()

    # Ground-truth columns for checkpoint rows
    gt_col_names = [
        "modified_sample",
        "original_sample",
        "attack_name",
        "label_binary",
        "label_category",
        "label_type",
        "prompt_hash",
        "benign_source",
        "is_synthetic_benign",
    ]
    gt_cols = [c for c in gt_col_names if c in df_eval.columns]

    # Accumulate checkpoint buffer
    checkpoint_buffer: list[dict] = []
    checkpoint_lock = threading.Lock()
    completed_count = [0]  # mutable counter for thread-safe increment

    def on_result(idx: int, result: dict):
        row = build_research_row(result)
        # Add ground-truth and sample_id
        row["sample_id"] = sample_ids[idx]
        for col in gt_cols:
            row[col] = df_eval.iloc[idx][col]
        with checkpoint_lock:
            checkpoint_buffer.append(row)
            completed_count[0] += 1
            if completed_count[0] % checkpoint_every == 0:
                append_checkpoint(args.split, checkpoint_buffer.copy())
                checkpoint_buffer.clear()
                print(f"\n  [checkpoint] saved {completed_count[0] + len(completed_ids)} total")

    results = classifier.predict_batch(
        texts,
        on_result=on_result,
        max_workers=max_concurrency,
    )

    # Flush remaining buffer
    with checkpoint_lock:
        if checkpoint_buffer:
            append_checkpoint(args.split, checkpoint_buffer)
            checkpoint_buffer.clear()

    # Save final output
    if args.research:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(PREDICTIONS_DIR / f"llm_predictions_{args.split}.parquet")
        finalize_checkpoint(args.split, out_path)
        n_total = len(completed_ids) + len(results)
        print(f"\nResearch predictions saved → {out_path} ({n_total} samples)")
    else:
        # Legacy mode: save CSV
        preds = pd.DataFrame(results)
        df_out = pd.concat([df_eval.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        out_path = args.output or str(PREDICTIONS_DIR / f"predictions_{args.split}.csv")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\nPredictions saved → {out_path}")

    # Print usage + rate limiter stats
    usage = classifier.usage.to_dict()
    limiter_stats = rate_limiter.stats.to_dict()
    combined_stats = {**usage, "rate_limiter": limiter_stats}
    print(f"\nUsage stats: {json.dumps(combined_stats, indent=2)}")

    if wandb.run is not None:
        wandb.log(
            {
                **usage,
                **{f"limiter/{k}": v for k, v in limiter_stats.items() if not isinstance(v, dict)},
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
