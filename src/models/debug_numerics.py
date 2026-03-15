"""Reusable numeric debugging helpers for PyTorch training loops.

Standalone module — no project-specific imports.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class TensorSummary:
    name: str
    shape: tuple
    dtype: str
    device: str
    has_nan: bool
    has_inf: bool
    min: float
    max: float
    mean: float
    std: float


@dataclass
class DebugConfig:
    enabled: bool = False
    first_n_batches: int = 0
    save_bad_batch: bool = False
    stop_on_nan: bool = True
    log_param_stats: bool = False
    log_batch_text: bool = False
    sanity_forward_only: bool = False
    sanity_batches: int = 3


# ── Tensor inspection ────────────────────────────────────────────────────────


def summarize_tensor(name: str, tensor: torch.Tensor) -> TensorSummary:
    """Compute summary statistics for a tensor."""
    t = tensor.detach().float()
    return TensorSummary(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        has_nan=bool(torch.isnan(t).any()),
        has_inf=bool(torch.isinf(t).any()),
        min=float(t.min()) if t.numel() > 0 else float("nan"),
        max=float(t.max()) if t.numel() > 0 else float("nan"),
        mean=float(t.mean()) if t.numel() > 0 else float("nan"),
        std=float(t.std()) if t.numel() > 0 else float("nan"),
    )


def check_tensor_finite(name: str, tensor: torch.Tensor) -> list[str]:
    """Return list of problems found (empty = OK)."""
    problems = []
    if torch.isnan(tensor).any():
        problems.append(f"{name}: contains NaN")
    if torch.isinf(tensor).any():
        problems.append(f"{name}: contains Inf")
    return problems


def find_nonfinite_grads(model: torch.nn.Module) -> list[tuple[str, TensorSummary]]:
    """Find all parameters with non-finite gradients."""
    bad = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append((name, summarize_tensor(name, p.grad)))
    return bad


def find_nonfinite_params(model: torch.nn.Module) -> list[tuple[str, TensorSummary]]:
    """Find all parameters with non-finite values."""
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p.data).all():
            bad.append((name, summarize_tensor(name, p.data)))
    return bad


# ── Label validation ─────────────────────────────────────────────────────────


def validate_labels(labels: list, num_labels: int) -> list[str]:
    """Check labels are valid integer class indices. Returns list of problems."""
    problems = []
    for i, lbl in enumerate(labels):
        if isinstance(lbl, float):
            if lbl != lbl:  # NaN check
                problems.append(f"Label at index {i} is NaN")
                continue
            if lbl != int(lbl):
                problems.append(f"Label at index {i} is non-integer float: {lbl}")
                continue
        try:
            val = int(lbl)
        except (ValueError, TypeError):
            problems.append(f"Label at index {i} is not int-compatible: {lbl!r}")
            continue
        if val < 0:
            problems.append(f"Label at index {i} is negative: {val}")
        elif val >= num_labels:
            problems.append(f"Label at index {i} is out of range: {val} (num_labels={num_labels})")
    return problems


def log_label_distribution(labels: list, id2label: dict, log: logging.Logger):
    """Log the distribution of labels."""
    from collections import Counter
    counts = Counter(labels)
    log.info("Label distribution:")
    for label_id in sorted(counts.keys()):
        name = id2label.get(label_id, f"unknown({label_id})")
        log.info(f"  {name} (id={label_id}): {counts[label_id]}")


# ── Parameter stats ──────────────────────────────────────────────────────────


def log_param_stats(model: torch.nn.Module, log: logging.Logger, top_k: int = 5):
    """Log mean/std/min/max for the top-k largest parameters."""
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params.sort(key=lambda x: x[1].numel(), reverse=True)
    log.info(f"Parameter stats (top {top_k} by size):")
    for name, p in params[:top_k]:
        s = summarize_tensor(name, p.data)
        log.info(f"  {name}: shape={s.shape} mean={s.mean:.6f} std={s.std:.6f} "
                 f"min={s.min:.6f} max={s.max:.6f} nan={s.has_nan} inf={s.has_inf}")


# ── Bad batch dump ───────────────────────────────────────────────────────────


def dump_bad_batch(
    output_dir: str | Path,
    epoch: int,
    step: int,
    stage: str,
    batch: dict,
    loss: torch.Tensor | None,
    logits: torch.Tensor | None,
    texts: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Save a bad batch to disk for post-mortem analysis.

    Creates ``output_dir/debug/bad_batch_e{E}_s{S}/`` with:
    - ``metadata.json`` — epoch, step, stage, texts, extra info
    - ``batch.pt`` — input tensors
    - ``loss.pt`` / ``logits.pt`` — if provided
    """
    output_dir = Path(output_dir)
    dump_dir = output_dir / "debug" / f"bad_batch_e{epoch}_s{step}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "epoch": epoch,
        "step": step,
        "stage": stage,
        "texts": texts,
        "extra": extra or {},
    }
    if loss is not None:
        metadata["loss_value"] = float(loss.detach().cpu()) if loss.numel() == 1 else None
    if logits is not None:
        s = summarize_tensor("logits", logits)
        metadata["logits_summary"] = asdict(s)

    (dump_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Save tensors
    batch_cpu = {k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    torch.save(batch_cpu, dump_dir / "batch.pt")
    if loss is not None:
        torch.save(loss.detach().cpu(), dump_dir / "loss.pt")
    if logits is not None:
        torch.save(logits.detach().cpu(), dump_dir / "logits.pt")

    logger.info(f"Bad batch dumped to {dump_dir}")
    return dump_dir
