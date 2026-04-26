"""DeBERTa-v3-base fine-tuned classifier for adversarial prompt detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.models.debug_numerics import (
    DebugConfig,
    check_tensor_finite,
    dump_bad_batch,
    find_nonfinite_grads,
    find_nonfinite_params,
    log_label_distribution,
    log_param_stats,
    summarize_tensor,
    validate_labels,
)
from src.utils import build_sample_id

logger = logging.getLogger(__name__)


def _format_monitor_metrics(epoch_metrics: dict) -> str:
    """Render compact per-split monitor metrics for the epoch log line."""
    parts = []
    split_names = sorted({
        key.removesuffix("_f1")
        for key in epoch_metrics
        if (
            key.endswith("_f1")
            and not key.startswith("eval_")
            and f"{key.removesuffix('_f1')}_precision" in epoch_metrics
            and f"{key.removesuffix('_f1')}_recall" in epoch_metrics
        )
    })
    for split in split_names:
        parts.append(
            f"{split}_f1={epoch_metrics[f'{split}_f1']:.4f} "
            f"{split}_prec={epoch_metrics[f'{split}_precision']:.4f} "
            f"{split}_rec={epoch_metrics[f'{split}_recall']:.4f}"
        )
    if not parts:
        return ""
    return " | " + " | ".join(parts)


# ── Training result ──────────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    success: bool
    failed_reason: str | None = None
    first_bad_epoch: int | None = None
    first_bad_step: int | None = None
    first_bad_stage: str | None = None  # "forward" / "backward" / "optimizer_step" / "post_step"
    first_bad_param: str | None = None
    debug_artifact_paths: list[str] = field(default_factory=list)
    train_history: list[dict] | None = None
    best_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    stopped_early: bool = False


# ── Dataset ──────────────────────────────────────────────────────────────────


class PromptDataset(Dataset):
    """Tokenised dataset for prompt classification."""

    def __init__(self, tokenizer, texts: list[str], labels: list[int] | None, max_length: int):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class _LightningDeBERTaModule(pl.LightningModule):
    """Lightning wrapper that preserves the existing manual training semantics."""

    def __init__(
        self,
        classifier: "DeBERTaClassifier",
        class_weights: torch.Tensor,
        debug: DebugConfig,
        train_loader_len: int,
        total_steps: int,
        warmup_steps: int,
        on_train_batch_end: Callable[[dict], None] | None,
    ):
        super().__init__()
        self.classifier = classifier
        self.model = classifier.model
        self.tokenizer = classifier.tokenizer
        self.debug = debug
        self.train_loader_len = train_loader_len
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = classifier.max_grad_norm
        self.logging_steps = classifier.logging_steps
        self.learning_rate = classifier.learning_rate
        self.weight_decay = classifier.weight_decay
        self.automatic_optimization = False
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.epoch_loss = 0.0
        self.n_batches = 0
        self.failure_result: TrainingResult | None = None
        self._manual_scheduler = None
        self._train_batch_callback = on_train_batch_end

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._manual_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.warmup_steps,
            self.total_steps,
        )
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_loss = 0.0
        self.n_batches = 0

    def _dump_failure_artifact(
        self, epoch: int, step: int, stage: str, batch, loss, logits,
    ) -> list[str]:
        if not self.debug.save_bad_batch:
            return []
        texts = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        path = dump_bad_batch(
            "artifacts/deberta_classifier",
            epoch,
            step,
            stage,
            batch,
            loss,
            logits,
            texts,
        )
        return [str(path)]

    def _record_failure(
        self,
        reason: str,
        epoch: int,
        step: int,
        stage: str,
        batch,
        loss,
        logits,
        first_bad_param: str | None = None,
    ):
        logger.error(reason)
        self.failure_result = TrainingResult(
            success=False,
            failed_reason=reason,
            first_bad_epoch=epoch,
            first_bad_step=step,
            first_bad_stage=stage,
            first_bad_param=first_bad_param,
            debug_artifact_paths=self._dump_failure_artifact(epoch, step, stage, batch, loss, logits),
        )
        self.trainer.should_stop = True

    def training_step(self, batch, batch_idx):
        if self.failure_result is not None:
            return None

        epoch = int(self.current_epoch)
        verbose = self.debug.enabled and batch_idx < self.debug.first_n_batches

        if verbose:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    problems = check_tensor_finite(f"input.{key}", value)
                    if problems:
                        logger.warning(f"Pre-forward: {problems}")
            if self.debug.log_batch_text:
                texts = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                logger.info(f"Batch {batch_idx} texts: {texts[:3]}")

        optimizer = self.optimizers()
        scheduler = self._manual_scheduler
        optimizer.zero_grad()

        outputs = self.model(**batch)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, batch["labels"])

        forward_problems = check_tensor_finite("loss", loss) + check_tensor_finite("logits", logits)
        if verbose:
            logger.info(
                f"  [B] post-forward step {batch_idx}: "
                f"loss={loss.item():.6f} "
                f"logits={summarize_tensor('logits', logits)}"
            )
        if forward_problems:
            reason = (
                f"Non-finite at epoch {epoch} step {batch_idx} (forward): "
                f"{forward_problems}"
            )
            self._record_failure(reason, epoch, batch_idx, "forward", batch, loss, logits)
            return loss.detach()

        self.manual_backward(loss)

        bad_grads = find_nonfinite_grads(self.model)
        if bad_grads:
            first_name = bad_grads[0][0]
            reason = (
                f"Non-finite grad at epoch {epoch} step {batch_idx}: "
                f"{len(bad_grads)} params, first={first_name}"
            )
            self._record_failure(reason, epoch, batch_idx, "backward", batch, loss, logits, first_name)
            return loss.detach()

        if verbose and self.debug.log_param_stats:
            log_param_stats(self.model, logger)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bad_params = find_nonfinite_params(self.model)
        if bad_params:
            first_name = bad_params[0][0]
            reason = (
                f"Non-finite params after optimizer step at epoch {epoch} "
                f"step {batch_idx}: {len(bad_params)} params, first={first_name}"
            )
            self._record_failure(reason, epoch, batch_idx, "post_step", batch, loss, logits, first_name)
            return loss.detach()

        if verbose:
            logger.info(f"  [D] post-optimizer step {batch_idx}: params OK")

        self.epoch_loss += loss.item()
        self.n_batches += 1

        if batch_idx % self.logging_steps == 0:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else self.learning_rate
            batch_metrics = {
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "global_step": int(self.global_step),
                "train_loss_step": float(loss.detach().cpu()),
                "learning_rate": float(lr),
            }
            if self._train_batch_callback is not None:
                self._train_batch_callback(batch_metrics)
            logger.info(
                f"  epoch {epoch} step {batch_idx}/{self.train_loader_len} "
                f"loss={loss.item():.4f} lr={lr:.2e}"
            )
        return loss.detach()


class _DeBERTaEpochEndCallback(pl.Callback):
    """Runs the existing evaluation/history/checkpoint logic at Lightning epoch end."""

    def __init__(
        self,
        classifier: "DeBERTaClassifier",
        val_loader: DataLoader,
        monitor_loaders: dict[str, DataLoader],
        selected_device: torch.device,
        on_epoch_end: Callable[[dict], None] | None,
    ):
        self.classifier = classifier
        self.val_loader = val_loader
        self.monitor_loaders = monitor_loaders
        self.selected_device = selected_device
        self.on_epoch_end = on_epoch_end
        self.best_metric = float("-inf")
        self.patience_counter = 0
        self.stopped_early = False

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.failure_result is not None:
            return

        epoch = int(trainer.current_epoch)
        avg_loss = pl_module.epoch_loss / max(pl_module.n_batches, 1)

        val_metrics = self.classifier._evaluate(self.val_loader, self.selected_device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **{f"eval_{key}": value for key, value in val_metrics.items()},
        }
        for split_name, monitor_loader in self.monitor_loaders.items():
            split_metrics = self.classifier._evaluate(monitor_loader, self.selected_device)
            epoch_metrics.update({
                f"{split_name}_{key}": value
                for key, value in split_metrics.items()
            })

        self.classifier.train_history.append(epoch_metrics)
        if self.on_epoch_end is not None:
            self.on_epoch_end(epoch_metrics)

        current_metric = val_metrics[self.classifier.metric_for_best_model]
        logger.info(
            f"Epoch {epoch + 1}/{self.classifier.num_epochs} — "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_f1_benign={val_metrics['f1_benign']:.4f} "
            f"val_f1_adversarial={val_metrics['f1_adversarial']:.4f} "
            f"val_prec={val_metrics['precision']:.4f} "
            f"val_rec={val_metrics['recall']:.4f}"
            f"{_format_monitor_metrics(epoch_metrics)}"
        )

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            self.classifier._update_best_checkpoint(epoch + 1, current_metric)
        else:
            self.patience_counter += 1
            if (
                self.classifier.early_stopping_patience > 0
                and self.patience_counter >= self.classifier.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(patience={self.classifier.early_stopping_patience})"
                )
                self.stopped_early = True
                trainer.should_stop = True


# ── Classifier ───────────────────────────────────────────────────────────────


class DeBERTaClassifier:
    """Fine-tuned DeBERTa binary classifier."""

    def __init__(self, cfg: dict):
        dcfg = cfg["deberta"]
        self.model_name = dcfg["model_name"]
        self.max_length = dcfg["max_length"]
        self.num_epochs = dcfg["num_epochs"]
        self.batch_size = dcfg["batch_size"]
        self.eval_batch_size = dcfg.get("eval_batch_size", 8)
        self.learning_rate = dcfg["learning_rate"]
        self.warmup_ratio = dcfg["warmup_ratio"]
        self.weight_decay = dcfg["weight_decay"]
        self.logging_steps = max(int(dcfg.get("logging_steps", 50)), 1)
        self.early_stopping_patience = dcfg["early_stopping_patience"]
        self.metric_for_best_model = dcfg.get("metric_for_best_model", "f1")
        self.max_grad_norm = dcfg.get("max_grad_norm", 0.5)
        self.threshold = dcfg.get("threshold", 0.5)
        self.label_order = dcfg.get("label_order", ["benign", "adversarial"])

        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.train_history = None
        self.best_checkpoint = None

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        """Capture a CPU copy of the current model weights."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _update_best_checkpoint(self, epoch: int, metric_value: float):
        """Store best-model weights and metadata in memory."""
        self.best_checkpoint = {
            "epoch": epoch,
            "metric_name": self.metric_for_best_model,
            "metric_value": float(metric_value),
            "model_state_dict": self._snapshot_model_state(),
        }

    def _restore_best_checkpoint(self):
        """Load the best observed model weights back into the active model."""
        if self.best_checkpoint is None:
            return
        device = next(self.model.parameters()).device
        state_dict = {
            name: tensor.to(device)
            for name, tensor in self.best_checkpoint["model_state_dict"].items()
        }
        self.model.load_state_dict(state_dict)

    # ── Training ──────────────────────────────────────────────────────────────

    def _select_device(self, force_cpu: bool = False, device: str | None = None) -> torch.device:
        if device and device != "auto":
            return torch.device(device)
        if force_cpu:
            return torch.device("cpu")
        return torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")

    def _assert_finite_model(self):
        """Raise ValueError if model contains non-finite parameters."""
        bad = find_nonfinite_params(self.model)
        if bad:
            names = [name for name, _ in bad[:5]]
            raise ValueError(
                f"Model contains non-finite parameters ({len(bad)} total): {names}"
            )

    def _sanity_forward(self, train_loader, device, debug: DebugConfig,
                        train_texts: list[str]) -> TrainingResult:
        """Run forward-only passes to check for NaN without backward/optimizer."""
        logger.info(f"Running sanity forward on {debug.sanity_batches} batches...")
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(train_loader):
                if step >= debug.sanity_batches:
                    break

                batch = {k: v.to(device) for k, v in batch.items()}

                if debug.log_batch_text:
                    # Decode input_ids back to text for logging
                    texts = self.tokenizer.batch_decode(batch["input_ids"],
                                                        skip_special_tokens=True)
                    logger.info(f"Sanity batch {step} texts: {texts[:3]}")

                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                logger.info(f"Sanity batch {step}: loss={loss.item():.6f}")
                loss_summary = summarize_tensor("loss", loss)
                logits_summary = summarize_tensor("logits", logits)
                logger.info(f"  loss: {loss_summary}")
                logger.info(f"  logits: min={logits_summary.min:.4f} max={logits_summary.max:.4f} "
                            f"nan={logits_summary.has_nan} inf={logits_summary.has_inf}")

                problems = check_tensor_finite("loss", loss)
                problems.extend(check_tensor_finite("logits", logits))
                if problems:
                    reason = f"Sanity forward NaN at batch {step}: {problems}"
                    logger.error(reason)
                    return TrainingResult(
                        success=False,
                        failed_reason=reason,
                        first_bad_epoch=0,
                        first_bad_step=step,
                        first_bad_stage="forward",
                    )

        logger.info("Sanity forward passed — no NaN detected.")
        return TrainingResult(success=True)

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
              text_col: str, label_col: str = "label_binary",
              force_cpu: bool = False,
              device: str | None = None,
              debug: DebugConfig | None = None,
              monitor_dfs: dict[str, pd.DataFrame] | None = None,
              on_epoch_end: Callable[[dict], None] | None = None,
              on_train_batch_end: Callable[[dict], None] | None = None) -> TrainingResult:
        if debug is None:
            debug = DebugConfig()

        self.label2id = {lbl: i for i, lbl in enumerate(self.label_order)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_order),
            id2label=self.id2label,
            label2id=self.label2id,
            dtype=torch.float32,
        )

        selected_device = self._select_device(force_cpu, device)
        self.model.to(selected_device)
        logger.info(f"Training on device: {selected_device}")

        train_labels = [self.label2id[l] for l in df_train[label_col]]
        val_labels = [self.label2id[l] for l in df_val[label_col]]

        # Validate labels
        label_problems = validate_labels(train_labels, len(self.label_order))
        if label_problems:
            reason = f"Invalid training labels: {label_problems[:5]}"
            logger.error(reason)
            return TrainingResult(success=False, failed_reason=reason)

        log_label_distribution(train_labels, self.id2label, logger)

        # Compute class weights to handle imbalanced data
        classes = np.arange(len(self.label_order))
        weights = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {dict(zip(self.label_order, weights))}")

        train_texts = df_train[text_col].tolist()
        train_ds = PromptDataset(self.tokenizer, train_texts, train_labels, self.max_length)
        val_ds = PromptDataset(self.tokenizer, df_val[text_col].tolist(), val_labels, self.max_length)

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=collator, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=self.eval_batch_size, shuffle=False,
                                collate_fn=collator, pin_memory=False)
        monitor_loaders = {}
        for split_name, monitor_df in (monitor_dfs or {}).items():
            monitor_labels = [self.label2id[l] for l in monitor_df[label_col]]
            monitor_ds = PromptDataset(
                self.tokenizer,
                monitor_df[text_col].tolist(),
                monitor_labels,
                self.max_length,
            )
            monitor_loaders[split_name] = DataLoader(
                monitor_ds,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                pin_memory=False,
            )

        # Sanity forward-only mode
        if debug.sanity_forward_only:
            return self._sanity_forward(train_loader, selected_device, debug, train_texts)

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        if debug.enabled and debug.log_param_stats:
            log_param_stats(self.model, logger)

        self.train_history = []
        self.model.train()

        lightning_module = _LightningDeBERTaModule(
            classifier=self,
            class_weights=class_weights,
            debug=debug,
            train_loader_len=len(train_loader),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            on_train_batch_end=on_train_batch_end,
        )
        epoch_callback = _DeBERTaEpochEndCallback(
            classifier=self,
            val_loader=val_loader,
            monitor_loaders=monitor_loaders,
            selected_device=selected_device,
            on_epoch_end=on_epoch_end,
        )
        trainer = pl.Trainer(
            accelerator=selected_device.type,
            devices=1,
            max_epochs=self.num_epochs,
            callbacks=[epoch_callback],
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        logger.info(
            f"Starting Lightning trainer.fit: epochs={self.num_epochs}, "
            f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
        )
        trainer.fit(lightning_module, train_dataloaders=train_loader)

        if lightning_module.failure_result is not None:
            failure_result = lightning_module.failure_result
            logger.error(
                f"Stopping training due to numeric failure at epoch "
                f"{failure_result.first_bad_epoch}"
            )
            failure_result.train_history = self.train_history
            return failure_result

        self._restore_best_checkpoint()
        logger.info("Training complete.")
        return TrainingResult(
            success=True,
            train_history=self.train_history,
            best_epoch=self.best_checkpoint["epoch"] if self.best_checkpoint else None,
            best_metric_name=self.metric_for_best_model,
            best_metric_value=self.best_checkpoint["metric_value"] if self.best_checkpoint else None,
            stopped_early=epoch_callback.stopped_early,
        )

    def _evaluate(self, val_loader, device):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                preds = torch.argmax(logits, dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", pos_label=1),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary",
                                         pos_label=1, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary",
                                    pos_label=1, zero_division=0),
        }
        for label_id, label_name in self.id2label.items():
            metrics[f"f1_{label_name}"] = f1_score(
                all_labels,
                all_preds,
                labels=[label_id],
                average=None,
                zero_division=0,
            )[0]
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        # Check model health before inference
        self._assert_finite_model()

        self.model.eval()
        device = next(self.model.parameters()).device
        texts = df[text_col].tolist()

        ds = PromptDataset(self.tokenizer, texts, labels=None, max_length=self.max_length)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        all_probs = []
        batch_size = self.eval_batch_size

        for start in range(0, len(ds), batch_size):
            batch_items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = collator(batch_items)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                logits = self.model(**batch).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

            # Check for NaN in this batch
            nan_count = np.isnan(probs).any(axis=1).sum()
            if nan_count > 0:
                batch_idx = start // batch_size
                sample_indices = list(range(start, min(start + batch_size, len(ds))))
                nan_rows = np.where(np.isnan(probs).any(axis=1))[0]
                nan_sample_indices = [sample_indices[r] for r in nan_rows]
                nan_texts = [texts[i][:100] for i in nan_sample_indices]
                nan_logits = logits[nan_rows].cpu().numpy()

                # Enhanced diagnostics
                param_health = "unknown"
                try:
                    bad_params = find_nonfinite_params(self.model)
                    param_health = f"{len(bad_params)} non-finite params" if bad_params else "all finite"
                except Exception:
                    pass

                raise ValueError(
                    f"NaN detected in batch {batch_idx} ({nan_count}/{len(probs)} samples). "
                    f"Device: {device}, logits dtype: {logits.dtype}, "
                    f"Model params: {param_health}, "
                    f"NaN logits: {np.isnan(nan_logits).any()}, "
                    f"Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}], "
                    f"Logits summary: {summarize_tensor('logits', logits)}, "
                    f"Sample indices: {nan_sample_indices}, "
                    f"Sample texts (truncated): {nan_texts}"
                )

            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        # Apply threshold on adversarial probability instead of argmax
        adv_idx = self.label2id["adversarial"]
        adv_probs = all_probs[:, adv_idx]
        pred_indices = np.where(adv_probs >= self.threshold, adv_idx, 1 - adv_idx)
        results = pd.DataFrame({
            "deberta_pred_binary": [self.id2label[i] for i in pred_indices],
            "deberta_conf_binary": np.max(all_probs, axis=1),
        })
        for i, lbl in self.id2label.items():
            results[f"deberta_proba_binary_{lbl}"] = all_probs[:, i]

        return results

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, output_dir):
        self._assert_finite_model()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / "model"
        tokenizer_dir = output_dir / "tokenizer"
        # Always save in float32 to avoid corrupt fp16 weights from MPS
        self.model.float().save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)

        label_mapping = {"label2id": self.label2id, "id2label": {str(k): v for k, v in self.id2label.items()}}
        (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2))

        if self.train_history is not None:
            (output_dir / "train_history.json").write_text(json.dumps(self.train_history, indent=2))

        if self.best_checkpoint is not None:
            torch.save(
                {
                    "epoch": self.best_checkpoint["epoch"],
                    "metric_name": self.best_checkpoint["metric_name"],
                    "metric_value": self.best_checkpoint["metric_value"],
                    "model_state_dict": self.best_checkpoint["model_state_dict"],
                    "label2id": self.label2id,
                    "id2label": self.id2label,
                },
                output_dir / "best_checkpoint.pt",
            )
            checkpoint_metadata: dict[str, Any] = {
                "epoch": self.best_checkpoint["epoch"],
                "metric_name": self.best_checkpoint["metric_name"],
                "metric_value": self.best_checkpoint["metric_value"],
            }
            (output_dir / "best_checkpoint.json").write_text(json.dumps(checkpoint_metadata, indent=2))

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, output_dir, cfg: dict, force_cpu: bool = False, device: str | None = None):
        output_dir = Path(output_dir)

        instance = cls(cfg)
        label_mapping = json.loads((output_dir / "label_mapping.json").read_text())
        instance.label2id = label_mapping["label2id"]
        instance.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

        instance.tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            output_dir / "model", dtype=torch.float32,
        )

        selected_device = instance._select_device(force_cpu, device)
        instance.model.to(selected_device)
        logger.info(f"Model loaded on device: {selected_device}")

        # Warn if loaded model has non-finite params
        bad_params = find_nonfinite_params(instance.model)
        if bad_params:
            names = [name for name, _ in bad_params[:5]]
            logger.warning(f"Loaded model has {len(bad_params)} non-finite parameters: {names}")

        return instance
