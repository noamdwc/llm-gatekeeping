"""DeBERTa-v3-base fine-tuned classifier for adversarial prompt detection."""

import json
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.utils import build_sample_id

logger = logging.getLogger(__name__)


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
            item["labels"] = self.labels[idx]
        return item


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
        self.early_stopping_patience = dcfg["early_stopping_patience"]
        self.max_grad_norm = dcfg.get("max_grad_norm", 0.5)
        self.label_order = dcfg.get("label_order", ["benign", "adversarial"])

        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.train_history = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
              text_col: str, label_col: str = "label_binary"):
        self.label2id = {lbl: i for i, lbl in enumerate(self.label_order)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_order),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"Training on device: {device}")

        train_labels = [self.label2id[l] for l in df_train[label_col]]
        val_labels = [self.label2id[l] for l in df_val[label_col]]

        train_ds = PromptDataset(self.tokenizer, df_train[text_col].tolist(), train_labels, self.max_length)
        val_ds = PromptDataset(self.tokenizer, df_val[text_col].tolist(), val_labels, self.max_length)

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=collator, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=self.eval_batch_size, shuffle=False,
                                collate_fn=collator, pin_memory=False)

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        self.train_history = []
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            nan_hit = False

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss

                if not torch.isfinite(loss):
                    logger.error(f"Non-finite loss at epoch {epoch} step {step}, "
                                 f"loss={loss.item()}")
                    nan_hit = True
                    break

                loss.backward()

                # Check for NaN gradients
                for name, p in self.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        logger.error(f"Bad grad in {name} at epoch {epoch} step {step}")
                        nan_hit = True
                        break

                if nan_hit:
                    break

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

                if step % 50 == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(f"  epoch {epoch} step {step}/{len(train_loader)} "
                                f"loss={loss.item():.4f} lr={lr:.2e}")

            if nan_hit:
                logger.error(f"Stopping training due to NaN at epoch {epoch}")
                break

            avg_loss = epoch_loss / max(n_batches, 1)

            # Evaluate
            val_metrics = self._evaluate(val_loader, device)
            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **{f"eval_{k}": v for k, v in val_metrics.items()},
            })

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} — "
                f"train_loss={avg_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f}"
            )

            # Early stopping on F1
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (patience={self.early_stopping_patience})")
                    break

        logger.info("Training complete.")

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

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", pos_label=1),
            "precision": precision_score(all_labels, all_preds, average="binary",
                                         pos_label=1, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary",
                                    pos_label=1, zero_division=0),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
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
                raise ValueError(
                    f"NaN detected in batch {batch_idx} ({nan_count}/{len(probs)} samples). "
                    f"Device: {device}, logits dtype: {logits.dtype}, "
                    f"NaN logits: {np.isnan(nan_logits).any()}, "
                    f"Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}], "
                    f"Sample indices: {nan_sample_indices}, "
                    f"Sample texts (truncated): {nan_texts}"
                )

            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        results = pd.DataFrame({
            "deberta_pred_binary": [self.id2label[i] for i in np.argmax(all_probs, axis=1)],
            "deberta_conf_binary": np.max(all_probs, axis=1),
        })
        for i, lbl in self.id2label.items():
            results[f"deberta_proba_binary_{lbl}"] = all_probs[:, i]

        return results

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, output_dir):
        from pathlib import Path
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

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, output_dir, cfg: dict):
        from pathlib import Path
        output_dir = Path(output_dir)

        instance = cls(cfg)
        label_mapping = json.loads((output_dir / "label_mapping.json").read_text())
        instance.label2id = label_mapping["label2id"]
        instance.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

        instance.tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            output_dir / "model", dtype=torch.float32,
        )

        return instance
