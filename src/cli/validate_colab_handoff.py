"""Validate manual Colab local-LLM classifier handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cli.colab_handoff_schema import REQUIRED_COLUMNS
from src.utils import PREDICTIONS_DIR, PREDICTIONS_EXTERNAL_DIR, REPORTS_DIR, load_config


DEFAULT_MAIN_SPLITS = ["val", "test", "unseen_val", "unseen_test", "safeguard_test"]
DEFAULT_OUTPUT = REPORTS_DIR / "colab_handoff_validation.json"


def _classifier_path(predictions_dir: Path, split: str) -> Path:
    return predictions_dir / f"llm_predictions_{split}_colab_local_classifier.parquet"


def _deberta_path(predictions_dir: Path, split: str) -> Path:
    return predictions_dir / f"deberta_predictions_{split}.parquet"


def _external_classifier_path(predictions_external_dir: Path, dataset: str) -> Path:
    return predictions_external_dir / f"llm_predictions_external_{dataset}.parquet"


def _external_deberta_path(predictions_external_dir: Path, dataset: str) -> Path:
    return predictions_external_dir / f"deberta_predictions_external_{dataset}.parquet"


def _require_path(path: Path, *, role: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {role}: {path}. "
        "Run notebooks/colab_local_llm_classifier.ipynb for manual handoff "
        "artifacts, or run the upstream DVC stage for DVC-produced artifacts."
    )


def _require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def validate_artifact_pair(
    *,
    name: str,
    classifier_path: Path,
    deberta_path: Path,
) -> dict[str, Any]:
    _require_path(classifier_path, role="manual Colab handoff artifact")
    _require_path(deberta_path, role="DVC-produced DeBERTa prediction artifact")

    classifier = pd.read_parquet(classifier_path)
    deberta = pd.read_parquet(deberta_path)

    _require_columns(classifier, classifier_path, list(REQUIRED_COLUMNS))
    _require_columns(
        deberta,
        deberta_path,
        ["sample_id", "deberta_proba_binary_adversarial"],
    )

    judge_columns = sorted(column for column in classifier.columns if column.startswith("judge_"))
    if judge_columns:
        raise ValueError(
            f"{classifier_path} must be classifier-only; " f"found judge columns: {judge_columns}"
        )

    invalid_stage_rows = classifier[classifier["llm_stages_run"] != 1]
    if not invalid_stage_rows.empty:
        raise ValueError(
            f"{classifier_path} must have llm_stages_run == 1 for every row; "
            f"found {len(invalid_stage_rows)} invalid rows"
        )

    classifier_ids = classifier[["sample_id"]].drop_duplicates()
    deberta_ids = deberta[["sample_id"]].drop_duplicates()
    joined = classifier_ids.merge(deberta_ids, on="sample_id", how="inner")
    dropped_classifier_only = len(classifier_ids) - len(joined)
    dropped_deberta_only = len(deberta_ids) - len(joined)
    if len(joined) == 0 or dropped_classifier_only or dropped_deberta_only:
        raise ValueError(
            f"{name} join mismatch between {classifier_path} and {deberta_path}: "
            f"joined={len(joined)}, classifier_only={dropped_classifier_only}, "
            f"deberta_only={dropped_deberta_only}"
        )

    return {
        "name": name,
        "classifier_path": str(classifier_path),
        "deberta_path": str(deberta_path),
        "rows_classifier": int(len(classifier)),
        "rows_deberta": int(len(deberta)),
        "rows_joined": int(len(joined)),
        "rows_dropped_classifier_only": int(dropped_classifier_only),
        "rows_dropped_deberta_only": int(dropped_deberta_only),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Colab classifier handoff artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument(
        "--predictions-external-dir",
        type=Path,
        default=PREDICTIONS_EXTERNAL_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    artifacts: list[dict[str, Any]] = []
    for split in DEFAULT_MAIN_SPLITS:
        artifacts.append(
            validate_artifact_pair(
                name=split,
                classifier_path=_classifier_path(args.predictions_dir, split),
                deberta_path=_deberta_path(args.predictions_dir, split),
            )
        )

    for dataset in cfg.get("external_datasets", {}):
        artifacts.append(
            validate_artifact_pair(
                name=f"external_{dataset}",
                classifier_path=_external_classifier_path(
                    args.predictions_external_dir,
                    dataset,
                ),
                deberta_path=_external_deberta_path(
                    args.predictions_external_dir,
                    dataset,
                ),
            )
        )

    payload = {"ok": True, "artifacts": artifacts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Validated {len(artifacts)} Colab handoff artifacts -> {args.output}")


if __name__ == "__main__":
    main()
