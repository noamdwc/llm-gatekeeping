import hashlib

import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Organized output paths ───────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_EXTERNAL_DIR = DATA_DIR / "predictions_external"
RESEARCH_DIR = DATA_DIR / "research"
CALIBRATION_DIR = RESEARCH_DIR / "calibration"
RESEARCH_EXTERNAL_DIR = DATA_DIR / "research_external"
BASELINES_DIR = DATA_DIR / "baselines"
REPORTS_DIR = ROOT / "reports"
REPORTS_RESEARCH_DIR = REPORTS_DIR / "research"
REPORTS_EXTERNAL_DIR = REPORTS_DIR / "research_external"
REPORTS_ARTIFACTS_DIR = REPORTS_DIR / "artifacts"
REPORTS_BASELINES_DIR = REPORTS_DIR / "baselines"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        DATA_DIR, SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR, PREDICTIONS_EXTERNAL_DIR,
        RESEARCH_DIR, CALIBRATION_DIR, RESEARCH_EXTERNAL_DIR, BASELINES_DIR,
        REPORTS_DIR, REPORTS_RESEARCH_DIR, REPORTS_EXTERNAL_DIR, REPORTS_ARTIFACTS_DIR,
        REPORTS_BASELINES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def build_sample_id(text: str) -> str:
    """Deterministic ID for a modified sample, used to align prediction DataFrames."""
    return hashlib.md5(text.encode()).hexdigest()


def load_config(path: str = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
