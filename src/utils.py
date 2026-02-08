import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)