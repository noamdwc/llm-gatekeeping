"""
CLI prediction tool — reads input text, outputs hierarchical classification + metadata.

Usage:
    # Single text from stdin
    echo "some text" | python -m src.cli.predict

    # From a file (one text per line)
    python -m src.cli.predict --input texts.txt

    # Use hybrid router (ML first, escalate to LLM if uncertain)
    python -m src.cli.predict --mode hybrid --input texts.txt

    # ML-only (no API calls)
    python -m src.cli.predict --mode ml --input texts.txt
"""

import argparse
import json
import sys

import dotenv

dotenv.load_dotenv()

from src.utils import load_config, MODELS_DIR


def predict_llm(texts: list[str], cfg: dict) -> list[dict]:
    """Predict using the LLM-only classifier."""
    from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier
    classifier = HierarchicalLLMClassifier(cfg)
    return classifier.predict_batch(texts, desc="Predicting (LLM)")


def predict_ml(texts: list[str], cfg: dict) -> list[dict]:
    """Predict using the ML-only classifier."""
    import pandas as pd
    from src.ml_classifier.ml_baseline import MLBaseline

    model_path = MODELS_DIR / "ml_baseline.pkl"
    if not model_path.exists():
        print(f"Error: ML model not found at {model_path}. Run ml_model stage first.", file=sys.stderr)
        sys.exit(1)

    ml = MLBaseline(cfg)
    ml.load(str(model_path))

    text_col = cfg["dataset"]["text_col"]
    df = pd.DataFrame({text_col: texts})
    preds = ml.predict(df, text_col)

    results = []
    for i in range(len(texts)):
        row = preds.iloc[i]
        results.append({
            "label_binary": row["pred_label_binary"],
            "label_category": row["pred_label_category"],
            "label_type": row["pred_label_type"],
            "confidence_binary": float(row["confidence_label_binary"]),
            "confidence_category": float(row["confidence_label_category"]),
            "confidence_type": float(row["confidence_label_type"]),
            "routed_to": "ml",
        })
    return results


def predict_hybrid(texts: list[str], cfg: dict) -> list[dict]:
    """Predict using the hybrid ML+LLM router."""
    import pandas as pd
    from src.ml_classifier.ml_baseline import MLBaseline
    from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier
    from src.hybrid_router import HybridRouter

    model_path = MODELS_DIR / "ml_baseline.pkl"
    if not model_path.exists():
        print(f"Error: ML model not found at {model_path}. Run ml_model stage first.", file=sys.stderr)
        sys.exit(1)

    ml = MLBaseline(cfg)
    ml.load(str(model_path))
    llm = HierarchicalLLMClassifier(cfg)
    router = HybridRouter(ml, llm, cfg)

    text_col = cfg["dataset"]["text_col"]
    df = pd.DataFrame({text_col: texts})
    return router.predict_batch(df, text_col, desc="Predicting (hybrid)")


def main():
    parser = argparse.ArgumentParser(description="Predict attack type for input text")
    parser.add_argument("--input", default=None, help="Input file (one text per line). Defaults to stdin.")
    parser.add_argument("--mode", choices=["llm", "ml", "hybrid"], default="hybrid",
                        help="Prediction mode (default: hybrid)")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Read input texts
    if args.input:
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    # Predict
    dispatch = {"llm": predict_llm, "ml": predict_ml, "hybrid": predict_hybrid}
    results = dispatch[args.mode](texts, cfg)

    # Output
    for text, result in zip(texts, results):
        output = {
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            **result,
        }
        indent = 2 if args.pretty else None
        print(json.dumps(output, indent=indent, default=str))


if __name__ == "__main__":
    main()

