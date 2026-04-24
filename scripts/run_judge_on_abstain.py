"""Run the judge (70B) on all abstain samples from the test trace.

Reads the 180 abstain samples, reconstructs the classifier output from
llm_predictions, and calls the judge. Saves results to a parquet file
for offline analysis (no true labels used in the script).
"""

import json
import sys
from pathlib import Path

import dotenv
import pandas as pd
from tqdm import tqdm

dotenv.load_dotenv()

from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier
from src.utils import load_config

OUTPUT_PATH = Path("data/processed/research/judge_on_abstain.parquet")


def main():
    cfg = load_config()
    llm = HierarchicalLLMClassifier(cfg)

    # Load abstain sample IDs from trace
    trace = pd.read_parquet("data/processed/research/hybrid_margin_trace_test.parquet")
    abstain_ids = set(trace.loc[trace["route"] == "abstain", "sample_id"])

    # Load LLM predictions to get classifier output + text
    llm_preds = pd.read_parquet("data/processed/predictions/llm_predictions_test.parquet")
    abstain_df = llm_preds[llm_preds["sample_id"].isin(abstain_ids)].copy()
    print(f"Running judge on {len(abstain_df)} abstain samples...")

    results = []
    for _, row in tqdm(abstain_df.iterrows(), total=len(abstain_df)):
        # Reconstruct classifier output dict as the judge expects it
        clf_output = {
            "label": row["clf_label"],
            "confidence": row["clf_confidence"],
            "evidence": row.get("clf_evidence", ""),
            "nlp_attack_type": row.get("clf_nlp_attack_type", "none"),
        }

        text = row["modified_sample"]
        judge_result = llm.judge(text, clf_output)

        results.append({
            "sample_id": row["sample_id"],
            "judge_independent_label": judge_result.get("independent_label"),
            "judge_independent_confidence": judge_result.get("independent_confidence"),
            "judge_computed_decision": judge_result.get("computed_decision"),
            "judge_benign_task_override": judge_result.get("judge_benign_task_override"),
            "judge_override_reason": judge_result.get("judge_override_reason"),
            "judge_final_label": judge_result.get("final_label"),
            "judge_final_confidence": judge_result.get("final_confidence"),
            "judge_raw_response_text": judge_result.get("_raw_response_text"),
            "judge_parse_success": judge_result.get("_parse_success", False),
            "judge_token_logprobs": json.dumps(judge_result.get("_token_logprobs")),
        })

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} judge results to {OUTPUT_PATH}")

    # Summary (no true labels)
    print(f"\nJudge independent label distribution:")
    print(results_df["judge_independent_label"].value_counts(dropna=False))
    print(f"\nJudge computed decision:")
    print(results_df["judge_computed_decision"].value_counts(dropna=False))
    print(f"\nBenign task override:")
    print(results_df["judge_benign_task_override"].value_counts(dropna=False))
    print(f"\nUsage: {llm.usage.to_dict()}")


if __name__ == "__main__":
    main()
