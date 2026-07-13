"""End-to-end pipeline: generate data -> features -> model -> quality measures -> report inputs.

Usage:
    python src/run_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import generate_data
from features import build_features
from quality_measures import compute_measures, measures_by_plan
from train_model import train_and_evaluate

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")


def main() -> None:
    print("=== 1/4 Generating synthetic payer data ===")
    generate_data.main(str(DATA_DIR), n_members=8000)

    members = pd.read_csv(DATA_DIR / "members.csv")
    claims = pd.read_csv(DATA_DIR / "claims.csv")
    sdoh = pd.read_csv(DATA_DIR / "sdoh.csv")

    print("=== 2/4 Building features ===")
    feats = build_features(members, claims, sdoh)
    print(f"{feats.shape[0]:,} members x {feats.shape[1]} columns; "
          f"high-cost base rate: {feats.target_high_cost_y2.mean():.1%}")

    print("=== 3/4 Training high-cost member model ===")
    results = train_and_evaluate(feats, REPORT_DIR / "figures")
    (REPORT_DIR / "model_results.json").write_text(json.dumps(results, indent=2))
    gb = results["gradient_boosting"]
    print(f"ROC-AUC {gb['roc_auc']} | lift@5% {gb['lift_at_5pct']}x | "
          f"capture@20% {gb['capture_at_20pct']:.0%}")

    print("=== 4/4 Computing quality measures ===")
    overall = compute_measures(members, claims, year=2)
    by_plan = measures_by_plan(members, claims, year=2)
    overall.to_csv(REPORT_DIR / "quality_measures_overall.csv", index=False)
    by_plan.to_csv(REPORT_DIR / "quality_measures_by_plan.csv", index=False)
    print(overall.to_string(index=False))
    print(f"\nArtifacts written to {REPORT_DIR}/ (figures, model_results.json, measure CSVs)")


if __name__ == "__main__":
    main()
