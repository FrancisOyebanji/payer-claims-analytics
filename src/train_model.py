"""Train and evaluate the high-cost member prediction model.

Framing: predict whether a member will be in the top cost decile in year 2
using only year-1 utilization, demographics, chronic conditions, and SDOH.
This mirrors the prospective identification problem behind care-management
targeting: every member the model surfaces correctly is an intervention
opportunity before the cost is incurred.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import feature_columns

SEED = 42


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """Lift of the top-k% highest-scored members vs. base rate."""
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean() / y_true.mean())


def capture_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """Share of all true high-cost members captured in the top-k% of scores."""
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].sum() / y_true.sum())


def train_and_evaluate(feats: pd.DataFrame, fig_dir: str | Path) -> dict:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(feats)
    X = feats[cols].astype(float).to_numpy()
    y = feats["target_high_cost_y2"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    # Baseline: logistic regression on the single strongest heuristic feature set
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    baseline.fit(X_tr, y_tr)
    p_base = baseline.predict_proba(X_te)[:, 1]

    model = GradientBoostingClassifier(random_state=SEED, n_estimators=300,
                                       max_depth=3, learning_rate=0.05, subsample=0.8)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    results = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "base_rate": round(float(y.mean()), 4),
        "baseline_logreg": {
            "roc_auc": round(roc_auc_score(y_te, p_base), 4),
            "avg_precision": round(average_precision_score(y_te, p_base), 4),
        },
        "gradient_boosting": {
            "roc_auc": round(roc_auc_score(y_te, p), 4),
            "avg_precision": round(average_precision_score(y_te, p), 4),
            "lift_at_5pct": round(lift_at_k(y_te, p, 0.05), 2),
            "lift_at_10pct": round(lift_at_k(y_te, p, 0.10), 2),
            "capture_at_10pct": round(capture_at_k(y_te, p, 0.10), 3),
            "capture_at_20pct": round(capture_at_k(y_te, p, 0.20), 3),
        },
    }

    # --- Figures ---
    fpr_b, tpr_b, _ = roc_curve(y_te, p_base)
    fpr_m, tpr_m, _ = roc_curve(y_te, p)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_m, tpr_m, label=f"Gradient boosting (AUC {results['gradient_boosting']['roc_auc']})")
    plt.plot(fpr_b, tpr_b, "--", label=f"Logistic baseline (AUC {results['baseline_logreg']['roc_auc']})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("High-cost member prediction — ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=140); plt.close()

    imp = pd.Series(model.feature_importances_, index=cols).sort_values().tail(15)
    plt.figure(figsize=(7.5, 6))
    imp.plot.barh(color="#2749c9")
    plt.title("Top 15 feature importances (gradient boosting)")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=140); plt.close()

    ks = np.arange(0.02, 0.52, 0.02)
    plt.figure(figsize=(6.5, 5))
    plt.plot(ks * 100, [capture_at_k(y_te, p, k) * 100 for k in ks], label="Gradient boosting")
    plt.plot(ks * 100, [capture_at_k(y_te, p_base, k) * 100 for k in ks], "--", label="Logistic baseline")
    plt.plot(ks * 100, ks * 100, ":", color="gray", label="Random targeting")
    plt.xlabel("% of members targeted (ranked by model score)")
    plt.ylabel("% of future high-cost members captured")
    plt.title("Care-management targeting efficiency (gain curve)")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "gain_curve.png", dpi=140); plt.close()

    results["top_features"] = imp.sort_values(ascending=False).head(10).round(4).to_dict()
    return results


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    from features import build_features
    members = pd.read_csv(f"{data_dir}/members.csv")
    claims = pd.read_csv(f"{data_dir}/claims.csv")
    sdoh = pd.read_csv(f"{data_dir}/sdoh.csv")
    feats = build_features(members, claims, sdoh)
    res = train_and_evaluate(feats, "reports/figures")
    print(json.dumps(res, indent=2))
