"""Feature engineering: year-1 claims + demographics + SDOH -> member-level features.

Target: whether a member lands in the top cost decile in year 2
(the standard "high-cost claimant" framing used in payer analytics).
"""
from __future__ import annotations

import pandas as pd

CONDITIONS = ["diabetes", "chf", "copd", "ckd", "depression", "hypertension"]


def build_features(members: pd.DataFrame, claims: pd.DataFrame, sdoh: pd.DataFrame) -> pd.DataFrame:
    y1 = claims[claims.plan_year == 1]
    y2 = claims[claims.plan_year == 2]

    # --- Year-1 utilization features ---
    agg = y1.groupby("member_id").agg(
        y1_total_paid=("paid_amount", "sum"),
        y1_claim_count=("paid_amount", "size"),
        y1_avg_claim=("paid_amount", "mean"),
        y1_max_claim=("paid_amount", "max"),
    )
    by_cat = (y1.pivot_table(index="member_id", columns="service_category",
                             values="paid_amount", aggfunc="sum", fill_value=0.0)
                .add_prefix("y1_paid_"))
    er_visits = (y1[y1.service_category == "emergency"]
                 .groupby("member_id").size().rename("y1_er_visits"))
    ip_admits = (y1[y1.service_category == "inpatient"]
                 .groupby("member_id").size().rename("y1_ip_admits"))

    feats = (members.set_index("member_id")
             .join([agg, by_cat, er_visits, ip_admits])
             .fillna({"y1_total_paid": 0, "y1_claim_count": 0, "y1_avg_claim": 0,
                      "y1_max_claim": 0, "y1_er_visits": 0, "y1_ip_admits": 0}))
    for col in feats.columns:
        if col.startswith("y1_paid_"):
            feats[col] = feats[col].fillna(0.0)

    # --- SDOH join ---
    feats = feats.reset_index().merge(sdoh, on="zip3", how="left").set_index("member_id")

    # --- Derived clinical burden ---
    feats["condition_count"] = feats[[f"cond_{c}" for c in CONDITIONS]].sum(axis=1)

    # --- Encodings ---
    feats["sex_f"] = (feats["sex"] == "F").astype(int)
    feats = pd.get_dummies(feats, columns=["plan_type"], prefix="plan")

    # --- Target: top-decile year-2 cost ---
    y2_total = y2.groupby("member_id")["paid_amount"].sum().rename("y2_total_paid")
    feats = feats.join(y2_total).fillna({"y2_total_paid": 0.0})
    threshold = feats["y2_total_paid"].quantile(0.90)
    feats["target_high_cost_y2"] = (feats["y2_total_paid"] >= threshold).astype(int)

    return feats.drop(columns=["sex", "zip3"])


FEATURE_EXCLUDE = {"y2_total_paid", "target_high_cost_y2"}


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in FEATURE_EXCLUDE]
