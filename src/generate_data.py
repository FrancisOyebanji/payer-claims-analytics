"""Synthetic payer data generator.

Creates a realistic-shaped (but fully synthetic) payer dataset:
  - members.csv: demographics, plan type, chronic condition flags
  - claims.csv:  claim lines across two plan years with service categories
  - sdoh.csv:    zip-level social determinants of health indices

No real member data is used or represented anywhere in this project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

CONDITIONS = ["diabetes", "chf", "copd", "ckd", "depression", "hypertension"]
CONDITION_PREV = [0.12, 0.04, 0.06, 0.05, 0.15, 0.30]
# Multiplicative effect of each condition on expected annual cost
CONDITION_COST_MULT = {
    "diabetes": 1.9, "chf": 3.2, "copd": 2.4, "ckd": 2.8,
    "depression": 1.5, "hypertension": 1.3,
}
SERVICE_CATEGORIES = ["inpatient", "outpatient", "emergency", "pharmacy", "primary_care", "specialist"]
PLAN_TYPES = ["HMO", "PPO", "Medicaid MCO", "Medicare Advantage"]


def make_members(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = rng.integers(0, 90, n)
    members = pd.DataFrame({
        "member_id": [f"M{i:06d}" for i in range(n)],
        "age": age,
        "sex": rng.choice(["F", "M"], n),
        "plan_type": rng.choice(PLAN_TYPES, n, p=[0.30, 0.35, 0.20, 0.15]),
        "zip3": rng.choice([f"z{z:03d}" for z in range(50)], n),
        "months_enrolled_y1": np.clip(rng.normal(11.2, 2.0, n).round(), 1, 12).astype(int),
    })
    # Chronic conditions correlate with age
    age_factor = np.clip((age - 20) / 70, 0, 1)
    for cond, prev in zip(CONDITIONS, CONDITION_PREV):
        p = np.clip(prev * (0.4 + 1.6 * age_factor), 0, 0.85)
        members[f"cond_{cond}"] = rng.binomial(1, p)
    return members


def make_sdoh(rng: np.random.Generator) -> pd.DataFrame:
    zips = [f"z{z:03d}" for z in range(50)]
    deprivation = rng.beta(2, 3, len(zips))  # area deprivation index proxy, 0-1
    return pd.DataFrame({
        "zip3": zips,
        "adi_proxy": deprivation.round(3),
        "food_insecurity_rate": np.clip(deprivation * rng.normal(0.9, 0.15, len(zips)), 0.02, 0.6).round(3),
        "transport_barrier_rate": np.clip(deprivation * rng.normal(0.6, 0.2, len(zips)), 0.01, 0.5).round(3),
        "broadband_access_rate": np.clip(1 - deprivation * rng.normal(0.7, 0.15, len(zips)), 0.3, 0.99).round(3),
    })


def expected_annual_cost(members: pd.DataFrame, sdoh: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Latent expected cost per member-year (drives both y1 and y2 claim draws)."""
    base = 1200 + 45 * np.maximum(members["age"] - 40, 0)
    mult = np.ones(len(members))
    for cond, m in CONDITION_COST_MULT.items():
        mult *= np.where(members[f"cond_{cond}"] == 1, m, 1.0)
    adi = members.merge(sdoh[["zip3", "adi_proxy"]], on="zip3", how="left")["adi_proxy"].to_numpy()
    mult *= 1 + 0.6 * adi  # deprivation raises expected cost
    frailty = rng.lognormal(0, 0.7, len(members))  # unobserved heterogeneity
    return base * mult * frailty


def make_claims(members: pd.DataFrame, exp_cost: np.ndarray, year: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw claim lines whose totals approximate each member's expected cost."""
    rows = []
    cat_p = np.array([0.06, 0.30, 0.08, 0.28, 0.16, 0.12])
    cat_mean = {"inpatient": 9500, "outpatient": 650, "emergency": 1400,
                "pharmacy": 210, "primary_care": 160, "specialist": 340}
    for i, m in enumerate(members.itertuples(index=False)):
        # Year-to-year drift in utilization
        drift = rng.lognormal(0, 0.35)
        target = exp_cost[i] * drift * (m.months_enrolled_y1 / 12 if year == 1 else 1.0)
        n_claims = max(1, int(rng.poisson(np.clip(target / 900, 0.5, 60))))
        cats = rng.choice(SERVICE_CATEGORIES, n_claims, p=cat_p)
        for c in cats:
            amt = rng.lognormal(np.log(cat_mean[c]), 0.8)
            rows.append((m.member_id, year, c, round(float(amt), 2),
                         int(rng.integers(1, 366))))
    return pd.DataFrame(rows, columns=["member_id", "plan_year", "service_category", "paid_amount", "service_day"])


def main(out_dir: str, n_members: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    members = make_members(n_members, rng)
    sdoh = make_sdoh(rng)
    exp_cost = expected_annual_cost(members, sdoh, rng)

    claims_y1 = make_claims(members, exp_cost, 1, rng)
    claims_y2 = make_claims(members, exp_cost, 2, rng)
    claims = pd.concat([claims_y1, claims_y2], ignore_index=True)

    members.to_csv(out / "members.csv", index=False)
    sdoh.to_csv(out / "sdoh.csv", index=False)
    claims.to_csv(out / "claims.csv", index=False)
    print(f"Wrote {len(members):,} members, {len(claims):,} claim lines to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--members", type=int, default=8000)
    args = ap.parse_args()
    main(args.out, args.members)
