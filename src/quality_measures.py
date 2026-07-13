"""Simplified, HEDIS-inspired quality measures computed from claims.

These are *illustrative* implementations of the measure pattern
(eligible population -> numerator -> rate), not certified NCQA HEDIS logic.
Real HEDIS requires licensed value sets, continuous-enrollment rules,
and certified measure engines.
"""
from __future__ import annotations

import pandas as pd

CONDITIONS = ["diabetes", "chf", "copd", "ckd", "depression", "hypertension"]


def _rate(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else float("nan")


def compute_measures(members: pd.DataFrame, claims: pd.DataFrame, year: int = 2) -> pd.DataFrame:
    """Compute plan-level, HEDIS-style measure rates for a plan year."""
    yr = claims[claims.plan_year == year]
    visits = yr.groupby(["member_id", "service_category"]).size().unstack(fill_value=0)
    m = members.set_index("member_id").join(visits).fillna(0)

    rows = []

    # 1. Diabetes care engagement (proxy for HbA1c testing / eye exam measures):
    #    diabetic members with >= 2 primary care or specialist visits in the year.
    diab = m[m.cond_diabetes == 1]
    num = ((diab.get("primary_care", 0) + diab.get("specialist", 0)) >= 2).sum()
    rows.append(("Diabetes care engagement (proxy CDC measure)", len(diab), int(num), _rate(int(num), len(diab))))

    # 2. Follow-up after inpatient stay: members with an inpatient claim who also
    #    have a primary-care or specialist claim in the same year (FUH-style pattern).
    ip = m[m.get("inpatient", 0) >= 1]
    num = ((ip.get("primary_care", 0) + ip.get("specialist", 0)) >= 1).sum()
    rows.append(("Follow-up after inpatient stay (FUH-style)", len(ip), int(num), _rate(int(num), len(ip))))

    # 3. Avoidable ED reliance: share of members whose ED visits exceed primary-care
    #    visits (lower is better; reported as % NOT ED-reliant to keep direction consistent).
    active = m[(m.get("emergency", 0) + m.get("primary_care", 0)) > 0]
    reliant = (active.get("emergency", 0) > active.get("primary_care", 0)).sum()
    rows.append(("Members not ED-reliant (EDU-style, inverse)", len(active),
                 int(len(active) - reliant), _rate(int(len(active) - reliant), len(active))))

    # 4. Behavioral health engagement: members with depression flag who have any
    #    primary-care/specialist contact (AMM-style engagement proxy).
    dep = m[m.cond_depression == 1]
    num = ((dep.get("primary_care", 0) + dep.get("specialist", 0)) >= 1).sum()
    rows.append(("Behavioral health engagement (AMM-style proxy)", len(dep), int(num), _rate(int(num), len(dep))))

    # 5. Medication engagement for chronic members: chronic-condition members with
    #    >= 4 pharmacy claims in the year (adherence-pattern proxy, PDC-inspired).
    chronic = m[m[[f"cond_{c}" for c in CONDITIONS]].sum(axis=1) >= 1]
    num = (chronic.get("pharmacy", 0) >= 4).sum()
    rows.append(("Chronic pharmacy engagement (PDC-inspired proxy)", len(chronic), int(num), _rate(int(num), len(chronic))))

    return pd.DataFrame(rows, columns=["measure", "eligible_population", "numerator", "rate_pct"])


def measures_by_plan(members: pd.DataFrame, claims: pd.DataFrame, year: int = 2) -> pd.DataFrame:
    """Measure rates segmented by plan type — the cut payer strategy teams ask for first."""
    frames = []
    for plan in sorted(members.plan_type.unique()):
        sub = members[members.plan_type == plan]
        df = compute_measures(sub, claims[claims.member_id.isin(sub.member_id)], year)
        df.insert(0, "plan_type", plan)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
