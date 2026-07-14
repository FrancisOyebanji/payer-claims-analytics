# Insights Report — High-Cost Member Prediction & Quality Analytics

*Generated from the pipeline in this repo (synthetic data, seed=42). Numbers below reproduce exactly with `python src/run_pipeline.py`.*

## Executive summary

Using only year-1 claims utilization, demographics, chronic-condition flags, and zip-level SDOH indices, the model identifies 66% of next year's top-decile-cost members while targeting just 20% of the population — a 3.3x improvement over untargeted outreach. At a tighter 5% targeting budget, precision is 6.1x the base rate. For a payer running care-management programs where each engaged high-risk member avoids even a fraction of expected cost, this ranking directly translates into program ROI.

## Model performance (held-out test set, n=2,000)

| Metric | Logistic baseline | Gradient boosting | MARS |
|---|---|---|---|
| ROC-AUC | 0.834 | 0.829 | **0.834** |
| Average precision | 0.516 | 0.492 | **0.525** |
| Lift @ top 5% | — | 6.1x | **6.7x** |
| Capture @ top 20% | — | 66.0% | **66.5%** |

**An honest finding worth leading with:** the two simpler models beat gradient boosting. Prior-year utilization is such a dominant, near-linear signal here that black-box complexity buys nothing. The recommended model is MARS (multivariate adaptive regression splines, implemented from scratch in `src/mars.py`): it matches the logistic baseline on AUC, wins on precision and lift, and its entire decision logic is 12 human-readable hinge functions — e.g., risk inflects above ~$1,350 in year-1 pharmacy spend, past age 44, and at the first chronic condition. Those knots aren't just model internals; they're presentation-ready thresholds a Chief Actuary can interrogate. Chasing decimal points of AUC with unexplainable models is how analytics teams lose that room.

## What drives risk

Feature importances (see `figures/feature_importance.png`) are dominated by year-1 claim count (0.48), followed by pharmacy spend, specialist spend, and age. Chronic-condition count and SDOH deprivation appear with smaller but non-zero weight — consistent with SDOH acting through utilization rather than alongside it. Implication for product design: SDOH data earns its integration cost primarily for members with *thin claims history* (new enrollees), where utilization signal doesn't exist yet. That's a segmentation insight, not a modeling insight.

## Quality measure snapshot (plan year 2)

| Measure (HEDIS-inspired proxy) | Eligible | Rate |
|---|---|---|
| Diabetes care engagement | 946 | 61.4% |
| Follow-up after inpatient stay | 1,989 | 77.2% |
| Members not ED-reliant | 4,654 | 75.5% |
| Behavioral health engagement | 1,191 | 75.1% |
| Chronic pharmacy engagement | 4,113 | **22.6%** |

The chronic pharmacy engagement rate is the outlier and the opportunity: fewer than a quarter of chronic-condition members show a consistent pharmacy fill pattern. For a Medicare Advantage plan, medication-adherence measures are triple-weighted in CMS Star Ratings — this is where quality dollars and cost-of-care dollars point at the same intervention.

## Recommended next steps (client framing)

1. **Pilot targeting:** route the model's top-5% list to care management; measure engaged-vs-unengaged cost trajectory over two quarters.
2. **Close the adherence gap:** cross the low pharmacy-engagement cohort with the high-risk list; pharmacy outreach on the intersection is the highest-leverage single program.
3. **New-enrollee cold start:** stand up the SDOH-weighted model variant for members with <6 months of claims history.

## Limitations

Synthetic data cannot capture real coding behavior, benefit-design effects, or care-seeking heterogeneity; performance numbers demonstrate the *method*, not a production benchmark. Measure implementations follow the HEDIS eligible-population/numerator pattern but are not certified NCQA logic.
