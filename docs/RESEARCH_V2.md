# Research V2 — Robustness and Validation

This extension strengthens the original study by testing whether its conclusions survive changes in assumptions, samples, state definitions and learning randomness.

## Research questions

- **RQ1 — Out-of-sample validity:** does the Health Index and the resulting maintenance policy generalize to engines not used to fit preprocessing/PCA?
- **RQ2 — Transition uncertainty:** does the optimal policy remain stable when the empirical degradation transition matrix is re-estimated from different engine samples?
- **RQ3 — Reward sensitivity:** how do maintenance cost, failure cost, reduced-load reward and reduced-load effectiveness change the optimal policy?
- **RQ4 — Pipeline dependence:** which preprocessing choices materially affect the Health Index and policy?
- **RQ5 — State-definition sensitivity:** does the result depend on the chosen Health Index thresholds?
- **RQ6 — RL stability:** does tabular Q-Learning converge to the model-based Value Iteration policy across random seeds?
- **RQ7 — Objective trade-off:** how do return, failures, maintenance frequency and availability move together under alternative policies/scenarios?

## Five-fold engine-level held-out validation

Engines are split, not individual rows. Sensor selection, operating-regime normalization and PCA are fitted only on training engines and then applied to held-out engines.

| Dataset | Held-out Spearman | Mean VI return | Mean baseline return | Difference |
|---|---:|---:|---:|---:|
| FD001 | 0.776 | 198.601 | 196.013 | +2.588 |
| FD002 | 0.783 | 197.537 | 193.520 | +4.017 |
| FD003 | 0.841 | 198.294 | 197.156 | +1.138 |
| FD004 | 0.778 | 196.807 | 196.078 | +0.729 |

The training-derived policy had a higher discounted return than the early-maintenance baseline in every held-out fold. The smallest fold-level advantage was approximately +0.003 in FD004, so the margin can become negligible for some samples.

## Bootstrap uncertainty of empirical transitions

Whole engines were resampled with replacement **1,000 times per subset**. The Health Index/state construction was held fixed so this experiment isolates sampling uncertainty in the empirical transition model.

| Dataset | Base-policy stability | Mean bootstrap return | SD return | Mean P(critical→failure) |
|---|---:|---:|---:|---:|
| FD001 | 100% | 198.569 | 0.111 | 0.0520 |
| FD002 | 100% | 197.520 | 0.096 | 0.0566 |
| FD003 | 100% | 198.249 | 0.278 | 0.0503 |
| FD004 | 100% | 197.098 | 0.287 | 0.0489 |

The Value Iteration policy was unchanged in all **4,000 transition bootstraps**. This supports stability to engine-sampling uncertainty, conditional on the selected Health Index/state construction.

## Reward and action-effect sensitivity

A factorial grid evaluated **225 configurations per subset (900 total)**, varying preventive-maintenance cost, failure cost, reduced-load reward and reduced-load effectiveness.

| Dataset | Base policy | Earlier maintenance | Always operate | Other / reduced-load |
|---|---:|---:|---:|---:|
| FD001 | 215 | 0 | 8 | 2 |
| FD002 | 215 | 0 | 8 | 2 |
| FD003 | 179 | 36 | 8 | 2 |
| FD004 | 151 | 63 | 8 | 3 |

The base policy dominates a broad parameter region, but is not universal. FD003 and FD004 switch more often to earlier maintenance when failure is expensive relative to preventive maintenance. Reduced load is optimal only in a small region of the tested space.

## Ablation study

Four variants were compared: full pipeline, no operating-regime normalization, all nonconstant sensors instead of top-10 selection, and no rolling smoothing.

Key findings:

- Removing regime normalization has almost no effect on FD001/FD003 because those subsets use one operating condition.
- In FD002, removing regime normalization drops HI–RUL Spearman from **0.782 to 0.045**; in FD004, from **0.774 to 0.085**. The policy also degenerates to always operating.
- Removing smoothing preserves relatively high HI–RUL correlation but substantially reduces MDP return, showing that local sensor noise affects estimated state transitions.
- Using all nonconstant sensors does not consistently improve the downstream decision problem; top-sensor selection is therefore justified.

## State-threshold sensitivity

Five threshold schemes were tested. The base policy remained unchanged in **19 of 20** dataset/threshold combinations. The exception was FD003 with thresholds `(0.2, 0.4, 0.6)`, where preventive maintenance moved one state earlier.

Thus, the qualitative result is robust to reasonable threshold changes, while FD003 demonstrates that discretization remains a real modeling assumption.

## Alternative Health Index

PCA was compared with a simpler sign-aligned mean of selected standardized sensors. Both representations produced the **same Value Iteration policy in all four subsets**, and neither method uniformly dominated the other in correlation/return.

This indicates that the main policy result is not uniquely dependent on PCA.

## Q-Learning random-seed stability

Q-Learning was rerun with **20 random seeds per subset** using 2,000 training episodes per seed for the robustness experiment.

| Dataset | Q-Learning matches VI |
|---|---:|
| FD001 | 20/20 (100%) |
| FD002 | 20/20 (100%) |
| FD003 | 17/20 (85%) |
| FD004 | 15/20 (75%) |

Mismatches in FD003/FD004 selected earlier maintenance in the severe state. Therefore the original statement that Q-Learning converges to the same policy should be qualified: it is highly stable in FD001/FD002 and more sensitive to seed/training budget in FD003/FD004.

## Additional baselines and long-run trade-offs

Three interpretable rule policies are considered: early maintenance, late maintenance and a reduced-load heuristic. Under the base utility assumptions, Value Iteration equals the late-maintenance heuristic.

This is an important interpretive result: the contribution is not that VI discovers a mysterious policy, but that the optimization framework provides a principled way to recover and change the policy as costs and dynamics change.

FD003/FD004 make the safety trade-off clear. Earlier maintenance can have lower discounted utility under the chosen objective while strongly suppressing failures. Return and reliability metrics must therefore be reported together.

## Stress scenarios

High failure cost moves FD003 and FD004 to earlier preventive maintenance. Cheap maintenance also moves FD004 to earlier maintenance. FD001/FD002 remain at the base policy across the tested scenarios.

## Transition denoising assumption

The original model redirects apparent backward Health Index transitions to the same state, treating them as measurement/representation noise. The V2 explicitly compares this with raw observed state transitions.

The optimal policy remains unchanged in all four subsets under both treatments, although returns change slightly.

## Supported conclusion

> Under the stated Health Index construction and maintenance-environment assumptions, operating through the severe state and performing preventive maintenance in the critical state is stable to engine resampling, most reasonable state discretizations, an alternative Health Index construction and a broad region of utility parameters. Reliability implications, however, depend strongly on the relative cost assigned to failure, particularly for FD003 and FD004.

These experiments do **not** establish an optimal policy for a real industrial production system. C-MAPSS contains simulated turbofan degradation trajectories and no historical maintenance actions or industrial cost observations. Real deployment requires field calibration and external validation.
