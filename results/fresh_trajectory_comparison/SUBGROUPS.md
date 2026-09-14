# P1 versus B3: all facility and geometry subgroups

Fresh-trajectory 18-epoch comparison, five model seeds. Differences are P1 minus B3.
Positive recall/F1 and negative FPR differences are favorable. No subgroups are selected for model tuning.

## Aggregate and facility groups

| Scope | Facility | Recall difference | F1 difference | FPR difference |
|---|---|---|---|---|
| all | __all__ | +0.0359 | +0.0415 | -0.0036 |
| seen | __all__ | +0.0583 | +0.0545 | +0.0021 |
| seen | harsh | +0.0833 | +0.0654 | +0.0042 |
| seen | standard | +0.0333 | +0.0311 | +0.0000 |
| unseen | __all__ | +0.0225 | +0.0312 | -0.0071 |
| unseen | harsh | +0.0567 | +0.0591 | +0.0042 |
| unseen | standard | -0.0117 | -0.0055 | -0.0183 |

## All facility × geometry groups

Wins count strict within-seed improvements; ties are not wins. Five seeds share trajectories.

| Scope | Facility | Geometry | Recall difference | F1 difference | FPR difference | Recall wins / 5 | F1 wins / 5 | FPR reductions / 5 |
|---|---|---|---|---|---|---|---|---|
| seen | harsh | central_furniture | +0.0583 | +0.0440 | +0.0042 | 4 | 4 | 0 |
| seen | harsh | clear | +0.0583 | +0.0335 | +0.0042 | 1 | 1 | 0 |
| seen | harsh | vertical_mid | +0.1333 | +0.1428 | +0.0042 | 4 | 4 | 0 |
| seen | standard | central_furniture | +0.0417 | +0.0338 | +0.0083 | 3 | 4 | 1 |
| seen | standard | clear | +0.0500 | +0.0444 | +0.0125 | 3 | 3 | 0 |
| seen | standard | vertical_mid | +0.0083 | +0.0180 | -0.0208 | 1 | 2 | 3 |
| unseen | harsh | horizontal_new | +0.0833 | +0.0996 | +0.0042 | 3 | 3 | 0 |
| unseen | harsh | low_new | +0.0833 | +0.0720 | +0.0083 | 4 | 4 | 0 |
| unseen | harsh | two_new | +0.0500 | +0.0509 | +0.0042 | 3 | 3 | 0 |
| unseen | harsh | vertical_new | +0.0333 | +0.0435 | +0.0000 | 2 | 2 | 0 |
| unseen | harsh | wide_new | +0.0333 | +0.0285 | +0.0042 | 3 | 3 | 0 |
| unseen | standard | horizontal_new | -0.0500 | -0.0484 | -0.0083 | 0 | 0 | 2 |
| unseen | standard | low_new | +0.0167 | +0.0230 | -0.0083 | 2 | 3 | 2 |
| unseen | standard | two_new | +0.0083 | +0.0073 | -0.0042 | 1 | 3 | 2 |
| unseen | standard | vertical_new | -0.0083 | +0.0185 | -0.0500 | 2 | 3 | 5 |
| unseen | standard | wide_new | -0.0250 | -0.0194 | -0.0208 | 0 | 1 | 3 |

## Mean subgroup directions

| Scope | Subgroups | Recall positive / zero / negative | F1 positive / zero / negative | FPR lower / equal / higher |
|---|---|---|---|---|
| seen | 6 | 6 / 0 / 0 | 6 / 0 / 0 | 1 / 0 / 5 |
| unseen | 10 | 7 / 0 / 3 | 8 / 0 / 2 | 5 / 1 / 4 |

Group-level counts are descriptive, not independent statistical trials. Positive aggregate differences do not imply uniform benefits.
The common calibration FPR budget is 0.05; it is not a matched test-FPR comparison.
See the original grouped CSVs for policy values, sample standard deviations and all seeds.
