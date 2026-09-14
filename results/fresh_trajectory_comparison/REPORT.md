# Fresh-trajectory comparison: fixed six and eighteen epochs

Additional analysis; original main results are preserved. New calibration seeds 13001/13002, test seeds 14001/14002.
Same training inputs, initializations and first-six-epoch losses verified. Same new evaluation windows and frozen condition estimator.
Mean ± sample SD across five model seeds sharing trajectories. Calibration FPR budget 0.05.

| Epochs | Scope | Policy | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|---|---|
| 6 | all | B1 | 0.148 ± 0.162 | 0.699 ± 0.186 | 0.225 ± 0.213 | 0.724 ± 0.127 | 0.018 ± 0.005 |
| 6 | unseen | B1 | 0.123 ± 0.153 | 0.710 ± 0.180 | 0.190 ± 0.207 | 0.721 ± 0.125 | 0.014 ± 0.006 |
| 6 | all | B2 | 0.071 ± 0.022 | 0.624 ± 0.065 | 0.127 ± 0.032 | 0.671 ± 0.065 | 0.023 ± 0.011 |
| 6 | unseen | B2 | 0.056 ± 0.021 | 0.610 ± 0.067 | 0.101 ± 0.033 | 0.669 ± 0.060 | 0.019 ± 0.011 |
| 6 | all | B3 | 0.073 ± 0.026 | 0.680 ± 0.070 | 0.130 ± 0.041 | 0.671 ± 0.065 | 0.018 ± 0.012 |
| 6 | unseen | B3 | 0.055 ± 0.022 | 0.673 ± 0.111 | 0.100 ± 0.035 | 0.669 ± 0.060 | 0.015 ± 0.013 |
| 6 | all | P1 | 0.067 ± 0.025 | 0.673 ± 0.063 | 0.120 ± 0.037 | 0.671 ± 0.065 | 0.018 ± 0.013 |
| 6 | unseen | P1 | 0.051 ± 0.023 | 0.658 ± 0.079 | 0.093 ± 0.037 | 0.669 ± 0.060 | 0.015 ± 0.014 |
| 18 | all | B1 | 0.383 ± 0.067 | 0.884 ± 0.038 | 0.531 ± 0.064 | 0.883 ± 0.003 | 0.026 ± 0.011 |
| 18 | unseen | B1 | 0.337 ± 0.065 | 0.869 ± 0.053 | 0.482 ± 0.066 | 0.869 ± 0.004 | 0.026 ± 0.012 |
| 18 | all | B2 | 0.341 ± 0.056 | 0.938 ± 0.030 | 0.498 ± 0.056 | 0.877 ± 0.020 | 0.012 ± 0.007 |
| 18 | unseen | B2 | 0.301 ± 0.074 | 0.926 ± 0.031 | 0.449 ± 0.080 | 0.866 ± 0.022 | 0.013 ± 0.008 |
| 18 | all | B3 | 0.334 ± 0.073 | 0.822 ± 0.047 | 0.471 ± 0.077 | 0.877 ± 0.020 | 0.037 ± 0.016 |
| 18 | unseen | B3 | 0.301 ± 0.084 | 0.798 ± 0.054 | 0.432 ± 0.096 | 0.866 ± 0.022 | 0.037 ± 0.014 |
| 18 | all | P1 | 0.370 ± 0.067 | 0.852 ± 0.048 | 0.512 ± 0.063 | 0.877 ± 0.020 | 0.033 ± 0.017 |
| 18 | unseen | P1 | 0.323 ± 0.076 | 0.843 ± 0.053 | 0.463 ± 0.082 | 0.866 ± 0.022 | 0.030 ± 0.016 |

## P1 minus B3 within each epoch setting

| Epochs | Scope | Recall difference | F1 difference | FPR difference |
|---|---|---|---|---|
| 6 | all | -0.006 ± 0.013 | -0.011 ± 0.022 | -0.000 ± 0.002 |
| 6 | unseen | -0.004 ± 0.015 | -0.008 ± 0.026 | 0.000 ± 0.002 |
| 18 | all | 0.036 ± 0.034 | 0.041 ± 0.038 | -0.004 ± 0.004 |
| 18 | unseen | 0.022 ± 0.025 | 0.031 ± 0.030 | -0.007 ± 0.004 |

## Epoch 18 minus epoch 6 on the same new data

| Scope | Policy | Recall difference | F1 difference | FPR difference |
|---|---|---|---|---|
| all | B1 | 0.235 ± 0.172 | 0.306 ± 0.215 | 0.008 ± 0.014 |
| unseen | B1 | 0.214 ± 0.159 | 0.292 ± 0.205 | 0.013 ± 0.016 |
| all | B2 | 0.270 ± 0.055 | 0.371 ± 0.058 | -0.011 ± 0.009 |
| unseen | B2 | 0.245 ± 0.072 | 0.348 ± 0.078 | -0.006 ± 0.011 |
| all | B3 | 0.261 ± 0.072 | 0.341 ± 0.077 | 0.018 ± 0.021 |
| unseen | B3 | 0.246 ± 0.074 | 0.332 ± 0.078 | 0.023 ± 0.019 |
| all | P1 | 0.303 ± 0.071 | 0.393 ± 0.070 | 0.015 ± 0.024 |
| unseen | P1 | 0.272 ± 0.075 | 0.371 ± 0.081 | 0.015 ± 0.022 |

These are synthetic endpoint metrics, not measured-data or continuous-monitoring results.
Equal calibration FPR budgets do not imply equal test FPR. All four policies and both durations are retained.
No best seed, epoch or operating point was selected using these test results.

Verification: 20 checkpoints, 1,240 grouped confusion-count rows, 16 calibration inferences per checkpoint, identical training/evaluation data and matching six-epoch loss prefixes.
