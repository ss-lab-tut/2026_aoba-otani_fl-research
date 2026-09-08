# Tables generated from the verified comparison

Mean ± sample standard deviation over five model seeds sharing fixed trajectories.
Main calibration FPR budget: 0.05. These are simulated endpoint metrics.

## all

| Method | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.111 ± 0.070 | 0.646 ± 0.054 | 0.183 ± 0.096 | 0.647 ± 0.131 | 0.029 ± 0.014 |
| B2 | 0.078 ± 0.020 | 0.553 ± 0.104 | 0.135 ± 0.032 | 0.588 ± 0.063 | 0.032 ± 0.011 |
| B3 | 0.072 ± 0.019 | 0.582 ± 0.065 | 0.128 ± 0.031 | 0.588 ± 0.063 | 0.026 ± 0.006 |
| P1 | 0.070 ± 0.016 | 0.620 ± 0.056 | 0.125 ± 0.026 | 0.588 ± 0.063 | 0.022 ± 0.007 |

## seen

| Method | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.150 ± 0.097 | 0.595 ± 0.089 | 0.233 ± 0.124 | 0.663 ± 0.134 | 0.045 ± 0.012 |
| B2 | 0.108 ± 0.028 | 0.527 ± 0.116 | 0.179 ± 0.043 | 0.599 ± 0.072 | 0.049 ± 0.014 |
| B3 | 0.101 ± 0.024 | 0.533 ± 0.081 | 0.170 ± 0.036 | 0.599 ± 0.072 | 0.044 ± 0.011 |
| P1 | 0.096 ± 0.022 | 0.546 ± 0.078 | 0.162 ± 0.032 | 0.599 ± 0.072 | 0.040 ± 0.012 |

## unseen

| Method | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.087 ± 0.054 | 0.711 ± 0.091 | 0.151 ± 0.078 | 0.638 ± 0.129 | 0.020 ± 0.016 |
| B2 | 0.059 ± 0.017 | 0.607 ± 0.124 | 0.107 ± 0.027 | 0.582 ± 0.058 | 0.022 ± 0.012 |
| B3 | 0.055 ± 0.019 | 0.665 ± 0.075 | 0.101 ± 0.032 | 0.582 ± 0.058 | 0.015 ± 0.007 |
| P1 | 0.054 ± 0.014 | 0.748 ± 0.099 | 0.100 ± 0.024 | 0.582 ± 0.058 | 0.010 ± 0.006 |

## Paired P1 minus B3

Differences are computed within each seed before averaging. Negative FPR differences mean fewer false positives.

| Scope | Recall difference | Precision difference | F1 difference | AUROC difference | FPR difference |
|---|---|---|---|---|---|
| all | -0.0026 ± 0.0049 | +0.0386 ± 0.0348 | -0.0033 ± 0.0080 | +0.0000 ± 0.0000 | -0.0042 ± 0.0025 |
| seen | -0.0056 ± 0.0076 | +0.0130 ± 0.0368 | -0.0074 ± 0.0118 | +0.0000 ± 0.0000 | -0.0042 ± 0.0057 |
| unseen | -0.0008 ± 0.0090 | +0.0835 ± 0.0626 | -0.0004 ± 0.0152 | +0.0000 ± 0.0000 | -0.0042 ± 0.0015 |

## Source SHA-256

- `summary.csv`: `2a22b7c8230203f67831fe6c76d4dd9c6fd03e43853f4d5b3aa25e3fd8ddca82`
- `paired_P1_minus_B3_summary.csv`: `ea62c030161542ad086c95554300fd6fc6193d27f2a27ce3678c75ec924a2675`
