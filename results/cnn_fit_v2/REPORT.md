# Frozen CNN fit diagnosis

No retraining. Operating metrics use the original calibration-only B1/B2 thresholds (budget 0.05).
Train rows are the actual windows used by each model. Seeds share trajectories; these are descriptive diagnostics.

| Method | Scope | AUROC mean ± SD | Recall mean ± SD | FPR mean ± SD |
|---|---|---|---|---|
| B1 | calibration_all | 0.614 ± 0.152 | 0.135 ± 0.081 | 0.049 ± 0.000 |
| B1 | calibration_clear | 0.625 ± 0.160 | 0.233 ± 0.148 | 0.098 ± 0.017 |
| B1 | calibration_seen | 0.614 ± 0.152 | 0.135 ± 0.081 | 0.049 ± 0.000 |
| B1 | test_all | 0.647 ± 0.131 | 0.111 ± 0.070 | 0.029 ± 0.014 |
| B1 | test_clear | 0.671 ± 0.137 | 0.258 ± 0.143 | 0.094 ± 0.024 |
| B1 | test_seen | 0.663 ± 0.134 | 0.150 ± 0.097 | 0.045 ± 0.012 |
| B1 | test_unseen | 0.638 ± 0.129 | 0.087 ± 0.054 | 0.020 ± 0.016 |
| B1 | train_actual | 0.815 ± 0.110 | 0.327 ± 0.185 | 0.021 ± 0.015 |
| B2 | calibration_all | 0.527 ± 0.055 | 0.081 ± 0.027 | 0.049 ± 0.000 |
| B2 | calibration_clear | 0.537 ± 0.071 | 0.129 ± 0.050 | 0.085 ± 0.020 |
| B2 | calibration_seen | 0.527 ± 0.055 | 0.081 ± 0.027 | 0.049 ± 0.000 |
| B2 | test_all | 0.588 ± 0.063 | 0.078 ± 0.020 | 0.032 ± 0.011 |
| B2 | test_clear | 0.610 ± 0.083 | 0.196 ± 0.056 | 0.100 ± 0.037 |
| B2 | test_seen | 0.599 ± 0.072 | 0.108 ± 0.028 | 0.049 ± 0.014 |
| B2 | test_unseen | 0.582 ± 0.058 | 0.059 ± 0.017 | 0.022 ± 0.012 |
| B2 | train_actual | 0.781 ± 0.086 | 0.152 ± 0.071 | 0.004 ± 0.004 |

Train/calibration/test mixtures differ: compare clear test with B1 training, and known-geometry test with B2 training cautiously.
Training performance is descriptive and is not an estimate of generalization. A low training AUROC alone does not identify an optimization or representation cause.
Post-hoc diagnostics do not change the primary comparison or select an improved model.

## Verified input hashes

- `training.json`: `2234f902c9cd592fe83b1152afd63f7236167ce98ebe773fe6a170fb52de9293`
- `windows.csv`: `b6cf6901cf16467822347d12df2a121176581793c134b14bc1666915e174a182`
- `scores.csv`: `cc61637a456403cbc6d66c57101757e46440085376831761a89ab8a7dcb2c7cd`
- `observations.npz`: `4f73c7b84c3d80c7d75d084233405d005b2ad6650a3be0043ffb80d547bf45e7`
- Diagnostic script: `2af8090c32c3d5e2e1b85a80b6168a33bf8641afee5a3700320fa7fcc64e06aa`
