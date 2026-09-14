# Frozen-score replay verification

Recomputed from the compressed scores and protocol saved in Git. No local observations or CNN checkpoints are read.

| Epochs | Metric rows | Summary rows | Paired rows | Paired summary rows |
|---|---|---|---|---|
| 6 | 620 | 124 | 155 | 31 |
| 18 | 620 | 124 | 155 | 31 |

All confusion counts, operating metrics, seed summaries and P1-minus-B3 differences match the published tables (numeric tolerance 1e-12).
The input score hashes and all four exported-table hashes were checked against each audit record.
This validates score-to-table reproduction; it does not retrain models or independently validate the simulator.

Reproduce with `python experiments/verify_fresh_score_replay.py` (NumPy required).

Script SHA-256: `1e4e39684a93f7f3063e8438c4f767dca398680af664f17f50d4adc126c37bde`
