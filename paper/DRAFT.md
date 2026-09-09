# Facility-Specific and Condition-Adaptive Thresholds for Simulated Fall Detection: A Five-Seed Comparison

Working manuscript, 2026-09-08. Venue-independent draft for internal review. Related work, verified references, author details and venue formatting remain to be completed. Numerical tables are generated in [TABLES.md](TABLES.md). This draft reports a controlled simulation study, not measured-sensor or federated-learning validation.

## Abstract

We evaluate whether adapting a fall-detection threshold to an estimated observation condition improves performance relative to using a facility-specific threshold. A fixed condition estimator is connected to an existing convolutional neural network (CNN), and four policies are compared using the same data partitions and evaluation endpoints: a clear-input baseline, a geometry-augmented baseline, facility-specific thresholds, and condition-adaptive thresholds. The last three policies share the same fall scores. Thresholds are calibrated using only a held-out calibration partition with an empirical false-positive-rate budget of 0.05. Experiments use two simulated facilities, three known geometries, five held-out geometries and five model seeds. On held-out geometries, the condition-adaptive policy attains mean recall 0.054, F1 0.100 and false-positive rate 0.010, compared with 0.055, 0.101 and 0.015 for facility-specific thresholds. Thus, the adaptive policy reduces false positives in this setting without improving mean recall or F1. Low absolute recall, controlled event sampling and the absence of measured data limit the conclusions. The study provides a reproducible downstream comparison and identifies the limits of threshold adaptation with the evaluated fall scores.

## 1. Introduction

The operating threshold is part of a fall-detection system: the same fall score can produce different detection and false-positive rates depending on how that threshold is assigned. Facility-specific calibration assigns a threshold using a facility identifier. An alternative is to estimate the observation condition from available sensor statistics and select a threshold using that estimate. These alternatives should be compared using the same fall scores to separate the effect of the threshold policy from differences in the detector.

This study asks whether a fixed condition estimator yields a downstream benefit when connected to an existing fall classifier. We focus on a controlled comparison rather than introducing a new classifier or improving the condition estimator. The central comparison is between facility-specific and condition-adaptive thresholds. Two additional baselines distinguish the effects of geometry augmentation and threshold assignment.

Our contributions within this experimental scope are: (1) a paired four-policy evaluation with disjoint recording partitions and held-out geometries; (2) an explicit endpoint-label protocol linking the CNN context and condition-estimation context to the same decision time; and (3) a five-seed analysis of detection metrics, false-positive rates and paired differences. The results show a reduction in false positives for the adaptive policy at the main calibration budget, but do not establish an improvement in mean recall or F1.

## 2. Related work — pending literature verification

This section is not yet written. The final manuscript needs verified primary sources for LiDAR-based fall detection, observation-condition or domain adaptation, and threshold selection under false-positive constraints. No novelty or state-of-the-art claim is made in this draft. The relationship to prior work must be assessed before submission.

## 3. Detection and threshold policies

Let x_t be a causal observation window ending at decision time t. The existing CNN produces a fall score s(x_t), taken as the softmax probability of its fall class. The architecture has three convolution/pooling stages and two fully connected layers; its three-output structure is retained, with non-fall and fall training labels mapped to classes 0 and 2. Each input has two channels, 64 temporal frames and 430 angular bins. The channels encode range and height, normalized by fixed scales of 8 m and 2.5 m.

The frozen condition estimator uses observable shape statistics from the trailing five frames and yields a score relative to its previously fixed condition cutoff. A nonnegative margin selects the degraded state; a negative margin selects the clear state. It is not retrained in these experiments. Its output selects one of two calibrated thresholds; it does not perform continuous threshold regression or online learning.

The policies are defined as follows:

| Policy | Fall score | Threshold assignment |
|---|---|---|
| B1 | CNN trained on clear inputs | One common threshold |
| B2 | CNN trained with known-geometry replacement | One common threshold |
| B3 | Same score as B2 | One threshold per facility |
| P1 | Same score as B2 | One threshold per estimated condition state |

A window is classified as positive when its score is greater than or equal to its assigned threshold. For each calibration group, we sort negative scores in descending order and allow at most floor(alpha times n_negative) false positives. The threshold is the next representable value above the next negative score, which handles tied scores under the inclusive comparison rule. The main budget alpha is 0.05. B1 and B2 use all calibration windows; B3 groups them by facility and P1 by estimated condition. Insufficient calibration groups fall back to the B2 threshold; no such fallback occurred in the completed comparison. Thresholds are never fitted using test labels.

## 4. Experimental protocol

### 4.1 Simulation and endpoint labels

There are no measured data in this study. The simulator produces three-coordinate point clouds with 2.5-D occlusion. Two facility configurations differ in sensor noise. The experiment uses eight training recordings, four calibration recordings and four test recordings, with disjoint generation seeds. Each recording contains 6,000 simulation steps at one second per step.

Positive endpoints are the third frame of a FALL or RECOVERED_FALL episode. NEAR_FALL examples are negative endpoints after the episode has ended and the state has returned to non-abnormal. Other negatives have no ABNORMAL state in their trailing five frames. The CNN receives the 64 frames ending at the endpoint; the condition estimator uses the last five of those frames. Simulator subtype labels determine sampling and ground truth, and are not inference inputs.

An initial connection diagnostic used the presence of any ABNORMAL frame in a window as its target. Subsequent inspection found no strict FALL episodes in its training and calibration selections and frequent disagreement between the window label and endpoint state. Those diagnostic results are excluded from the main efficacy analysis. The corrected simulation increases the ABNORMAL self-transition probability from 0.2 to 0.8 to obtain the intended events, while proportionally reducing the other transitions in that row. Other transition rows are unchanged. The initial and corrected protocols are not used as a direct before/after performance comparison.

The corrected selections contain 96 positive training events, 48 calibration events and 48 test events before geometry replication. Of these, 59, 31 and 26 are strict FALL events; the remainder are RECOVERED_FALL. Sampling fixes positive prevalence at one third and balances the case mix across facilities. Selection uses ground-truth categories and fixed random seeds, not detector scores.

### 4.2 Geometry and training controls

Known geometries are clear, vertical_mid and central_furniture. Held-out geometries are vertical_new, horizontal_new, low_new, wide_new and two_new. These held-out geometries are absent from CNN training, threshold calibration and condition-estimator training. They were examined in earlier condition-estimation investigations, so they are not globally untouched research settings. The current experiment uses new trajectories.

For each model seed from 0 to 4, B1 and B2 start from matching CNN initialization and use equal numbers of training windows, the same minibatch order, six epochs, batch size 16 and Adam learning rate 0.001. B1 uses clear observations. B2 replaces each training window with one sampled known-geometry version without increasing the number of windows. B2, B3 and P1 then use identical checkpoints and evaluation scores. The five model seeds share the same trajectories; they measure model-training variation rather than five independent dataset replications.

### 4.3 Metrics and verification

We report recall, precision, F1, raw-score AUROC, false-positive rate (FPR) and confusion counts by facility, geometry and their intersection, with separate known- and held-out-geometry summaries. Results use the mean and sample standard deviation over model seeds. P1-minus-B3 differences are computed within each seed before aggregation. Raw-score AUROC is identical for B2, B3 and P1 by construction; threshold-policy effects must be assessed using operating metrics. These endpoint metrics do not measure event-level recall or detection latency.

Checks cover ten checkpoint hashes, recording partitions, endpoint labels, array alignment and 620 grouped metric rows. Actual inference on 16 calibration windows per checkpoint reproduces cached scores. The frozen condition-estimator artifact is unchanged. A saved base commit and patch reproduce all 23 recorded simulator runtime source hashes. Full retraining in a separate environment has not been performed.

## 5. Results

Insert the generated all, seen and unseen tables from [TABLES.md](TABLES.md). Preserve all four policies in the main presentation; do not report only the favorable P1 comparisons.

Across all geometries, B1 has the highest mean recall, F1 and AUROC: 0.111, 0.183 and 0.647, respectively. B3 attains recall 0.072, precision 0.582, F1 0.128 and FPR 0.026. P1 attains recall 0.070, precision 0.620, F1 0.125 and FPR 0.022. The mean paired P1-minus-B3 differences are -0.0026 in recall, -0.0033 in F1 and -0.0042 in FPR. Thus, the adaptive threshold does not improve mean recall or F1 in the overall evaluation.

On held-out geometries, B3 and P1 attain mean recalls of 0.055 and 0.054, mean F1 scores of 0.101 and 0.100, and mean FPRs of 0.015 and 0.010. P1 has lower FPR in all five seeds, but recall differences vary across seeds and the mean recall difference is slightly negative. These findings support a descriptive reduction in false positives in the evaluated setting, not statistical significance or overall superiority.

A descriptive sensitivity analysis recalibrates thresholds at budgets 0.01, 0.025, 0.05, 0.10, 0.15 and 0.20 using the same frozen scores. It does not establish consistent dominance across operating points. At budget 0.15, higher mean recall for P1 accompanies higher mean test FPR. Equal calibration budgets do not imply equal realized test FPR. The primary budget remains 0.05; the test results are not used to select a replacement operating point.

## 6. Discussion and limitations

A post-hoc analysis of the frozen checkpoints compares their actual training windows with test subsets (see [the diagnostic figures](figures/README.md)). Mean training AUROC is 0.815 for B1 and 0.781 for B2, versus 0.671 and 0.610 on clear test inputs, 0.663 and 0.599 on known test geometries, and 0.638 and 0.582 on held-out geometries. The drop is therefore already present on known-condition test data; held-out geometry alone cannot account for the observed performance. Training and test mixtures differ, especially for geometry-augmented B2, and these descriptive gaps do not isolate optimization or overfitting as the cause. This analysis uses the same frozen models and does not change the main experiment.

The central observation is that connecting the frozen condition estimator to threshold selection changes the detector's operating behavior but does not establish better recall or F1. A reduction in false positives should be presented together with the missed-detection metrics. The clear-input baseline also exceeds the augmented policies in mean recall, F1 and AUROC, so the current evidence does not support a general benefit from the chosen augmentation protocol.

Absolute recall is low. The current comparison does not determine whether optimization, limited training diversity, the long input context or other aspects of the protocol are the principal cause. These are hypotheses requiring separate development-data experiments, not demonstrated explanations. Further experiments should retain the existing model scope and use fresh evaluation trajectories after fixing any changes on development data.

The controlled prevalence and selected decision times limit external validity. Precision and F1 cannot be transferred to natural continuous-monitoring frequencies. Geometry replications share latent events and do not constitute independent event samples. Only two simulated facilities are included. The one-second simulation step, generated point clouds and centralized training do not validate measured 2D LiDAR, deployment-time detection latency or federated learning. Earlier use of the held-out geometries also limits claims about completely novel configurations. No claim of clinical readiness or broad generalization follows from these results.

## 7. Conclusion

A paired five-seed simulation comparison connects a fixed condition estimator to an existing fall classifier and compares common, facility-specific and condition-adaptive threshold policies. At the primary calibration budget, the adaptive policy reduces false-positive rates relative to facility-specific thresholds while leaving mean recall and F1 slightly lower. This result motivates reporting the full operating tradeoff and resolving the limitations of the existing detection pipeline before making broader effectiveness claims.

## Reproducibility materials

The repository contains comparison code, configurations, grouped CSVs, paired differences, frozen scores, the small condition-estimator artifact and simulator source reconstruction checks. See [the comparison report](../docs/TEACHER_REQUEST_STATUS.md) and [reproduction instructions](../docs/OPERATING_POINT_REVIEW.md). Large observation caches and CNN checkpoints are retained locally. Exact environment requirements and the repository revision should be included in the final submission artifact description.
