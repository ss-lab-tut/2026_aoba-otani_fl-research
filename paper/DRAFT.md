# Facility-Specific and Condition-Adaptive Thresholds for Simulated Fall Detection: A Five-Seed Comparison

Working manuscript, updated 2026-09-15. Venue-independent draft for internal review. An initial set of verified references is included; related-work coverage, author details and venue formatting still require review. Original numerical tables are generated in [TABLES.md](TABLES.md), with [fresh-trajectory results](../results/fresh_trajectory_comparison/REPORT.md) reported separately. This draft reports a controlled simulation study, not measured-sensor or federated-learning validation.

## Abstract

We evaluate condition-adaptive versus facility-specific thresholds for an existing fall-detection CNN. Four policies are compared over five model seeds in two simulated facilities with three known and five held-out geometries. The augmented common-threshold, facility-specific and condition-adaptive policies share identical fall scores; thresholds are fitted on calibration data at an empirical false-positive-rate budget of 0.05. The initial six-epoch experiment found no mean recall or F1 improvement from condition adaptation. A development-only training-duration check motivated a separately fixed comparison of six and eighteen epochs using new calibration and test trajectories. At eighteen epochs on held-out geometries, the adaptive policy achieves mean recall 0.323, F1 0.463 and false-positive rate 0.030, compared with 0.301, 0.432 and 0.037 for facility-specific thresholds. However, the clear-input baseline has higher recall and F1, while the common-threshold augmented baseline has fewer false positives. Facility-level effects differ: the aggregate benefit is not a uniform improvement within each facility. These results show that the observed benefit of threshold adaptation depends on the evaluated training condition and comparator. Controlled event sampling, shared trajectories across model seeds and the absence of measured data limit broader conclusions.

## 1. Introduction

The operating threshold is part of a fall-detection system: the same fall score can produce different detection and false-positive rates depending on how that threshold is assigned. Facility-specific calibration assigns a threshold using a facility identifier. An alternative is to estimate the observation condition from available sensor statistics and select a threshold using that estimate. These alternatives should be compared using the same fall scores to separate the effect of the threshold policy from differences in the detector.

This study asks whether a fixed condition estimator yields a downstream benefit when connected to an existing fall classifier. We focus on a controlled comparison rather than introducing a new classifier or improving the condition estimator. The central comparison is between facility-specific and condition-adaptive thresholds. Two additional baselines distinguish the effects of geometry augmentation and threshold assignment.

Our contributions within this experimental scope are: (1) a paired four-policy evaluation with disjoint recording partitions and held-out geometries; (2) an explicit endpoint-label protocol linking the CNN context and condition-estimation context to the same decision time; and (3) a five-seed analysis separating training-duration effects from threshold-policy effects. The initial six-epoch results and the development-driven fresh-trajectory comparison are both retained. The additional eighteen-epoch comparison improves the adaptive policy's mean recall and F1 relative to facility-specific thresholds, without establishing superiority over all baselines.

## 2. Related work — initial verified references

Bouazizi et al. study activity and fall detection using multiple 2D Lidars, an image-like fused representation and a ConvLSTM in simulated furnished environments. Their work provides relevant context for occlusion-aware sensing. Our experiment instead holds an existing CNN's scores fixed when comparing threshold policies; different data, tasks and metrics prevent a direct numerical performance comparison. [Bouazizi et al., 2024](https://www.mdpi.com/1424-8220/24/2/626).

Guo et al. study confidence calibration of neural-network probabilities. This distinction matters here because a threshold-calibration partition sets decision boundaries but does not by itself establish well-calibrated probabilities. Our development results report AUROC and cross entropy separately; we do not apply probability-calibration methods in the present comparison. [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html).

Neyman-Pearson classification treats type-I and type-II errors asymmetrically; Tong et al. analyze classifiers and sample requirements for this setting. Our threshold rule enforces an empirical calibration FPR budget only. It is not a high-probability population-FPR guarantee, particularly with dependent geometry replications. [Tong et al., 2020](https://jmlr.org/papers/v21/18-577.html).

This initial context does not establish novelty or exhaustive related-work coverage. Observation-condition adaptation and closely matched fall-detection baselines still require a focused review before submission. Bibliographic entries are provided in [references.bib](references.bib).

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

## 5. Original six-epoch results

The original all, seen and unseen tables are available in [TABLES.md](TABLES.md). [RESULTS_TABLES.md](RESULTS_TABLES.md) brings the original unseen comparison, both fresh-data durations and facility-level paired differences together; [results_tables.tex](results_tables.tex) provides the corresponding LaTeX tables. All four policies are retained.

Across all geometries, B1 has the highest mean recall, F1 and AUROC: 0.111, 0.183 and 0.647, respectively. B3 attains recall 0.072, precision 0.582, F1 0.128 and FPR 0.026. P1 attains recall 0.070, precision 0.620, F1 0.125 and FPR 0.022. The mean paired P1-minus-B3 differences are -0.0026 in recall, -0.0033 in F1 and -0.0042 in FPR. Thus, the adaptive threshold does not improve mean recall or F1 in the overall evaluation.

On held-out geometries, B3 and P1 attain mean recalls of 0.055 and 0.054, mean F1 scores of 0.101 and 0.100, and mean FPRs of 0.015 and 0.010. P1 has lower FPR in all five seeds, but recall differences vary across seeds and the mean recall difference is slightly negative. These findings support a descriptive reduction in false positives in the evaluated setting, not statistical significance or overall superiority.

A descriptive sensitivity analysis recalibrates thresholds at budgets 0.01, 0.025, 0.05, 0.10, 0.15 and 0.20 using the same frozen scores. It does not establish consistent dominance across operating points. At budget 0.15, higher mean recall for P1 accompanies higher mean test FPR. Equal calibration budgets do not imply equal realized test FPR. The primary budget remains 0.05; the test results are not used to select a replacement operating point.

## 6. Additional analyses and limitations

### Frozen-model diagnosis

A post-hoc analysis of the frozen checkpoints compares their actual training windows with test subsets (see [the diagnostic figures](figures/README.md)). Mean training AUROC is 0.815 for B1 and 0.781 for B2, versus 0.671 and 0.610 on clear test inputs, 0.663 and 0.599 on known test geometries, and 0.638 and 0.582 on held-out geometries. The drop is therefore already present on known-condition test data; held-out geometry alone cannot account for the observed performance. Training and test mixtures differ, especially for geometry-augmented B2, and these descriptive gaps do not isolate optimization or overfitting as the cause. This analysis uses the same frozen models and does not change the main experiment.

### Development-only training-duration check

Following the frozen-model diagnosis, we predefined a training-duration check for the B2 backbone using only the original training recordings. Generation seeds 10001–10003 supplied fitting data and seed 10004 supplied development data in each facility. The existing calibration and test partitions were not evaluated. Architecture, geometry replacement, Adam learning rate and batch size were held fixed. Five model seeds were trained for 18 epochs, with the primary contrast fixed in advance as epoch 18 minus epoch 6.

Development AUROC increased from 0.5519 ± 0.0366 to 0.7901 ± 0.0597 (paired difference +0.2382 ± 0.0637), while development three-class cross entropy increased from 0.6578 ± 0.0328 to 1.0108 ± 0.3654. Both directions held in all five seeds. Fitting AUROC rose from 0.7088 to 0.9964. Longer training therefore improved ranking on this development split without improving cross entropy. This smaller fitting partition is not directly comparable with the main experiment. The check does not establish improved low-FPR detection, held-out-geometry performance or P1 superiority. Intermediate epochs are reported but no best epoch is selected. The six-epoch main comparison remains unchanged. The fresh-trajectory comparison described next evaluates this fixed eighteen-epoch candidate with all four threshold policies. See [the development results](../docs/CNN_TRAINING_CHECK_RESULTS.md).

### Fresh-trajectory six/eighteen-epoch comparison

After the development check described above, we fixed an eighteen-epoch condition and new calibration seeds 13001/13002 and test seeds 14001/14002. The original six-epoch checkpoints were evaluated on the same new data. Training recordings, windows, initializations, augmentation selections and first-six-epoch loss sequences were verified to match. All four policies were recalibrated separately within each training-duration condition using the same new calibration partition and budget 0.05. No epoch or threshold budget was selected using the new test results.

On the new held-out geometries, B2 recall rises from 0.0558 at six epochs to 0.3008 at eighteen epochs, F1 from 0.1011 to 0.4493, and AUROC from 0.6690 to 0.8664. This is a training-duration result, not a P1 effect. Within eighteen epochs, P1 versus B3 yields recall 0.3233 versus 0.3008, F1 0.4633 versus 0.4321, and FPR 0.0304 versus 0.0375. The paired differences are approximately +0.0225 ± 0.0253 in recall, +0.0312 ± 0.0295 in F1, and -0.0071 ± 0.0043 in FPR. Four seeds improve recall/F1 and one worsens them; all five reduce FPR. B1 nevertheless has higher mean recall and F1 (0.3367 and 0.4818), while B2 has lower FPR (0.0129).

The new six-epoch evaluation does not show an adaptive-policy benefit over B3 in mean recall or F1. Thus, both the negative six-epoch findings and the favorable eighteen-epoch P1-versus-B3 comparison must be presented. Equal calibration budgets do not yield equal test FPR, and the P1/B3 recall gains across durations accompany higher FPR. See [all policies, paired differences and the figure](../results/fresh_trajectory_comparison/REPORT.md). These are additional simulated endpoint results, not a replacement for the original analysis or evidence of statistical significance.

Facility-level results qualify the aggregate benefit. On held-out geometries, the harsh facility has P1-minus-B3 differences of +0.0567 in recall and +0.0042 in FPR, while the standard facility has -0.0117 in recall and -0.0183 in FPR. Thus, neither facility exhibits both higher recall and lower FPR in its own aggregate result. Among the ten held-out facility-by-geometry groups, mean recall increases in seven and decreases in three; mean F1 increases in eight and decreases in two. Mean FPR decreases in five, is unchanged in one, and increases in four. These dependent subgroups are descriptive, not independent statistical trials. All sixteen known/held-out facility-by-geometry groups are disclosed in [SUBGROUPS.md](../results/fresh_trajectory_comparison/SUBGROUPS.md); the favorable pooled result must not be interpreted as uniform facility-level improvement.

### Interpretation of the main comparison

In the original six-epoch analysis, connecting the frozen condition estimator to threshold selection changes the detector's operating behavior but does not improve mean recall or F1. In the additional eighteen-epoch analysis, P1 improves those means relative to B3, but still does not outperform every baseline. False positives and missed detections must therefore be presented together. The clear-input baseline retains higher mean recall and F1, so the evidence does not support a general superiority claim for the chosen augmentation and adaptation combination.

Absolute recall is low. The current comparison does not determine whether optimization, limited training diversity, the long input context or other aspects of the protocol are the principal cause. These are hypotheses requiring separate development-data experiments, not demonstrated explanations. Further experiments should retain the existing model scope and use fresh evaluation trajectories after fixing any changes on development data.

The controlled prevalence and selected decision times limit external validity. Precision and F1 cannot be transferred to natural continuous-monitoring frequencies. Geometry replications share latent events and do not constitute independent event samples. Only two simulated facilities are included. The one-second simulation step, generated point clouds and centralized training do not validate measured 2D LiDAR, deployment-time detection latency or federated learning. Earlier use of the held-out geometries also limits claims about completely novel configurations. No claim of clinical readiness or broad generalization follows from these results.

## 7. Conclusion

A paired five-seed simulation study connects a fixed condition estimator to an existing fall classifier and compares common, facility-specific and condition-adaptive threshold policies. Initial six-epoch results do not show improved mean recall or F1 from adaptation. Following a development-only duration check, a separately fixed eighteen-epoch comparison on fresh trajectories shows higher mean recall/F1 and lower FPR for P1 than B3. The benefit does not extend to superiority over every baseline. Preserving all stages of the evaluation exposes the dependence of the observed result on the training condition and supports a limited, comparator-specific conclusion.

## Reproducibility materials

The repository contains comparison code, configurations, grouped CSVs, paired differences, frozen scores, the small condition-estimator artifact and simulator source reconstruction checks. See [the comparison report](../docs/TEACHER_REQUEST_STATUS.md) and [reproduction instructions](../docs/OPERATING_POINT_REVIEW.md). Both fresh-trajectory comparisons were reproduced from their Git-sized compressed score artifacts, without reading local observations or checkpoints; confusion counts, summaries and paired differences match at numeric tolerance 1e-12. See [the replay verification](../results/fresh_trajectory_comparison/REPLAY_VERIFICATION.md). Large observation caches and CNN checkpoints are retained locally. Exact environment requirements and the repository revision should be included in the final submission artifact description.
