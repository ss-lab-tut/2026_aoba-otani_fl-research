# 比較実装・5 seeds実行・結果整理：研究概要

2026-09-14。目的は、既存の条件推定器を転倒判定へ接続し、施設別thresholdのB3と条件適応thresholdのP1を公平に比較すること。
条件推定器単体の改良、新CNN、P2、GANには広げていない。

## 実施済み

同一入力・分割・学習条件でB1/B2/B3/P1を比較し、5 model seedsのRecall・Precision・F1・AUROC・FPRを
施設・geometry・unseen別にCSVへ保存した。B2/B3/P1は同じCNN scoreを使い、thresholdはcalibrationだけで決める。
全データは合成で、元の学習データと評価データは記録単位で分離している。

## 論文に残す3段階の結果

| 段階 | 確認したこと | 結論の範囲 |
|---|---|---|
| 初期6 epochs比較 | P1のFPRはB3より低いが平均Recall/F1は改善せず | 初期の否定的な結果を保持 |
| 学習時間の開発検証 | 18 epochsで開発AUROCが上昇、cross entropyは悪化 | 学習時間を確認する根拠。転倒性能改善そのものではない |
| 新規軌跡で6/18 epochs比較 | 同じ新規データ上で18 epochsの識別性能が改善。18 epochs内のP1はB3より平均Recall/F1が高くFPRが低い | 学習の効果とthresholdの効果を区別した追加検証 |

新規unseen・18 epochsの5 seeds平均：

| 方法 | Recall | F1 | FPR |
|---|---:|---:|---:|
| B1 | 0.337 | 0.482 | 0.026 |
| B2 | 0.301 | 0.449 | 0.013 |
| B3 | 0.301 | 0.432 | 0.037 |
| P1 | 0.323 | 0.463 | 0.030 |

## 平均値だけでは説明できない点

P1対B3のunseen比較では、harsh施設でRecallが増えるとともにFPRも増え、
standard施設ではFPRが下がるとともにRecallも下がった。
全体平均での改善を、すべての施設・geometryで同時に改善したとは説明しない。
B1は平均Recall/F1が高く、B2はFPRが低いため、P1が全方式で最良とも言えない。

論文の主張は「今回の学習条件と評価範囲では、条件適応thresholdが施設別thresholdに対する
集計性能を改善した。ただし施設ごとに利点と不利益が異なった」に限定する。
初期6 epochsの結果も含め、検出器の学習条件によって効果が異なることを示す。

## レビューで決めたいこと

1. 3段階を併記した限定的な主張で論文をまとめるか。
2. 全体・unseen主表に加え、施設別の利点と不利益を本文へどの形で示すか。
3. 投稿先・ページ制限に合わせ、原稿の長さと関連研究の範囲を確定する。

実測・自然頻度の連続監視・event recall・検知遅延・FLは未検証。
5 seedsは固定軌跡上のモデル乱数であり、独立した5データセットではない。
追加学習やtestを使った設定変更は、この資料整理では行っていない。

## 資料への入口

- [英語草稿](DRAFT.md)
- [最新比較と再現手順](../docs/FRESH_TRAJECTORY_RESULTS.md)
- [全16施設×geometry条件](../results/fresh_trajectory_comparison/SUBGROUPS.md)
- [Git内scoreからの再集計検証](../results/fresh_trajectory_comparison/REPLAY_VERIFICATION.md)
- [初期比較の表](TABLES.md)
