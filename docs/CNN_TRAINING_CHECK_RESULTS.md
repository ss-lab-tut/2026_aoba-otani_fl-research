# 既存CNNの学習時間：追加検証結果（2026-09-09）

既存B2のCNNを変更せず、元の学習記録の中でfit/developmentを記録単位で分離して、
事前固定した6 epochsと18 epochsを5 seedsで比較した。
既存calibration/testは評価に使用していない。条件推定器とB3/P1のthresholdを変更していない。

## 結果

平均±標本標準偏差：

| 評価 | 指標 | 6 epochs | 18 epochs | 対応差（18−6） |
|---|---|---:|---:|---:|
| fit | AUROC | 0.7088±0.0581 | 0.9964±0.0065 | +0.2876±0.0542 |
| development | AUROC | 0.5519±0.0366 | 0.7901±0.0597 | +0.2382±0.0637 |
| fit | cross entropy | 0.5908±0.0290 | 0.0645±0.0721 | −0.5263±0.0560 |
| development | cross entropy | 0.6578±0.0328 | 1.0108±0.3654 | +0.3531±0.3573 |

5 seedsすべてでdevelopment AUROCは上昇した。同時に、5 seedsすべてでdevelopment cross entropyは悪化した。
AUROCは高いほどよく、cross entropyは低いほどよい。この2つは異なる性能面を評価している。

![学習曲線](../results/cnn_training_duration/learning_curves.png)

## 分かったことと、まだ分からないこと

この開発分割では、6 epochs時点からさらに学習すると、学習記録外でもスコアの順位付けが改善する。
したがって、以前の低AUROCを既存CNNの表現能力の限界だけとして扱うのは早い。
ただし開発cross entropyは悪化し、fit/developmentの差も大きいため、
18 epochsを総合的に優れた設定としたり、学習時間だけで問題が解決したとしたりはできない。
cross entropy単独から確率校正の悪化や過学習の原因までを確定することもできない。

この結果は、低FPRでのRecall/F1改善、unseen geometryでの改善、B3に対するP1の優位性を検証したものではない。
developmentは2記録で、5 seedsは同じ分割上のモデル乱数の反復である。
主実験よりfit記録数を減らしたため、主実験の6 epochsとの直接比較はしない。
12 epochs等の途中結果も開示するが、最良epochを選び直していない。

## 次の判断

学習時間は引き続き確認すべき要因となった。次に転倒検知としての効果を確認する場合は、
比較する学習条件を先に固定し、独立したcalibration記録でthresholdを調整して、
新しい評価軌跡でB1/B2/B3/P1を同じ条件・5 seedsで比較する必要がある。
AUROCの上昇だけを理由に主解析を更新せず、現在の6 epochs主解析と本開発検証を両方保存する。
推定器改善や新CNNへ広げる根拠は、この検証からは得ていない。

## 成果物と検証

- [実行前に固定した計画](CNN_TRAINING_CHECK.md)
- [全epochの指標と図](../results/cnn_training_duration/REPORT.md)
- [seed別CSV](../results/cnn_training_duration/metrics.csv)
- [18−6の対応差CSV](../results/cnn_training_duration/paired_18_minus_6.csv)
- [実験プロトコルとhash](../results/cnn_training_duration/protocol.json)

10 checkpointsのhash、50行の指標、全対応差、実行コードのhashを検証した。
記録分割のテスト3件を追加し、全47 unit testsが成功した。学習曲線のPNGも表示確認した。
詳細checkpointは`results/local_simulated_fall_p1_v2_duration/`にローカル保存している。

```powershell
.\fl_env\Scripts\python.exe experiments/check_cnn_training_duration.py --resume
.\fl_env\Scripts\python.exe experiments/report_cnn_training_duration.py
```

`--resume`は同じ入力・設定・コードを検証し、完成済みseedを再利用する。
