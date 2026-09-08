# シミュレーションでのB1/B2/B3/P1比較

**2026-09-07追加監査：下位分類FALLが学習・調整とも0件だったため、以下の初回設定は
転倒検知の本評価用には使わない。** [入力・ラベル・データ監査](../results/simulated_cnn_input_audit/REPORT.md)を参照。
広いABNORMAL窓分類を用いた接続診断として保存し、対象ラベル・時間範囲・データ構成を先に是正する。

2026-09-07：5 seedsの実行・整合性確認・初回診断が完了した。
[結果レビュー](../results/simulated_fall_p1/REVIEW.md)と[全比較表](../results/simulated_fall_p1/REPORT.md)を参照。
P1はB3より平均Recall/F1が上がったが、転倒CNNの評価AUROCが約0.51であり、有効性の主張は保留する。

2026-09-06に実データが存在しないことを確認した。投稿準備はシミュレーションを対象とし、
実データの受領を待たない。以前の`real_*`診断は出自未確認の既存配列についての記録で、今回の比較に混ぜない。

## 固定した構成

- CNNは`model.LidarCNN`を変更せず再利用する。新しいCNNは追加しない。
- 点群から430方位binの平均距離・平均高さを作り、64 framesの2チャンネル入力にする。
  正規化は固定の8 m/2.5 m、欠損binは0。ラベルや遮蔽物の真値は入力しない。
- 転倒ラベルは窓内に一つでもABNORMAL状態があること。これは条件劣化ラベルと別である。
  ABNORMALはシミュレータの転倒・転倒後状態を含む代理ラベルで、臨床的な転倒イベント定義ではない。
- P1は既存QuadraticRidgeScore・11形状特徴・因果的な5-frame平均を維持する。
  既存の学習/調整cacheから同じモデルを復元し、以前の固定thresholdとの一致を検証する。
  条件推定器の追加改善や候補探索はしない。
- 条件scoreは各窓の末端frameで計算する。P1はclear/degraded推定に応じて二つの転倒thresholdを切り替える。
- 施設はstandard（noise=1）とharsh（noise=3）。同じ5 m×5 mの部屋で観測ノイズを変える。
- trainは生成seed301/302、calibrationは401/402、testは501/502。
  geometry違いの同じ潜在軌跡は同一splitにまとめる。
- 既知geometryはclear、vertical_mid、central_furniture。
- 未知geometryは既存条件検証のvertical_new、horizontal_new、low_new、wide_new、two_new。
  これらは条件推定器学習、CNN学習、転倒threshold調整には入れない。
  条件推定の過去の評価で見たgeometryなので、研究全体で完全に未見の確認実験とは呼ばない。
- 5 model seeds（0〜4）は同じ生成データと評価分割を共有し、初期化・ミニバッチ順・補強選択を変える。

| 方法 | 学習入力 | 転倒threshold |
|---|---|---|
| B1 | clearの窓 | 共通 |
| B2 | 同じ窓を既知geometryのいずれかへ置換 | 共通 |
| B3 | B2の同一checkpoint | 施設別 |
| P1 | B2の同一checkpoint | 条件推定状態別 |

B1/B2の窓数・ラベル・初期化・6 epochs・batch size・学習率をそろえる。
B2の補強は既知geometryから窓ごとに一つ選ぶため、窓数を増やさない。
B1とB2のcalibration/testの入力は完全に共通とする。
転倒thresholdの調整はcalibrationのみ、非転倒窓FPR予算0.05を各方式に同じように適用する。

## 実行と出力

```powershell
.\fl_env\Scripts\python.exe -u experiments/run_simulated_fall_p1.py
.\fl_env\Scripts\python.exe experiments/report_simulated_fall_p1.py
.\fl_env\Scripts\python.exe experiments/diagnose_simulated_fall_p1.py
```

設定：`configs/simulated_fall_p1.json`。
出力：`results/local_simulated_fall_p1/`。既存結果の上書きを防ぐため空の出力先が必要。
再実行する場合は`--output-dir results/local_simulated_fall_p1_rerun`などを指定する。
集計・診断スクリプトは既存成果物を読み取り、小さな表を`results/simulated_fall_p1/`へ保存する。
これら二つだけの再実行ならCNNを再学習しない。別の実験先を読む場合は`--source`を指定する。

- `protocol.json`：設定、環境、入力・コードhash
- `condition_model.npz`：固定条件推定器
- `observations.npz`、`windows.csv`：生成観測と窓対応
- `B1_seed*.pt`、`B2_seed*.pt`：既存CNNの5 seeds学習結果
- `scores_seed*.csv`、`scores.csv`：全方式で共有する転倒・条件score
- `training.json`：各seedのloss、学習窓、checkpoint hash
- `comparison/metrics.csv`、`summary.csv`：施設/geometry/seen/unseen別指標
- `comparison/paired_P1_minus_B3*.csv`：対応するseedの差とその平均・標本標準偏差
- `comparison/predictions.csv`、`manifest.json`：適用threshold、予測、fallbackと来歴

指標はRecall、Precision、F1、raw-score AUROC、threshold差分score AUROC、FPR、混同行列。
B2/B3/P1のraw-score AUROCは同じ。判定改善はRecall/FPR/F1と対応差で判断する。

## 報告の範囲

これはシミュレーションの点群を使う既存CNNの集中学習診断で、FLの有効性を示す比較ではない。
高さを持つ合成点群と2.5-D遮蔽モデルを用いており、実測2D LiDARでの有効性は主張しない。
時間窓は重なり、同じ軌跡のgeometry違いも含む。独立標本数を窓数と同一視しない。
event recall、検知遅延、event誤報/時間はまだ評価しない。

9月8日までの目標は5 seeds比較と結果点検、9月9日のレビューではB3/P1主表、
unseen geometry、RecallとFPRの対応差、最悪条件と失敗例を見て論文の主張と必要な追加実験を決める。
日付指定の自動連絡は設定していない。
