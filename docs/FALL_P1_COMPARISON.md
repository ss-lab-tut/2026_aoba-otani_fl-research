# 9月8日比較準備・9月9日結果レビュー

**2026-09-06追記（本書の前提を訂正）：実データは存在しないことを確認した。**
元資料の受領待ちを解除し、[シミュレーションによるB1/B2/B3/P1比較](SIMULATED_FALL_P1.md)へ移行する。
以下の「実データ」は出自未確認の既存配列を誤ってそう呼んだもので、実測結果を意味しない。
以下の元資料待ちの日程は廃止し、参照先のシミュレーション日程を優先する。

2026-09-06時点。投稿準備を優先し、範囲をB1・B2・B3・P1に固定する。
主比較はB3対P1、特にunseen geometry。condition estimatorの追加改善、P2、新CNN、GAN、B4は今回の作業範囲外。

## 実装済みと未完了

- 共通の転倒スコアからB1/B2/B3/P1を比較する5 seeds評価器を実装した。
- Recall、Precision、F1、raw-score AUROC、threshold差分スコアのAUROC、FPR、混同行列を窓単位で出力する。
- 施設別、geometry別、施設×geometry別、seen/unseen別、全体を出力する。
- 各seedの値、平均・標本標準偏差、P1−B3の対応差、適用thresholdと予測を保存する。
- 既存実データのB1/B2相当について、保存済み5 seedsを実際に集計した。
- **実データでのB3/P1接続とunseen geometry評価は未完了**。施設・geometry・記録対応と、実データに適合する条件推定入力が不足している。
- 新規の5 seeds学習実験を完了したわけではない。既存結果の再集計と、合成テスト入力による評価器の動作検証を区別する。

## 固定する比較条件

| 方法 | 転倒スコア | threshold |
|---|---|---|
| B1 | 既存の素のCNN | calibration全体で一つ |
| B2 | 既存の補強CNN | calibration全体で一つ |
| B3 | B2と同一 | 施設ごとに一つ |
| P1 | B2と同一 | frozen estimatorの推定状態ごとに切替 |

P1の最小実装はclear/degradedの2状態ルーティング。入力`condition_score`は観測から得た
**固定済み条件推定器のscore − 固定済み条件判定threshold**とし、設定のcutoffを0にする。
真の遮蔽ラベル・geometry ID・シミュレータ内部設定でP1のthresholdを選ばない。
条件推定器のモデル・前処理・判定thresholdはこの比較のtestを使わず固定する。
これは連続補間型のthresholdではない。

すべてcalibrationの非転倒窓FPR予算0.05からthresholdを選ぶ。
B1/B2は全体、B3は施設、P1は推定状態ごとに同じ予算を適用する。
予算はtest FPRの保証ではない。RecallとFPRを必ず併記する。
施設・状態のcalibrationに片方のクラスしかない場合、および未登録施設・状態はB2へfallbackする。
fallbackはmanifestに記録する。B3にtestのラベル付き施設調整データを渡さない。

AUROCはthresholdを選んだだけでは変化しないので、B2/B3/P1のraw-score AUROCは同一になる。
動的判定の順序変化を見る`margin_auroc`は別列とし、raw AUROCと混同しない。
分母ゼロの指標は空欄。5 seedsの一部で未定義なら平均も空欄にして有効seed数を併記する。

## 入力と一括実行

`configs/fall_policy_comparison.example.json`をコピーし、実際の推定器artifactのSHA-256、
学習geometry・学習記録IDをseedごとに記入する。空欄のまま実験を実行できない。
推定器を全seedで共有する場合は同じ来歴を記入する。

入力CSVの列：

```csv
seed,sample_id,recording_id,split,facility,geometry,label,b1_score,augmented_score,condition_score
```

- `split`: train/calibration/test。train行は分割監査用でscore3列は空欄可。
- `sample_id`: seed内で一意。同一窓のB1/B2スコアを同一行に置くことで比較を対応させる。
- `recording_id`: 元記録のID。同一記録を複数splitに入れると停止する。
- `label`: 転倒1、非転倒0。条件劣化ラベルではない。
- `facility`、`geometry`: 元資料の対応を使用。人工beam欠損の位置を実施設IDに代用しない。
- score生成側で、同じCNN構造・学習量・分割・前処理を用い、B2/B3/P1は同じcheckpointに固定する。
- 各seedで同一の評価groupが必要。欠損seedを平均で隠さず停止する。

```powershell
.\fl_env\Scripts\python.exe experiments/compare_fall_policies.py results/fall_scores.csv --config configs/fall_policy_comparison.json --output-dir results/fall_p1_comparison
```

出力は`metrics.csv`、`summary.csv`、`predictions.csv`、`paired_P1_minus_B3.csv`、
`paired_P1_minus_B3_summary.csv`、`manifest.json`。
入力・コード・設定のhashと各seedのthresholdを記録する。
評価器はscoreを消費する処理で、CNN学習や元点群からの条件推定を代行するものではない。
入力作成者によるartifact/前処理の来歴宣言は保存するが、宣言だけで独立性を認証しない。

unseen geometryはCNNのtrain/calibrationと条件推定器trainの和集合にないtest geometry。
学習にないgeometryでもthreshold調整に使っていればseenとする。
geometry holdoutと新施設holdoutは異なる。既知施設の新geometryでもB3はその施設の既存thresholdを使用できる。
被験者独立性は記録IDだけでは保証しない。元データを受領後に被験者分割も確認する。

## 現時点の実行結果

既存の通常入力・purged within-series診断の再集計（5 seeds、平均±標本標準偏差）：

| 方法相当 | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.869±0.198 | 0.921±0.096 | 0.878±0.107 | 0.972±0.045 | 0.038±0.047 |
| B2 | 0.844±0.201 | 0.956±0.099 | 0.880±0.114 | 0.978±0.029 | 0.024±0.054 |

保存先：`results/fall_comparison_existing/`。B3・P1の性能改善を示す結果ではない。
入力hash、checkpoint hash、分割一致、時間窓のフレーム非共有を検証してから保存した。
既存のB2はbeam-mask補強であり、過去の研究計画のB2再現とは区別する。
seedごとに時間foldも変わるため、標準偏差は初期化と分割の両方の変動を含む。
施設・geometry不明、被験者独立性未確認、元scalerのfit範囲未確認なので投稿用の正式な汎化結果にはしない。

再集計：

```powershell
.\fl_env\Scripts\python.exe experiments/export_existing_fall_metrics.py
.\fl_env\Scripts\python.exe -m unittest discover -s tests -v
```

検証結果：34 tests passed。新規4テストで既知混同行列・AUROCの同点処理、
施設/推定状態routing、testラベル非依存、記録漏洩拒否、seed不足拒否、
5 seedsのCSV出力、条件推定学習geometryも含むunseen判定を確認した。

## 9月8日・9月9日に向けて

9月8日目標：元資料を対応づけ、実点群と条件推定器の入力適合性を確認し、
同じ分割のCNNスコアと条件推定scoreを生成、5 seedsのB3/P1比較を実行する。
実点群が既存推定器の高さ特徴等を持たない場合、そのまま転用したり値を捏造したりせず、
シミュレーション先行か実データ入力対応かを研究方針として決める。
新しい推定器の精度探索は行わない。

9月9日レビュー資料：主表B3対P1、unseen geometry別表、5 seeds対応差、FPRとRecallの変化、
fallback件数、最悪geometry、負の結果も含む限界をまとめる。その結果を見て論文の主張と追加実験を決める。
この日程は作業目標であり、日付指定の自動実行・自動連絡は設定していない。
