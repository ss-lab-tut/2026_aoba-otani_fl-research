# 先生の指示への完了報告（2026-09-08）

**9月8日までの比較実装・5 seeds実行・結果整理は完了。論文投稿の準備全体は未完了。**
「改善につながるかを確認する」という指示への答えは、今回の合成データ条件では
「P1はB3より誤検知率を下げたが、平均Recall/F1の改善は確認できなかった」である。
好ましい結果が出るまで条件を選び直すことを完了条件にはしない。

## ご指示と今回の作業範囲

国際会議への投稿準備として、新しいアイデアの追加より既存内容の論文化を優先する。
9月8日頃までにB1/B2/B3/P1を同じ条件で比較できる状態にし、とくに施設別thresholdのB3と、
condition estimatorにより動的にthresholdを変えるP1の比較を進める、というご指示に対応した。
可能であれば5 seedsを一括実行し、施設・geometry別指標とunseen geometryの結果をCSVで提示する。
条件推定器単体の追加改善、P2、新CNN、GANは保留し、9月9日頃に結果を確認して
論文のストーリーと必要な追加実験を決めることを作業の区切りとしている。

実データは存在しないため、以下の結果はすべてシミュレーションに基づく。
「実際のfall detectionへ接続」は転倒判定までの処理を実装・評価したという意味であり、
実測データを使用したという意味ではない。

## 原文との照合

| 依頼 | 状態・証拠 |
|---|---|
| B1/B2/B3/P1を同じ条件で比較 | 完了。同じ元記録分割・評価窓・ラベル・学習量。B2/B3/P1は同一CNN score |
| B3とP1を優先 | 完了。施設別thresholdと固定条件推定器によるthreshold切替、seedごとの対応差を出力 |
| できれば5 seedsを一括実行 | 完了。0〜4の5 model seeds、計10 CNN checkpointsを検証 |
| facility/geometry別Recall・Precision・F1・AUROCのCSV | 完了。全体・施設・geometry・施設×geometry・seen/unseen別、平均と標本標準偏差も保存 |
| unseen geometryの評価 | 完了。この比較のCNN学習・threshold調整・推定器学習から除いた5配置。研究全体で初見ではない |
| condition estimatorの改善を止める | 完了。固定artifactのhash一致を確認して使用 |
| fall detectionへ接続 | 完了。観測→既存CNNと条件推定→threshold→判定。実測ではなく合成データの指定時点評価 |
| P2・新CNN・GANを保留 | 遵守。追加していない |
| 9月9日頃に結果を提示 | 資料準備完了。先生への送信・面談はまだ実施していない |
| 結果を見て論文のストーリー・追加実験を決める | 次のレビューで判断。論文本文の完成・投稿はこの比較完了とは別 |

## 数値と未解決の研究上の限界

全体の主解析（5 seeds平均±標本標準偏差）：

| 方法 | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.111±0.070 | 0.646±0.054 | 0.183±0.096 | 0.647±0.131 | 0.029±0.014 |
| B2 | 0.078±0.020 | 0.553±0.104 | 0.135±0.032 | 0.588±0.063 | 0.032±0.011 |
| B3 | 0.072±0.019 | 0.582±0.065 | 0.128±0.031 | 0.588±0.063 | 0.026±0.006 |
| P1 | 0.070±0.016 | 0.620±0.056 | 0.125±0.026 | 0.588±0.063 | 0.022±0.007 |

主解析、unseen geometry、5 seeds平均：

| 方法 | Recall | Precision | F1 | AUROC | FPR |
|---|---:|---:|---:|---:|---:|
| B3 | 0.055 | 0.665 | 0.101 | 0.582 | 0.015 |
| P1 | 0.054 | 0.748 | 0.100 | 0.582 | 0.010 |

- P1のRecall/F1改善は未確認。全体でも同様で、B1の平均Recall/F1/AUROCが最も高い。
- 絶対Recallが低い。実用性能や提案法の総合的な優越を主張できない。
- AUROCは共通CNN scoreから計算するため、B2/B3/P1では同値になる。threshold切替の効果は動作点の指標で読む。
- 5 seedsはモデル乱数の反復であり、独立した5組のデータではない。元軌跡は共通。
- 実データはなく、正例率1/3の制御された合成データ。連続監視のevent recallや検知遅延、実測・FLへの一般化は未検証。
- 同じcalibration FPR予算は同じtest FPRを意味しない。6動作点の追加集計でも一貫した優越は確認できない。

これらは実装未完了と区別する。現testに合わせた再学習・seed選択・主動作点の変更はしない。
先生とのレビューで追加実験が必要となった場合は、既存CNNの学習安定性や調整条件に範囲を絞り、
開発用データで変更を固定して別の評価軌跡で確認する。

## これまでの点検・修正と実験条件

### 入力・窓ラベル・学習データの点検

既存配列の出自と時間的重複を点検した後、合成データで既存CNNと条件推定器の接続を実装した。
初回の合成データ比較では、64 frames内にABNORMALがあるかという広い窓ラベルを使用していたが、
後続監査で学習・calibrationにstrict FALLが0件と判明した。
さらに正例窓の末端が正常状態である場合が多く、条件推定器の末端5 framesとラベルの対象時刻が一致しなかった。
この初回結果は接続診断として保存し、転倒検知の有効性の根拠には使用していない。
入力の配列対応・数値・raster化も点検した。

修正版ではFALL/RECOVERED_FALLのエピソード3番目のframeを正例判定時点に固定した。
NEAR_FALLは終了後に正常へ戻った時点、ADLは末端5 framesにABNORMALがない時点を負例とした。
CNNはその判定時点までの64 frames、条件推定器は同じ時点までの5 framesを参照する。
データ生成はABNORMAL持続確率のみ0.2から0.8へ変更し、対象イベントを確保した。
これはシミュレーション条件の変更であり、条件推定器やCNN構造の改良ではない。
初回と修正版は対象・データ条件が異なるため、両者を直接の改善率として比較しない。

### B1/B2/B3/P1の定義

| 方法 | CNNと学習データ | thresholdの決め方 |
|---|---|---|
| B1 | 既存CNN、clear入力で学習 | 全体共通 |
| B2 | 同じ既存CNN、既知geometryに窓を置換して学習 | 全体共通 |
| B3 | B2と同じ学習済みCNN・同じscore | 施設ごと |
| P1 | B2と同じ学習済みCNN・同じscore | 固定条件推定器が出すclear/degraded状態に応じて切替 |

P1の動的変更は2状態に対応するthresholdの切替であり、連続値threshold回帰やオンライン再学習ではない。
全thresholdはcalibrationのみでFPR予算0.05に合わせ、testラベルで調整しない。
P1は推定器の出力を使い、testのgeometry名や真の遮蔽状態を推論時の切替に使用しない。
条件推定器は以前に固定したartifactをそのまま使用し、hash一致を確認した。

### 同条件比較の具体的な設定

- 2施設（standard/harsh）、既知3 geometry、unseen 5 geometry。
- 元記録はtrain 8、calibration 4、test 4。生成seedと記録を分割し、初回実験の軌跡も再利用しない。
- geometryの複製を除く正例イベントはtrain 96、calibration 48、test 48。
  うちstrict FALLは59、31、26件で、残りはRECOVERED_FALL。
- 施設ごとの正負構成をそろえ、正例率は1/3。窓抽出はラベル・固定乱数で決め、検知scoreで選ばない。
- CNN入力は距離・高さの2 channels、64 frames、430方位bins。固定8 m/2.5 mで正規化。
- B1/B2とも同じ初期化seed、窓数、ミニバッチ順、6 epochs、batch 16、学習率0.001。
- model seeds 0〜4で計10 CNNを学習し、各seedで同一評価窓に4方式を適用。
- シミュレーションの1 stepは1秒。64 framesを実測6.4秒の窓とは扱わない。

詳細設定は[修正版プロトコル](FALL_COMPARISON_V2.md)、
元の点検結果は[入力・ラベル監査](../results/simulated_cnn_input_audit/REPORT.md)に保存した。

## 再現方法と検証

別リポジトリのローカル変更が必要との記述を検証したところ、保存済みパッチに必要な変更は全て含まれていた。
base commitとパッチから一時ディレクトリで復元し、実験記録の全23 runtimeファイルのhash一致を確認した。
[復元手順](OPERATING_POINT_REVIEW.md)と検証用スクリプトを追加した。既存の作業ツリーは変更していない。
全学習を別環境でやり直したという意味ではない。

44 unit testsが成功。圧縮した固定scoreから再集計し、主解析0.05の620 group行の指標が一致することも確認した。
10 checkpointsのhash、各モデル16 calibration窓の実推論と保存scoreの一致、
記録分割・窓ラベル・入力配列対応・全groupの混同行列を検証済み。

固定scoreからの再集計（大きな観測cache/CNN checkpointは不要）：

```powershell
.\fl_env\Scripts\python.exe experiments/sweep_fall_operating_points.py --output-dir results/local_simulated_fall_p1_v2/recheck_operating_points
```

学習から再実行する場合は、testbedを前述のbase commitと保存パッチで復元し、
PyTorch・NumPy・PyYAMLを用意して次を実行する。既存結果を上書きしない別出力先を使用する。

```powershell
.\fl_env\Scripts\python.exe experiments/run_simulated_fall_p1_v2.py --config configs/simulated_fall_p1_v2_reproduce.json --output-dir results/local_simulated_fall_p1_v2_reproduce
```

Gitにはコード、設定、集計CSV、監査JSON、小さな固定条件推定器、圧縮score、testbedの必要な差分を保存した。
大きな観測配列・CNN checkpointsはローカルに保持し、Gitには含めていない。
従来のFL実装は別の実行経路であり、今回の実験でFLの有効性まで検証したとは主張しない。

## 9月9日のレビューで相談したいこと

1. 誤検知率低下とRecall/F1改善が未確認という両方の結果を踏まえ、論文の主張をどう定めるか。
2. 低い絶対Recallとseed間のばらつきに対し、既存CNNの学習安定性を確認する追加実験が必要か。
3. 共通calibration予算での比較に加え、動作点の比較をどこまで論文に含めるか。

P2・新CNN・GANへは広げていない。追加実験の必要性を判断できるよう、
主解析を保持し、結果が良かったseedや動作点だけを採用しない形で資料をまとめた。

## 先生へ見せる資料

- [レビュー本文・全体とunseen主表](../results/simulated_fall_p1_v2/REVIEW.md)
- [施設・geometry別CSV](../results/simulated_fall_p1_v2/summary.csv)
- [P1−B3の対応差CSV](../results/simulated_fall_p1_v2/paired_P1_minus_B3_summary.csv)
- [動作点の感度分析・再現手順](OPERATING_POINT_REVIEW.md)

9月9日の自動連絡や先生への送信は設定していない。資料を用いて結果を説明できる状態までを完了した。
