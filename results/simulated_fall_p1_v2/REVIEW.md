# 9月9日レビュー用：B1/B2/B3/P1の修正版5 seeds比較

2026-09-08補足：[6動作点の感度分析と再現手順](../../docs/OPERATING_POINT_REVIEW.md)を追加した。
主解析の動作点は変更していない。

2026-09-07時点で、B1/B2/B3/P1の同条件比較・5 seeds実行・施設/geometry別CSV出力を完了した。
**現時点の結果は「P1はB3よりFPRを下げるが、平均Recall/F1の改善は確認できない」。**
改善を示したいという意図に合わせて結果を選び直さず、この表をレビューの出発点にする。

## 実施内容

| 項目 | 完了内容 |
|---|---|
| B1/B2/B3/P1の同条件比較 | 同じ分割・評価窓。B2/B3/P1は同一のCNN score |
| B3対P1を優先 | 施設別threshold対条件推定状態別threshold、対応するseed間の差を出力 |
| 5 seeds一括実行 | 0〜4の5 model seeds、B1/B2計10 checkpoints、同一学習量 |
| facility/geometry別CSV | Recall、Precision、F1、AUROC、FPR、混同行列、平均±標本標準偏差 |
| unseen geometry | CNN学習・threshold調整・条件推定器学習から外した5 geometryを評価 |
| 条件推定器の改善を止める | 以前のartifactをhash確認してそのまま使用 |
| P1までを完成 | 観測→既存CNN score・固定条件推定器→threshold切替→転倒判定まで接続 |
| P2・新CNN・GANを保留 | 追加していない |

## 主表：全体（平均±標本標準偏差）

| 方法 | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B1 | 0.111±0.070 | 0.646±0.054 | 0.183±0.096 | 0.647±0.131 | 0.029±0.014 |
| B2 | 0.078±0.020 | 0.553±0.104 | 0.135±0.032 | 0.588±0.063 | 0.032±0.011 |
| B3 | 0.072±0.019 | 0.582±0.065 | 0.128±0.031 | 0.588±0.063 | 0.026±0.006 |
| P1 | 0.070±0.016 | 0.620±0.056 | 0.125±0.026 | 0.588±0.063 | 0.022±0.007 |

全体のP1−B3の対応差はRecall −0.003±0.005、F1 −0.003±0.008、FPR −0.004±0.003。
Precisionは上がるが、検出性能全体の優越を示す結果ではない。
B1のRecall/F1/AUROCが平均では最も高く、補強を含むB2/B3/P1がB1を上回る結果でもない。
全方式でRecallは低いため、実用性能に到達したとは扱わない。

## Unseen geometry

| 方法 | Recall | Precision | F1 | AUROC | FPR |
|---|---|---|---|---|---|
| B3 | 0.055±0.019 | 0.665±0.075 | 0.101±0.032 | 0.582±0.058 | 0.015±0.007 |
| P1 | 0.054±0.014 | 0.748±0.099 | 0.100±0.024 | 0.582±0.058 | 0.010±0.006 |

unseenでのFPRは5 seedsすべてでP1が低い。
Recall差は3 seedsで+0.0042、1 seedで0、1 seedで−0.0167で、平均では−0.0008。
平均F1差もほぼ0（小さな負値）。統計的有意差を主張する検定は行っていない。
壁geometry（vertical_new/horizontal_new）のRecallは特に低く、geometry別表を主表と併読する。
施設×geometryには両方式ともRecall 0の条件が残る。平均値だけで改善と判定しない。

## 修正した実験条件

初回のABNORMAL窓診断にFALL不足が見つかったため、今回の表と初回の数値を直接の前後改善率にはしない。

- 対象はFALL/RECOVERED_FALL。開始後3番目のframeを判定時点とし、NEAR_FALLとADLは負例。
- シミュレータのABNORMAL持続確率を0.2→0.8に固定し、正しい対象イベントを確保。
- geometryの重複を除く正例イベント数は学習96、調整48、評価48。
  strict FALLはそれぞれ59、31、26件で、残りはRECOVERED_FALL。
- 記録数は学習8、調整4、評価4。元記録・生成seedを分割し、以前の軌跡は再利用していない。
- 正負比率を施設間でそろえ、正例率は1/3。Precision/F1はこの制御された構成に依存する。
- CNN構造、距離/高さ430-bin入力、固定条件推定器は据え置き。
- CNNは判定時点までの64 frames、条件推定器は同じ時点までの5 framesを参照する。
- 全方式に同じcalibration FPR予算0.05を適用し、testラベルでthresholdを選んでいない。
- B3/P1でcalibration不足によるfallbackは発生していない。

実測データは使用していない。合成3座標点群・2.5-D遮蔽・集中学習の制御比較であり、
実測2D LiDAR、FLの有効性、自然頻度の連続監視、event recall、検知遅延の検証とは区別する。
同じgeometryは過去の条件推定研究で調べているため、研究全体で完全な初見のgeometryとも呼ばない。

## 9月9日の判断材料

条件推定の良さが転倒Recall改善へそのままつながる、というストーリーは現時点で支持されていない。
現在支持される観察は、固定した転倒scoreに対するP1のthreshold切替で誤検知を減らせたこと。
ただしB3やB1とのRecall/F1比較、低い絶対Recallを併記する必要がある。

この結果を見て、論文をどの主張でまとめるか、最小限の追加実験が必要かを決める。
追加を行う場合も、P2や新CNN/GANへ広げず、既存CNNの学習の安定性と同じFPR水準での比較を先に検討する。
現在のtestに合わせたモデルやthresholdの選び直しは行っていない。

## 成果物と検証

- [summary.csv](summary.csv)：全体・施設・geometry・施設×geometry・seen/unseen別の集計
- [metrics.csv](metrics.csv)：各seedの指標と混同行列
- [paired_P1_minus_B3_summary.csv](paired_P1_minus_B3_summary.csv)：対応差の平均・標本標準偏差
- [REPORT.md](REPORT.md)：全比較・unseen geometry・最悪条件の表
- `event_preflight.json` / `selected_endpoints.csv`：学習前検査と抽出イベント
- `verification.json` / `audit.json`：設定・コード・artifact hashと検証結果

10 checkpoints、同一評価窓・label・B2/B3/P1 score、全group混同行列を検証した。
各モデル16 calibration窓の実推論が保存scoreと一致した。
対象下位分類・判定時点・正負構成・記録分割・配列対応・固定条件推定器の不変性も検証した。
42 unit testsが成功。再実行と途中再開の方法は [修正版プロトコル](../../docs/FALL_COMPARISON_V2.md) を参照。
