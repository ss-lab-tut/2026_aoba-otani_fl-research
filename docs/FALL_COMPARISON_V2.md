# B1/B2/B3/P1の修正版比較プロトコル

2026-09-07：5 seedsの学習・比較・検証が完了した。
[9月9日レビュー用まとめ](../results/simulated_fall_p1_v2/REVIEW.md)に主表と判断材料を保存した。
P1はB3よりFPRを減らしたが、平均Recall/F1の改善は確認できなかった。

目的はB1/B2/B3/P1の同条件比較、特にB3対P1とunseen geometryの結果を5 seedsでまとめ、
論文のストーリーと必要な追加実験を判断できる状態にすること。
点検や新手法の探索を独立した目的にせず、初回比較で見つかった不整合を直して再実行する。
条件推定器の改善、P2、新CNN、GAN、追加のモデル選択は行わない。

## 今回固定する修正

1. 正例はシミュレータの`FALL`または`RECOVERED_FALL`。
   そのエピソードの3番目のframeを判定時点とする。`NEAR_FALL`は負例に区別する。
   ADL負例は末端5 framesにABNORMALがない定間隔候補から抽出する。
   NEAR_FALL負例はエピソード終了の次の次のframe（正常へ戻っていることを検査）から抽出する。
2. 既存BehaviorModelのABNORMAL行だけを変更し、自己遷移確率を0.2→0.8に固定する。
   他の状態への遷移確率は合計0.2となるよう同比率で縮小し、他の行・観測モデルは変更しない。
   対象イベントを生成できるようにするためのシミュレーション条件変更であり、新しい検知モデルではない。
3. 学習は2施設×4生成seedの8記録、調整と評価はそれぞれ2施設×2生成seedの4記録。
   各記録は6000 steps。以前の学習・調整・評価と異なる生成seedを用いる。
   抽出はラベル・イベント数と固定乱数だけで決め、観測scoreや検知性能を見て選ばない。
4. 施設ごとに学習はFALL系48、NEAR_FALL24、ADL72の144窓、調整と評価は各24/12/36の72窓。
   複数の元記録から交互に選び、同じendpointの重複を除外する。
   strict FALLを施設・分割ごとに4件以上含める。必要な候補が不足したら学習前に停止する。
5. CNNの入力は判定時点までの64 frames、条件推定は同じ時点までの5 frames。
   元のラベルと観測時刻・event ID・下位分類・記録IDをCSVに残す。
   step間隔は従来どおり1秒と明記し、6.4秒の計測データには見せない。

## 据え置くもの

- `model.LidarCNN`の構造と3クラス出力。非転倒を0、転倒を2として学習。
- 距離・高さの430方位bin、固定8 m/2.5 m正規化。
- 過去に固定した条件推定器のartifactをhash検証してそのままコピーし、再学習しない。
- B1/B2は同じ窓・ラベル・初期化・6 epochs・学習率・ミニバッチ順。
  B2だけ既知geometryへ窓を置換する。B2/B3/P1は同じCNN checkpointとfall scoreを使う。
- B1/B2の共通threshold、B3の施設別threshold、P1の推定状態別thresholdを、
  calibrationだけでFPR予算0.05に合わせる。testを見て再調整しない。
- 2施設、既知3 geometry、unseen5 geometry。geometryは条件推定器学習からも除外されたものを使う。

## 実行

```powershell
.\fl_env\Scripts\python.exe -u experiments/run_simulated_fall_p1_v2.py
```

生成とイベント数検査の後、5 model seedsを順に実行する。
準備完了後に実行が切れた場合は、同じ設定・コード・観測hashを検証して再開できる。
完成したseedを再利用し、未完了seedはそのseedから再学習する。

```powershell
.\fl_env\Scripts\python.exe -u experiments/run_simulated_fall_p1_v2.py --resume
.\fl_env\Scripts\python.exe experiments/report_simulated_fall_p1.py --source results/local_simulated_fall_p1_v2 --output-dir results/simulated_fall_p1_v2
.\fl_env\Scripts\python.exe experiments/verify_simulated_fall_v2.py
```

詳細artifactは`results/local_simulated_fall_p1_v2/`。
レビュー用の表は`results/simulated_fall_p1_v2/`へ保存する。
施設・geometry・seen/unseen別のRecall/Precision/F1/AUROC/FPR、混同行列、5 seeds平均±標本標準偏差、
P1−B3の対応差を出力する。

## 結果の解釈

この比較はFALL系の判定時点を指定した制御比較で、自然発生頻度の連続監視実験ではない。
正例率は1/3であり、Precision/F1を実運用の値へ外挿しない。
特定時点のRecallをevent recallや検知遅延と呼ばない。
既存CNNの集中学習比較であり、FLの有効性、実測2D LiDARの性能は主張しない。
シミュレータの持続確率変更と点群生成条件は論文に開示する。

9月9日のレビューでは、B3/P1主表とunseen geometry、B1/B2との比較、
seedごとのばらつきと失敗条件をまとめて提示する。差が出ない場合もそのまま報告する。
