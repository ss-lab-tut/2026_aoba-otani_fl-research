# 新しい評価軌跡での追加比較（2026-09-10）

実行前に固定した追加実験。主解析の6 epochs結果は維持する。

## 目的

開発検証では18 epochsでAUROCが改善する一方、cross entropyが悪化した。
次は18 epochsのB1/B2/B3/P1を新しいcalibration/testで比較し、
低FPRでの転倒Recall/F1とB3対P1の差を確認する。
6 epochsの既存checkpointも同じ新規データで評価し、学習時間と評価軌跡の変更を区別する。

## 固定条件

- trainは元実験と同じ10001〜10004、2施設。
- 新しいcalibration生成seedは13001/13002、testは14001/14002。
- 18 epochs、model seeds 0〜4、B1/B2計10モデル。既存CNN・Adam・batch・学習率は同じ。
- 元実験の6 epochs checkpointsを追加学習せず新しい同一calibration/testで再評価する。
- 条件推定器・入力・ラベル・quota・known/unseen geometry・FPR予算0.05は変更しない。
- B3/P1は各学習時間内で同じB2 scoreを使用し、thresholdは新しいcalibrationのみで決める。
- 指標は全体・施設・geometry・seen/unseen別のRecall/Precision/F1/AUROC/FPRと対応差。
- 学習時間を比較する際は同じ新しい評価窓とseedを対応させる。
- testを見てepochやthreshold予算を変更しない。改善しなくても全結果を保存する。

新しい軌跡であってもgeometryは過去に調べたものを含む。実測・連続監視・FLの実証ではない。
18 epochsは開発結果に基づく候補条件であり、元主解析に置き換えたり最適設定と呼んだりしない。
