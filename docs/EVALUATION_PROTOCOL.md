# 条件別評価プロトコル

## 目的

B1〜B4、P1、P2を同じ定義で比較するため、モデル学習と評価処理を分離します。各モデルは評価データに対するfall scoreをCSVへ出力し、共通スクリプトで閾値曲線を計算します。

## 入力形式

CSVには次の3列が必要です。

```csv
label,score,condition
1,0.91,WALL_OCCLUDED_HELD_OUT
0,0.37,CORNER_LIDAR_ONLY
```

- `label`: 転倒を1、通常行動を0とした正解ラベル
- `score`: 転倒クラスの確率または単調な判定score
- `condition`: 固定した評価条件名

最低限、先生から指定された次の条件を区別します。

- `WALL_OCCLUDED_HELD_OUT`
- `CORNER_LIDAR_ONLY`
- `HARSH_FACILITY`
- `NORMAL_UNDER_OCCLUSION`
- `CEILING_FOV_EDGE`

## 実行例

```powershell
python experiments/evaluate_predictions.py predictions.csv `
  --model-id B1 `
  --seed 0 `
  --window-seconds 6.4 `
  --output-dir results/b1_seed0
```

出力は次の3ファイルです。

- `tradeoff.csv`: 条件・閾値ごとのrecall、false alerts、混同行列
- `worst_condition.csv`: 閾値ごとの最悪条件recall
- `evaluation_manifest.json`: model ID、seed、入力、false alert単位、天井結果の暫定フラグ

## false alertsの暫定定義

現在は「正解が通常行動であるwindowを転倒と判定した数」と定義します。`false_alert_rate` は負例windowに対する割合です。window秒数を渡した場合は `false_alerts_per_hour` も計算します。

連続windowの誤報を1件にまとめるevent-level指標はまだ実装していません。実験開始前に、論文の主指標をwindow単位、event単位、時間単位のどれにするか先生と確定する必要があります。

## 5 seed集計

`src.evaluation.aggregate_seed_curves()` は同じ条件・閾値について5 seedが揃っていることを検査し、平均と標本標準偏差（分母 `n-1`）を返します。seed不足や同一seedの重複はエラーにし、部分的な結果を正式値として集計しません。

## 天井条件

B4および `CEILING_FOV_EDGE` を含む天井視点結果は、実LiDAR点群で妥当性を確認するまで暫定です。B4のmanifestには暫定フラグを自動的に設定します。
