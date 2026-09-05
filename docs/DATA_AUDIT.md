# データ監査結果

## 監査の目的

B1以降の数値を再現可能にするため、現在ローカルにある学習・検証データについて、shape、dtype、ラベル分布、ファイル同一性、ラベル間の整合性を確認しました。元の `.npy` ファイルは変更していません。

## データ構成

| ファイル | shape | dtype | ラベル分布 |
|---|---:|---|---|
| `X_train.npy` | `[955, 64, 430, 2]` | float64 | — |
| `X_val_real.npy` | `[239, 64, 430, 2]` | float64 | — |
| `y_train_bin.npy` | `[955]` | int64 | 0: 657、1: 298 |
| `y_train_sub.npy` | `[955]` | int64 | 0: 276、1: 277、2: 104、3: 298 |
| `y_val_bin.npy` | `[239]` | int64 | 0: 164、1: 75 |
| `y_val_sub.npy` | `[239]` | int64 | 0: 69、1: 68、2: 28、3: 74 |

全ファイルについて、特徴量とラベルの行数は一致しています。

## 発見したラベル不整合

既存コードの定義に従い、`binary_label == (subclass_label == 3)` を期待して比較しました。

- 学習データ: 不整合なし
- 検証データ: 1件の不整合
- 該当index: `111`
- `y_val_bin[111] = 1`
- `y_val_sub[111] = 0`

この1件をどちらのラベルへ合わせるべきかは、元データの作成根拠を確認しなければ決められません。正式評価ではデータ作成者と正解ラベルを確認し、修正版データを新しいchecksumで固定します。それまではB1/B2の正式値を確定しません。

## SHA-256

| ファイル | SHA-256 |
|---|---|
| `X_train.npy` | `991455581dccf60cebb21386c7d6a2f7deb2e04d9aa519560e206652855225ca` |
| `X_val_real.npy` | `9fbdf9ac2265c444e2e1755c0a23964fa177e03e4dda6da9360dc82ed9a1e4e6` |
| `y_train_bin.npy` | `91f413664a7753caefd38e9e2e5bc3fa4532fec7a21c30755bd5737e60443ca2` |
| `y_train_sub.npy` | `c49d215af09ddfe4d5d2944e1537ff589cab5fec41964cfe1fb3dc2793f96219` |
| `y_val_bin.npy` | `b6287c074656c0e9bbbec91c5bf8823567ae7d940108c68f34b55ecdc152783e` |
| `y_val_sub.npy` | `89fd1075cbeda2b4f30f1ae582e614e63dfbf9d5449ce9a79973c4b48ea8d562` |

## 再実行方法

```powershell
python experiments/audit_dataset.py --data-dir . --output results/data_audit.json
```

`--strict` を付けると、不整合が存在する間は終了コード1になります。CIや正式実験の前処理ではstrict modeを使い、ラベル不整合を見逃さないようにします。

## 現在不足している情報

現在の配列には施設名、設置条件、遮蔽条件を対応付けるmetadataがありません。そのため、既存の検証データだけでは計画中の条件別評価へ分割できません。最低限、各sampleに次のいずれかを対応付ける必要があります。

- `WALL_OCCLUDED_HELD_OUT`
- `CORNER_LIDAR_ONLY`
- `HARSH_FACILITY`
- `NORMAL_UNDER_OCCLUSION`
- `CEILING_FOV_EDGE`
