# GitHub運用ルール

先生がいつでも進捗を確認し、早くレビューできることを目的とします。

## リポジトリの分担

| リポジトリ | 管理する内容 |
|---|---|
| `2026_aoba-otani_fl-research` | 研究計画、FLモデル、実験設定、評価結果、論文用資料 |
| `heterosense-fl-testbed` | シミュレータ、センサー観測生成、fall-motion、観測統計 |

ローカルの `heterosense-fl-testbed/` は独立したGitリポジトリです。研究本体側ではignoreし、二重登録やsubmodule化をしません。

## 必ず守る流れ

1. 作業開始前に対象リポジトリの `main` を更新する。
2. `feature/<短い名前>` ブランチを作る。
3. 一つの目的ごとに小さくコミットする。
4. 関係するテストを実行し、結果を記録する。
5. feature branchをGitHubへpushする。
6. PRを作り、目的、変更、検証、既知の制約を書く。
7. レビュー後にmainへ取り込む。mainへ直接pushしない。

```powershell
git switch main
git pull --ff-only
git switch -c feature/example

# 編集とテスト
git status
git add <対象ファイル>
git commit -m "変更内容"
git push -u origin feature/example
```

## PRに書く内容

```text
目的:
変更内容:
入力データ／条件:
実行コマンド:
テスト結果:
正式値か暫定値か:
既知の制約:
次の作業:
```

再現性に関わる変更では、使用したbranch、commit ID、Python/NumPy等のversion、seed、設定ファイル、出力先を必ず残します。

## データと成果物

- `*.npy` などの大容量実データはGitへ直接commitしない。
- 元データの場所、生成手順、前処理、分割、checksumをテキストのmanifestとしてcommitする。
- 小さなCSV/JSONの評価集計、図の生成コード、実験設定は原則としてcommitする。
- checkpointや大容量成果物が必要な場合は、保存先を先生と決め、PRから参照する。

## 現在の作業ブランチ

| リポジトリ | ブランチ | 内容 |
|---|---|---|
| `heterosense-fl-testbed` | `feature/fall-motion` | 子RNGによるfall-motion再導入とv1互換テスト |
| `heterosense-fl-testbed` | `feature/condition-estimator` | 観測可能統計と条件推定研究の設計 |
| `2026_aoba-otani_fl-research` | `feature/github-migration` | 全期間の進捗整理とGitHub運用への移行 |
