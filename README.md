# 2D LiDARを用いた連合学習による屋内転倒検知

高齢者見守りを想定し、カメラを使わずに2D LiDARなどの観測から転倒を検知する研究です。施設ごとの観測条件が異なる状況で、データを中央に集めずに学習する連合学習（Federated Learning）を扱います。

## 現在の研究課題

データ補強は誤報を減らせる一方で、条件によってはrecallも低下します。そこで現在は、設置ジオメトリ・遮蔽・センサー可用性を、実際に取得できる観測統計だけから推定し、その条件に応じて転倒判定を切り替える方法を検討しています。

研究質問、これまでの変更、検証済み事項、未完了事項は [研究進捗まとめ](docs/RESEARCH_PROGRESS.md) に記録しています。日々の作業方法は [GitHub運用ルール](docs/GITHUB_WORKFLOW.md) を参照してください。

## リポジトリの役割

- このリポジトリ: 連合学習モデル、実験設定、評価結果、論文用の記録
- [heterosense-fl-testbed](https://github.com/ss-lab-tut/heterosense-fl-testbed): センサーデータ生成、fall-motion多様化、観測統計・条件推定の基盤

ローカルでは両リポジトリを近くに置いていますが、Gitの履歴は別々に管理します。`heterosense-fl-testbed/` をこのリポジトリへ重複登録しません。

## 現在ある実装

- `model.py`: 3クラス分類用CNN
- `data_loader.py`: LiDARデータの読み込みと前処理
- `server.py` / `client.py`: Flowerによる連合学習の最小構成
- `configs/` / `experiments/`: 実験設定と実行用の枠組み

現状の `experiments/run.py` と `client.evaluate()` には仮の評価処理が残っています。このリポジトリだけでは、計画中のB1〜B4、P1、P2の比較実験をまだ再現できません。

## データ

入力は概ね `[N, 2, 64, 430]` 形式のNumPy配列を想定しています。大容量の `*.npy` はGitに登録しません。今後、入手方法、前処理、分割、チェックサムをデータマニフェストとして記録します。

## 最小セットアップ

```powershell
python -m venv fl_env
.\fl_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Flowerのサーバーとクライアントは別々のターミナルで起動します。

```powershell
python server.py
python client.py 0
python client.py 1
```

## 著者

大谷 青羽（豊橋技術科学大学 Smart System Laboratory）
