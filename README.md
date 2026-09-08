# 2D LiDARを用いた連合学習による屋内転倒検知

高齢者見守りを想定し、カメラを使わずに2D LiDARなどの観測から転倒を検知する研究です。施設ごとの観測条件が異なる状況で、データを中央に集めずに学習する連合学習（Federated Learning）を扱います。

## 現在の研究課題

[国際会議に向けた原稿準備](paper/README.md)：既存結果に基づく英語草稿と、CSVから生成した論文用の表を作成しました。

比較実装・5 seeds実行・結果整理の完了状況、実験条件と残る課題は
[B1/B2/B3/P1比較の完了報告（2026-09-08）](docs/TEACHER_REQUEST_STATUS.md)にまとめています。

修正版B1/B2/B3/P1の5 seeds比較が完了しました。
[主表・施設/geometry別CSV・9月9日レビュー資料](results/simulated_fall_p1_v2/REVIEW.md)を参照してください。
P1はB3よりFPRを下げましたが、平均Recall/F1の改善は確認できていません。

2026-09-06に実データが存在しないことを確認しました。投稿準備はシミュレーションの
B1/B2/B3/P1比較を優先します。[修正版比較プロトコル](docs/FALL_COMPARISON_V2.md)を参照してください。
既存ファイル名の`real`は実測データの証拠ではなく、既存配列の出自は未確認です。

従来の配列監査・CNN基準実験・欠損入力比較は
[転倒ラベル評価の記録](docs/REAL_FALL_EVALUATION.md) にまとめています。
元の学習・検証配列には観測フレームの共有があるため、診断実験には
`experiments/run_real_fall_baseline.py` の時間境界を除外する分割を使います。
被験者・施設・元の前処理が未確認なので、正式な独立評価とは扱いません。

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

従来の `experiments/run.py` と `client.evaluate()` には仮の評価処理が残っています。
今回のB1/B2/B3/P1の制御比較は、別の実行入口
`experiments/run_simulated_fall_p1_v2.py`で実装・実行済みです。
これは集中学習による合成データ比較であり、FL比較の完成を意味しません。
固定scoreの再集計と、testbedの保存済みパッチを用いた学習の再現手順は
[再現方法](docs/OPERATING_POINT_REVIEW.md)を参照してください。B4/P2は今回の対象外です。

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
