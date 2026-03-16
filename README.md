Markdown
# 2D LiDARを用いた連合学習による屋内転倒検知 (Federated Learning for Fall Detection)

## 1. プロジェクト概要
本研究は、高齢者の見守り支援等を目的とした「屋内転倒検知」を、プライバシー保護とデータセキュリティの両立を図りながら実現するプロジェクトです。

- **2D LiDARの採用**: カメラ画像を使用しないため、プライバシーを侵害せずに夜間や浴室等での検知が可能です。
- **連合学習 (Federated Learning)**: データを一箇所に集めず、各端末（クライアント）で学習を行うことで、個人情報の流出リスクを低減します。

## 2. ディレクトリ構成
研究室の標準構造に基づき、機能ごとにディレクトリを分離しています。

```text
.
├── src/                # メインソースコード
│   ├── model.py        # 3クラス分類用CNNモデル定義
│   └── data_loader.py  # LiDARデータのロード・前処理スクリプト
├── configs/            # 実験パラメータ設定（学習率、エポック数等）
├── experiments/        # 実験実行用スクリプト
├── server.py           # Flower サーバー側起動スクリプト
├── client.py           # Flower クライアント側起動スクリプト
├── README.md           # 本ドキュメント
└── requirements.txt    # 実行に必要なライブラリ一覧
3. 使用データ詳細
研究室で取得・前処理済みのリアルデータを使用しています。
※データ本体（.npy）はサイズの関係上 Git 管理から除外しています。

サンプリング: 430ビーム, 約10Hz

ウィンドウサイズ: 約6.4秒（64フレーム）単位で切り出し

入力形状: [N, 2, 64, 430] (サンプル数, チャンネル, 時間, ビーム)

チャンネル1: 距離データ（Robust Scaling済み）

チャンネル2: Δ距離データ（フレーム間差分）

ラベル (3クラス分類):

0: sit (座位)

1: walk (歩行・停止・歩行後の着座を含む)

2: fall (転倒)

4. 環境構築 (Setup)
Windows環境、VS Code + PowerShell での実行を想定しています。

仮想環境の作成と起動

Bash
python -m venv fl_env
.\fl_env\Scripts\activate
依存ライブラリのインストール

Bash
pip install -r requirements.txt
5. 実験実行手順 (How to Run)
連合学習のシミュレーションを行うため、ターミナルを3つ分割して起動してください。

Step 1: サーバーの起動 (Terminal 1)
Bash
python server.py
Step 2: クライアント0の起動 (Terminal 2)
Bash
python client.py 0
Step 3: クライアント1の起動 (Terminal 3)
Bash
python client.py 1
※3台目が接続された瞬間に、自動的に連合学習（2ラウンド）が開始されます。

6. 今後の課題
Time-C GAN を用いたデータ拡張による精度の向上検証。

検証用データ (X_val_real.npy) を用いた、各ラウンド終了後のサーバー側でのグローバル評価機能の実装。

Author: 大谷 青羽 (Aoba Otani)

Affiliation: 豊橋技術科学大学 スマートシステム研究室 (Smart System Laboratory)