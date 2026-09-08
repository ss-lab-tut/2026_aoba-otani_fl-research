# 9月9日レビュー補足：threshold動作点と再現方法

2026-09-08。B1/B2/B3/P1の主比較は維持し、同じ固定scoreから
calibration FPR予算を0.01/0.025/0.05/0.10/0.15/0.20にした記述的な感度分析を追加した。
CNNや条件推定器の再学習はしていない。主解析の予算0.05は変更しない。

## 結果の読み方

unseen geometryでの平均値：

| calibration FPR予算 | B3 Recall | P1 Recall | B3 test FPR | P1 test FPR |
|---:|---:|---:|---:|---:|
| 0.010 | 0.017 | 0.017 | 0.002 | 0.001 |
| 0.025 | 0.027 | 0.035 | 0.005 | 0.004 |
| **0.050（主解析）** | **0.055** | **0.054** | **0.015** | **0.010** |
| 0.100 | 0.078 | 0.079 | 0.043 | 0.041 |
| 0.150 | 0.124 | 0.136 | 0.074 | 0.077 |
| 0.200 | 0.165 | 0.167 | 0.109 | 0.109 |

主解析ではP1が誤検知を減らし、Recallはほぼ同じだが平均では小さく低下した。
予算0.025では両指標の平均が良くなる一方、0.15のRecall増加はFPR増加も伴う。
よって「どの動作点でもP1が優れる」「同じ実測FPRでP1のRecallが高い」とは結論しない。
この6点を見て0.025を新たな主解析に選ぶとtestを使った選択になるので、採用しない。
元の主解析・否定的な結果・全動作点を併記する。

同じcalibration予算でもtest FPRは一致しない。今回の表は同じ調整規則で得た動作点の比較であり、
厳密に同じtest FPRにそろえた性能比較ではない。次の実験を決める際はこの区別を維持する。

全体・施設・geometry・5 seedsの標本標準偏差と対応差は
[動作点比較](../results/fall_operating_points/REPORT.md)と同ディレクトリのCSVに保存した。
正例率1/3・判定時点を指定したシミュレーションという元実験の制約を引き継ぐ。

## Gitの保存物だけで再集計する

固定scoreは`results/simulated_fall_p1_v2/frozen_scores.csv.gz`に含まれる。
展開内容のSHA-256が主解析と一致することを実行時に検証する。
この再集計はNumPyだけで実行でき、ローカルの大きな観測cacheやCNN checkpointを必要としない。

```powershell
python experiments/sweep_fall_operating_points.py --output-dir results/local_operating_points_recheck
```

元の主解析0.05と全groupの混同行列・指標の一致、各方式のcalibration制約を検証する。
スコア改変と制約違反を拒否する新規テストを含め、44 unit testsが成功した。

## モデル学習から再実行する場合

固定条件推定器の1.6 KB artifactと来歴は`artifacts/frozen_condition_v2/`に保存した。
以前のローカル結果ディレクトリを参照しない設定は`configs/simulated_fall_p1_v2_reproduce.json`。
元の設定との違いは固定artifactの参照先だけで、artifact内容のhashは同じ。

```powershell
python experiments/run_simulated_fall_p1_v2.py --config configs/simulated_fall_p1_v2_reproduce.json --output-dir results/local_simulated_fall_p1_v2_recheck
```

こちらはPyTorch・NumPy・PyYAMLと隣接する`heterosense-fl-testbed`が必要。
実行環境はPython 3.14.3、NumPy 2.4.3、PyTorch 2.10.0+cpu。
testbedはbase commit `bd5d1380bb6a654a9ceff124537f60f964bae94e`とローカル変更を使用した。
実験時の全runtimeソースのhashは`results/simulated_fall_p1_v2/audit.json`に保存している。
隣接リポジトリは別管理であり、本リポジトリのpushだけでその作業ツリーが更新されるわけではない。
学習の厳密な再現には、testbedのruntime hashも一致させる必要がある。

2026-09-08にこの不足を確認した。既にGitに保存している
`results/condition_threshold_validation.testbed.patch`を上記base commitへ適用すると、
実験時の全23 runtimeファイルのhashが一致することを、一時ディレクトリで復元して検証した。
別リポジトリの未コミット作業を新たに取得する必要はない。
清潔なtestbed checkoutを上記commitに固定し、このパッチを`git apply`してから学習を実行する。
既存の変更済み作業ツリーにパッチを重ねて適用しない。

```powershell
.\fl_env\Scripts\python.exe experiments/verify_testbed_reconstruction.py
```

検証結果は`results/simulated_fall_p1_v2/runtime_reconstruction.json`。
この検証はソース復元の一致であり、別環境での全学習の再実行まで保証したものではない。

## 論文と追加実験を決めるための論点

- 支持される観察：主解析のP1はB3より誤検知を減らした。
- 未支持の主張：P1による安定したRecall/F1改善、B1を超える総合性能、実運用・実測・FLでの有効性。
- 課題：低い絶対Recall、seed間の学習のばらつき、壁遮蔽での失敗条件。
- 追加実験の判断：条件推定器やモデルの種類を増やす前に、既存CNNの学習安定性や調整条件の比較を検討する。

現testから良いseedや動作点を選び直してはいない。新しい確認実験を行う場合は、開発用データで
変更を固定した後、別の評価軌跡を使う。
