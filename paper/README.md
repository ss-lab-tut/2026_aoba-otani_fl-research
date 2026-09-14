# 国際会議投稿に向けた原稿準備

2026-09-14更新。比較実装・5 seeds実行・結果整理に続き、既存結果を英語原稿へまとめた。
投稿先・締切・ページ制限は未確認のため、現段階ではMarkdownによる内容レビュー用の草稿である。

## 作成したもの

- [統合した論文用表](RESULTS_TABLES.md)と[LaTeX表](results_tables.tex)：初期結果・新規軌跡の6/18 epochs・施設別対応差を併記。監査済みCSVから生成。
- [研究概要・レビュー用要点](REVIEW_BRIEF.md)：主題、3段階の結果、施設ごとの利点と不利益、投稿前の判断事項。
- [最新P1/B3の全16条件](../results/fresh_trajectory_comparison/SUBGROUPS.md)：改善・悪化・seed別の方向を全条件で確認。
- [Git内のscoreだけによる再集計検証](../results/fresh_trajectory_comparison/REPLAY_VERIFICATION.md)：6/18 epochsの指標・対応差を再現。
- [新規軌跡での6/18 epochs比較](../docs/FRESH_TRAJECTORY_RESULTS.md)：同一データ・5 seedsで全方式を再比較。2026-09-14検証完了。
- [関連研究メモ](LITERATURE_NOTES.md)と[参考文献](references.bib)：一次文献3件の位置付けを草稿へ追加。網羅性の確認は未完了。
- [学習時間の追加検証](../docs/CNN_TRAINING_CHECK_RESULTS.md)：開発分割・5 seedsで6対18 epochsを比較。AUROC改善とcross entropy悪化の両方を記録。
- [論文用の3図とCNN診断](figures/README.md)：B3/P1対応比較、施設×unseen Recall、学習・評価AUROC。2026-09-09追加。
- [英語原稿](DRAFT.md)：仮題、Abstract、Introduction、方法、実験設定、結果、Discussion、Conclusion。
- [論文用の表](TABLES.md)：全体・seen・unseenの4方式比較とP1−B3の対応差。検証済みCSVから生成。
- [比較の詳細報告](../docs/TEACHER_REQUEST_STATUS.md)：入力とラベルの修正経緯、設定、結果、再現方法。

原稿の中心は、固定条件推定器を既存CNNへ接続したときのthreshold選択の効果である。
初期6 epochsではRecall/F1改善が未確認、追加18 epochsではP1がB3より平均Recall/F1を高めFPRを下げた、という両結果を併記した。
施設別には利点と不利益が異なり、一様な改善や全方式への優越は示していない。
新モデルや推定器の追加改善、testに合わせた条件変更は行っていない。

## 主張と根拠

| 原稿の主張 | 根拠 | 適用範囲 |
|---|---|---|
| 4方式を同じ評価窓・5 model seedsで比較した | 設定、監査、620 group行の検証 | 固定軌跡でのモデル乱数の反復 |
| B3/P1は同一fall scoreでthresholdだけを変える | 比較実装、保存score、checkpoint検証 | P1は推定2状態による切替 |
| 主解析のP1はB3よりFPRが低い | 全体・unseenの平均、対応差CSV | calibration予算0.05、合成データ |
| 初期6 epochsでは平均Recall/F1改善は確認できなかった | 初期の全体・unseenの4方式表 | 初期結果を保持 |
| 追加18 epochsではP1がB3より平均Recall/F1を高めFPRを下げた | 新規軌跡の対応差と全16条件表 | 施設ごとの一様な改善や全方式への優越ではない |
| 未学習geometryで評価した | train/calibration/推定器学習geometryの検査 | 過去研究で調べたgeometryであり、研究全体の完全な初見ではない |

性能改善の主張は18 epochs・新規合成軌跡のP1対B3の集計結果に限定する。
実測2D LiDAR、連続監視、FL、臨床的有効性は実証していない。
低Recallの原因はまだ切り分けられておらず、Discussionの候補要因は仮説として記載した。

## 投稿までの残作業

| 項目 | 現状 | 次の作業 |
|---|---|---|
| 投稿先・締切・ページ数 | 未確認 | 会議名が判明後に公式募集要項を確認 |
| 論文の主張 | 結果に基づく草稿を作成 | レビューで採用する主張と追加実験の必要性を判断 |
| 関連研究・参考文献 | 一次文献3件を草稿へ追加 | 投稿領域に合わせ、近接研究と網羅性を確認 |
| 実験・結果本文 | 草稿作成済み | 内容レビュー、表の配置、必要な施設/geometry別図表の選択 |
| 著者・所属・謝辞 | 未確定 | 共著者情報、匿名査読要件などを確認 |
| テンプレート・ページ調整 | 未実施 | 投稿先確定後にLaTeX等へ移植・組版確認 |
| 最終提出 | 未実施 | 原稿レビューと提出内容確定後に実施 |

原稿の体裁を整える前に、低い絶対Recallの下でどの研究主張が妥当かをレビューする。
現段階の草稿は投稿可能性や採択を保証するものではない。

## 表の再生成

```powershell
.\fl_env\Scripts\python.exe experiments/export_paper_tables.py
```

`summary.csv`と`paired_P1_minus_B3_summary.csv`を監査記録のSHA-256と照合し、
4方式・5 seedsの指標がそろっていることを確認して`TABLES.md`を生成する。
原稿の数値はこの表と整合させる。今回の作業では学習・主解析を変更していない。

統合表を再生成・照合する場合：

```powershell
.\fl_env\Scripts\python.exe experiments/build_manuscript_results.py
.\fl_env\Scripts\python.exe experiments/build_manuscript_results.py --check
```

監査済みの元CSVを確認し、統合Markdown/LaTeX表と英語草稿の主要な数値記述9箇所を照合する。
これは原稿全体の自動査読ではない。確認対象はB2の学習時間差、18 epochsのP1対B3の指標・対応差、施設別Recall/FPR差。
LaTeXは表の移植用素材として作成しており、投稿先テンプレートでの組版は未確認。
