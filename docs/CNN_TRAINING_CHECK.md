# 既存CNNの学習時間に関する追加検証

2026-09-09、実行前に設定を固定。

## 検証する問い

B3/P1が共通で使うB2の既存CNNについて、6 epochsから18 epochsへ学習を延ばすと、
学習用記録内で分離した開発記録の識別性能が改善するかを確認する。
モデル構造、入力、窓ラベル、augmentation、optimizer、条件推定器は変更しない。
既存のcalibration/testを設定選択や追加検証の評価に使用しない。

## 固定する条件

- 元実験のtrain生成seed 10001/10002/10003をfit、10004をdevelopmentとする。各2施設。
- recording単位で分離する。両集合で同じイベントやgeometry複製が交差しない。
- fitはclear窓ごとに既知3 geometryから1つを固定乱数で選択するB2方式。
- developmentは既知3 geometryの全窓を評価する。unseenは評価しない。
- model seedsは0〜4。学習率0.001、batch 16、Adam、既存LidarCNN、18 epochs。
- 1/3/6/12/18 epochs時点でfitとdevelopmentのAUROCと3-class cross entropyを記録。
- 主比較は事前固定の18 minus 6 epochsの対応差。途中epochから最良値を選ばない。
- 6/18 epochsのcheckpointを別のローカル出力先へ保存。主実験を上書きしない。

## 解釈と次の段階

学習だけ改善しdevelopmentが改善しない場合、学習時間を延ばすだけの対応は支持しない。
developmentが改善しても、この開発記録だけで改善の一般性やB3/P1の優位性は主張しない。
各seedの差・平均・標本標準偏差を併記し、統計的有意差は主張しない。
学習記録数が主実験より減るため、主実験の6 epochs結果との直接比較はしない。

この検証はB2の学習時間を切り分ける開発実験であり、B1〜P1の新しい最終比較ではない。
設定変更を採用する場合は、開発結果を確認して固定した後、新たな評価軌跡で全方式を再比較する。
本検証から自動的に18 epochsを主解析へ採用することはしない。
