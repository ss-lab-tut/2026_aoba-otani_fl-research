"""Verify finished simulation exports and publish small, reviewable tables."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(source, output):
    protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('status') != 'complete_simulation_only':
        raise ValueError('simulation has not completed')
    if sha(source/'scores.csv') != protocol['scores_sha256'] or sha(source/'observations.npz') != protocol['observations_sha256']:
        raise ValueError('generated artifacts changed')
    training = json.loads((source/'training.json').read_text(encoding='utf-8'))
    cfg = protocol['config']
    expected = {(s,m) for s in cfg['seeds'] for m in ('B1','B2')}
    if {(r['seed'],r['method']) for r in training} != expected or len(training) != len(expected):
        raise ValueError('incomplete model set')
    for item in training:
        if sha(source/f'{item["method"]}_seed{item["seed"]}.pt') != item['checkpoint_sha256']:
            raise ValueError('CNN checkpoint changed')
        if len(item['losses']) != cfg['epochs']:
            raise ValueError('incomplete training')
    for seed in cfg['seeds']:
        counts = [len(r['training_windows']) for r in training if r['seed']==seed]
        if counts[0] != counts[1]:
            raise ValueError('unmatched training budgets')
    predictions = read_csv(source/'comparison/predictions.csv')
    metrics = read_csv(source/'comparison/metrics.csv')
    by_method = {}
    for seed in cfg['seeds']:
        for method in ('B1','B2','B3','P1'):
            subset = [r for r in predictions if int(r['seed'])==seed and r['method']==method]
            by_method[seed,method] = {r['sample_id']:r for r in subset}
            if len(subset) != len(by_method[seed,method]):
                raise ValueError('duplicate predictions')
        ref = by_method[seed,'B2']
        for method in ('B1','B3','P1'):
            candidate = by_method[seed,method]
            if set(candidate) != set(ref):
                raise ValueError('unpaired test windows')
            for key, r in ref.items():
                if r['label'] != candidate[key]['label']:
                    raise ValueError('unpaired labels')
                if method != 'B1' and r['score'] != candidate[key]['score']:
                    raise ValueError('B2/B3/P1 must share fall scores')
    for metric in metrics:
        selected = [r for r in by_method[int(metric['seed']),metric['method']].values()
                    if (metric['scope']=='all' or r['scope']==metric['scope'])
                    and (metric['facility']=='__all__' or r['facility']==metric['facility'])
                    and (metric['geometry']=='__all__' or r['geometry']==metric['geometry'])]
        y = np.array([int(r['label']) for r in selected])
        p = np.array([float(r['score']) >= float(r['threshold']) for r in selected])
        if any(int(r['prediction']) != int(v) for r,v in zip(selected,p)):
            raise ValueError('prediction/threshold inconsistency')
        counts = dict(n=len(y),tp=int(np.sum(p & (y==1))),fp=int(np.sum(p & (y==0))),
                      fn=int(np.sum(~p & (y==1))),tn=int(np.sum(~p & (y==0))))
        if any(counts[k] != int(metric[k]) for k in counts):
            raise ValueError('confusion matrix mismatch')
    summary = read_csv(source/'comparison/summary.csv')
    deltas = read_csv(source/'comparison/paired_P1_minus_B3_summary.csv')
    manifest = json.loads((source/'comparison/manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.csv','summary.csv','paired_P1_minus_B3.csv','paired_P1_minus_B3_summary.csv'):
        shutil.copyfile(source/'comparison'/name, output/name)
    def fmt(row, metric):
        if not row[metric+'_mean']:
            return '未定義'
        return f'{float(row[metric+"_mean"]):.3f} ± {float(row[metric+"_std"]):.3f}'
    corrected = cfg.get('target_label') == 'FALL_or_RECOVERED_FALL_at_third_episode_frame'
    description = ('実測データは使用していない。FALL/RECOVERED_FALLの3番目のframeを判定する修正版の制御比較。'
                   if corrected else '実測データは使用していない。既存CNN構造と固定済み条件推定器を接続した初回診断。')
    lines = ['# シミュレーション転倒窓評価：5 seeds', '', description, '',
             '| 評価範囲 | 方法 | Recall | Precision | F1 | AUROC | FPR |',
             '|---|---|---|---|---|---|---|']
    input_audit_path = ROOT/'results/simulated_cnn_input_audit/audit.json'
    input_audit = json.loads(input_audit_path.read_text(encoding='utf-8')) if input_audit_path.exists() else {}
    if input_audit.get('original_artifacts_sha256', {}).get('scores.csv') == protocol['scores_sha256']:
        lines[2:2] = ['**後続監査で、初回実験の学習・調整にシミュレータ下位分類FALLが0件と判明した。**',
                      '本表は広いABNORMAL窓分類の接続診断であり、転倒検知の有効性の根拠にはしない。',
                      '[入力・ラベル・データ監査](../simulated_cnn_input_audit/REPORT.md)を参照。', '']
    for row in summary:
        if row['facility']==row['geometry']=='__all__':
            lines.append('| '+' | '.join([row['scope'],row['method'],*[fmt(row,m) for m in ('recall','precision','f1','auroc','false_positive_rate')]])+' |')
    lines += ['', '## P1−B3：同じseedでの対応差', '',
              '| 範囲 | Recall差 | Precision差 | F1差 | FPR差 |', '|---|---|---|---|---|']
    for row in deltas:
        if row['facility']==row['geometry']=='__all__':
            lines.append('| '+' | '.join([row['scope'],*[fmt(row,m) for m in ('recall','precision','f1','false_positive_rate')]])+' |')
    lines += ['', '## Unseen geometry別のB3/P1', '',
              '| geometry | 方法 | Recall | F1 | FPR |', '|---|---|---|---|---|']
    for row in summary:
        if row['scope']=='unseen' and row['facility']=='__all__' and row['geometry']!='__all__' and row['method'] in ('B3','P1'):
            lines.append('| '+' | '.join([row['geometry'],row['method'],*[fmt(row,m) for m in ('recall','f1','false_positive_rate')]])+' |')
    lines += ['', '## 最低Recallの施設×geometry', '',
              '| 方法 | 施設 | geometry | Recall | FPR |', '|---|---|---|---|---|']
    for method in ('B1','B2','B3','P1'):
        candidates = [r for r in summary if r['method']==method and r['facility']!='__all__'
                      and r['geometry']!='__all__' and r['recall_mean']]
        worst = min(candidates, key=lambda r:float(r['recall_mean']))
        lines.append('| '+' | '.join([method,worst['facility'],worst['geometry'],fmt(worst,'recall'),fmt(worst,'false_positive_rate')])+' |')
    lines += ['', '## 検証と制約', '',
              '- 10 CNN checkpointsのhash、学習量、5 seedsの同一test窓・ラベルを検証。',
              '- B2/B3/P1の転倒score一致と、予測から再計算した全groupの混同行列一致を検証。',
              '- 5 seedsは同じ生成データを共有するモデルseed。平均±標本標準偏差は軌跡母集団の信頼区間ではない。',
              ('- 64-frameの過去文脈からFALL/RECOVERED_FALLの3番目のframeを分類。NEAR_FALLとADLは負例。'
               if corrected else '- 64-frame窓内のABNORMAL有無を分類。窓の重なり・geometry間の軌跡共有があり、独立イベント評価ではない。'),
              ('- 条件推定器は判定時点の直近5-frame平均。対象時点は一致させたが、CNNの文脈長は64 frames。'
               if corrected else '- 条件推定器は5-frame平均の窓末端scoreを使う。転倒対象窓全体と条件推定の時間範囲は異なる。'),
              '- 合成3座標点群・2.5-D遮蔽・集中学習。実測2D LiDARやFLの有効性はこの比較で主張しない。',
              '- 過去の条件推定評価で使ったgeometryを再利用している。完全な未見の研究確認実験ではない。',
              '- この結果を見てから同じtestに合わせてthresholdやモデルを選び直さない。', '',
              'thresholdとfallbackの詳細は`audit.json`。完全な生成観測・checkpoint・予測はローカル実験ディレクトリに保存。']
    if corrected:
        lines += ['', '修正版は正例率1/3に固定した抽出比較。Precision/F1はこの構成に依存し、自然頻度の値ではない。',
                  '生成器のABNORMAL持続確率は0.2から0.8に固定変更。実時間への妥当性、event recall、検知遅延は未検証。',
                  '学習前イベント数検査はローカルの`event_preflight.json`に保存。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit = dict(status='verified_simulation_only', local_artifacts=str(source.resolve()),
                 protocol=protocol, thresholds=manifest['thresholds'],
                 verified_models=len(training), verified_metric_rows=len(metrics),
                 source_reporter_sha256=sha(Path(__file__)),
                 exported_sha256={n:sha(output/n) for n in ('metrics.csv','summary.csv','paired_P1_minus_B3.csv','paired_P1_minus_B3_summary.csv')})
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(lines[:21]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/local_simulated_fall_p1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/simulated_fall_p1')
    args = parser.parse_args()
    report(args.source,args.output_dir)
