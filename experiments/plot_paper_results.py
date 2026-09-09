"""Create descriptive paper figures from frozen comparison and fit diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    source=ROOT/'results/simulated_fall_p1_v2'
    audit=json.loads((source/'audit.json').read_text())
    for name in ('metrics.csv','summary.csv'):
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=audit['exported_sha256'][name]:
            raise ValueError('Published comparison changed')
    rows=read(source/'metrics.csv')
    output=ROOT/'paper/figures'
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for i,scope in enumerate(('all','unseen')):
        for j,metric in enumerate(('recall','f1','false_positive_rate')):
            ax=axes[i,j]
            for seed in range(5):
                values=[]
                for method in ('B3','P1'):
                    r=[r for r in rows if r['seed']==str(seed) and r['method']==method and r['scope']==scope
                       and r['facility']==r['geometry']=='__all__']
                    if len(r)!=1: raise ValueError('Unpaired seed')
                    values.append(float(r[0][metric]))
                ax.plot([0,1],values,'o-',label=f'Seed {seed}',alpha=.8)
            ax.set_xticks([0,1],['B3','P1']); ax.set_xlim(-.2,1.2); ax.set_ylim(bottom=0)
            ax.set_title(f'{scope.capitalize()}: '+{'recall':'Recall','f1':'F1','false_positive_rate':'FPR'}[metric])
            ax.grid(axis='y',alpha=.2)
    axes[0,2].legend(fontsize=8)
    fig.suptitle('Paired model seeds at calibration FPR budget 0.05')
    save(fig,'paired_policies')
    summary=read(source/'summary.csv')
    groups=sorted({(r['facility'],r['geometry']) for r in summary if r['scope']=='unseen'
                   and r['facility']!='__all__' and r['geometry']!='__all__'})
    data=[]
    for facility,geometry in groups:
        values=[]
        for method in ('B3','P1'):
            r=[r for r in summary if r['scope']=='unseen' and r['facility']==facility and r['geometry']==geometry and r['method']==method]
            if len(r)!=1: raise ValueError('Missing facility/geometry')
            values.append(float(r[0]['recall_mean']))
        data.append(values)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    im=ax.imshow(data,vmin=0,vmax=max(.1,np.max(data)),cmap='Blues',aspect='auto')
    ax.set_xticks([0,1],['B3','P1']);ax.set_yticks(range(len(groups)),[f'{f} / {g}' for f,g in groups])
    for i,pair in enumerate(data):
        for j,value in enumerate(pair):
            ax.text(j,i,f'{value:.3f}',ha='center',va='center',color='white' if value>np.max(data)*.6 else 'black')
    fig.colorbar(im,ax=ax,label='Mean recall');ax.set_title('Unseen geometry: mean recall over five model seeds')
    save(fig,'unseen_recall')
    fit=read(ROOT/'results/cnn_fit_v2/metrics.csv')
    scopes=('train_actual','test_clear','test_seen','test_unseen')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,method in zip(axes,('B1','B2')):
        for seed in range(5):
            values=[]
            for scope in scopes:
                r=[r for r in fit if r['method']==method and r['seed']==str(seed) and r['scope']==scope]
                if len(r)!=1: raise ValueError('Missing fit diagnosis')
                values.append(float(r[0]['auroc']))
            ax.plot(range(4),values,'o-',alpha=.8,label=f'Seed {seed}')
        ax.axhline(.5,ls='--',color='gray',lw=1)
        ax.set_xticks(range(4),['Train*','Test clear','Test seen','Test unseen'],rotation=15)
        ax.set_ylim(0,1);ax.set_ylabel('Raw-score AUROC');ax.set_title(method);ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen CNN fit (*actual training windows; mixtures differ across groups)')
    save(fig,'cnn_fit')
    print('Created three PNG/SVG figure pairs')


if __name__=='__main__':
    main()
