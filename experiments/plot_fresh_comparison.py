"""Plot all policies at both fixed training durations on the new test set."""
import csv
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'fall-paper-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/fresh_trajectory_comparison'
    tables={}
    for n in (6,18):
        with (source/f'epochs{n}'/'summary.csv').open(newline='') as f:tables[n]=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    methods=('B1','B2','B3','P1')
    for i,scope in enumerate(('all','unseen')):
        for j,(metric,title) in enumerate([('recall','Recall'),('f1','F1'),('false_positive_rate','FPR')]):
            ax=axes[i,j]
            for n,offset,color in [(6,-.12,'#2476b4'),(18,.12,'#d76d20')]:
                selected=[]
                for method in methods:
                    rows=[r for r in tables[n] if r['scope']==scope and r['method']==method and r['facility']==r['geometry']=='__all__']
                    if len(rows)!=1:raise ValueError('Missing policy group')
                    selected.append(rows[0])
                ax.errorbar([v+offset for v in range(4)],[float(r[metric+'_mean']) for r in selected],
                            yerr=[float(r[metric+'_std']) for r in selected],fmt='o',capsize=3,color=color,label=f'{n} epochs')
            ax.set_xticks(range(4),methods);ax.set_title(f'{scope.capitalize()}: {title}')
            ax.set_ylim(bottom=min(0,ax.get_ylim()[0]));ax.axhline(0,color='gray',lw=.6);ax.grid(axis='y',alpha=.2)
    axes[0,2].legend()
    fig.suptitle('Same fresh test trajectories; five seeds (mean ± sample SD)')
    for extension in ('png','svg'):fig.savefig(source/f'comparison.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Fresh comparison PNG/SVG saved')


if __name__=='__main__':main()
