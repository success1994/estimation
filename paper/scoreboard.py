"""Win/tie/loss of Ours (LMDPDE with data-driven gamma) vs MLE and Logit-QR, n=300, paired bootstrap on replications."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cols=['setting','n','scenario','rep','estimator','slope_squared_error']
a=pd.read_csv(ROOT/'results_full/sim_fits.csv',usecols=cols);a['tau']=.5
b=pd.read_csv(ROOT/'results_tau/sim_tau.csv',usecols=cols+['tau'])
d=pd.concat([a,b]);d=d[(d.n==300)&d.estimator.isin(['MLE','LMDPDE_auto','LogitQR'])]
rng=np.random.default_rng(1);rows=[]
for (t,s,sc),g in d.groupby(['tau','setting','scenario']):
    w=g.pivot(index='rep',columns='estimator',values='slope_squared_error').to_numpy()
    cols_=list(g.pivot(index='rep',columns='estimator',values='slope_squared_error').columns)
    ours=w[:,cols_.index('LMDPDE_auto')];idx=rng.integers(0,len(ours),(2000,len(ours)))
    r={'tau':t,'setting':s,'scenario':sc,'mse_ours':ours.mean()}
    for c in ('MLE','LogitQR'):
        x=w[:,cols_.index(c)];ratio=x[idx].mean(1)/ours[idx].mean(1);lo,hi=np.percentile(ratio,[2.5,97.5])
        r[f'mse_{c}']=x.mean();r[f'ratio_{c}']=x.mean()/ours.mean();r[f'lo_{c}']=lo;r[f'hi_{c}']=hi
        r[f'res_{c}']='win' if lo>1 else ('loss' if hi<1 else 'tie')
    rows.append(r)
R=pd.DataFrame(rows);R.to_csv(ROOT/'paper/tables/scoreboard_cells.csv',index=False)
grp={'C0':'Clean data','C1':'Gross outliers','C2':'Gross outliers','C3':'Bad leverage','C4':'Mild outliers','C5':'Mild outliers'}
R['group']=R.scenario.map(grp)
for c in ('MLE','LogitQR'):
    print(c);print(R.groupby('group')[f'res_{c}'].value_counts().unstack(fill_value=0))
