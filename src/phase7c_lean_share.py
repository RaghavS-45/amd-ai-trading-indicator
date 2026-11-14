"""
phase7c_lean_share.py  –  how often should the direction arrow show? Compares a daily-reset cut
that shows the arrow on 10%, 20% or 30% of signals (cut = that quantile of |p_up - p_down| over the
previous 20 sessions, no outcomes) on the E5.6 walk-forward predictions (training data only).
Scored like the live scorecard: IEX closes, entry at the signal bar's close, one trade at a time.
Usage: python src/phase7c_lean_share.py
"""

import numpy as np, pandas as pd
from tradable import COST, non_overlapping
from evaluate import clustered_se
W=20
d=pd.read_parquet('report/phase5_E5.6_oos.parquet').sort_index()
d['edge']=d['p_up']-d['p_down']; d['a']=d['edge'].abs()
days=sorted(d['date'].unique()); by={k:g['a'].to_numpy() for k,g in d.groupby('date')}
def cuts(q):
    return {day: np.quantile(np.concatenate([by[x] for x in days[i-W:i]]), q) for i,day in enumerate(days) if i>=W}
def score(x, sig):
    ret=x['fwd_ret'].to_numpy(); on=sig!=0
    hit=(np.sign(ret[on])==sig[on]).astype(float)
    rows=non_overlapping(x.index, sig)
    net=(sig[rows]*ret[rows]-COST/x['close'].to_numpy()[rows])*1e4
    nd=x['date'].nunique()
    return dict(share=on.mean(), per_day=on.sum()/nd, acc=hit.mean(), se=clustered_se(hit,x['date'].to_numpy()[on]),
                trades=len(rows), bps=net.mean(), bps_total=net.sum(), base=(ret>0).mean())
out=[]
for name,q in [('adaptive 10%',0.90),('adaptive 20%',0.80),('adaptive 30%',0.70)]:
    c=cuts(q); x=d[d['date'].isin(c)]; cut=x['date'].map(c).to_numpy()
    sig=np.where(x['edge']>=cut,1,np.where(x['edge']<=-cut,-1,0))
    for lab,m in [('all',np.ones(len(x),bool)),('folds 6-9',(x['fold']>=6).to_numpy())]:
        out.append(dict(rule=name,rows=lab,**score(x[m],sig[m])))
    pf=[score(x[x['fold']==f],sig[(x['fold']==f).to_numpy()]) for f in sorted(x['fold'].unique())]
    print(name,'per fold acc',[round(p['acc']*100,1) for p in pf],'bps',[round(p['bps'],2) for p in pf])
    dd=pd.DataFrame({'date':x['date'],'on':sig!=0}).groupby('date')['on'].mean()
    print('   daily fire share: 5th %.3f median %.3f 95th %.3f'%tuple(dd.quantile([.05,.5,.95])))
print(pd.DataFrame(out).round(4).to_string())
