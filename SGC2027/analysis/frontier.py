import pandas as pd, numpy as np
h=pd.read_csv('/home/claude/w/data/heldout_all_run_metrics.csv',low_memory=False)
h['method']=np.where(h.setting_kind=='PRISM','PRISM_'+h.variant.astype(str),h.scheduler)
def env(B,T):
    ok=B[B.transmitted_kbits<=T]
    if len(ok): return ok.critical_coverage.max()
    return B[B.transmitted_kbits==B.transmitted_kbits.min()].critical_coverage.max()
rows=[]
for (ds,xai,eps,seed),d in h.groupby(['dataset','xai_method','epsilon_budget','seed']):
    B=d[d.setting_kind=='BASELINE'][['scheduler','baseline_duty_cycle','transmitted_kbits','critical_coverage']]
    for (m,s),p in d[d.setting_kind=='PRISM'].groupby(['method','setting']):
        r=p.iloc[0]; e=env(B,r.transmitted_kbits)
        rows.append(dict(dataset=ds,xai=xai,eps=eps,seed=seed,method=m,setting=s,kbit=r.transmitted_kbits,cov=r.critical_coverage,env=e,margin=r.critical_coverage-e))
R=pd.DataFrame(rows); R.to_csv('/home/claude/w/analysis/frontier_margins_heldout.csv',index=False)
g=R.groupby(['dataset','method']).agg(n=('margin','size'),above=('margin',lambda x:(x>0).mean()),med=('margin','median'),q10=('margin',lambda x:x.quantile(.1)))
print(g.round(3))
c=R[R.method=='PRISM_event_paced'].groupby(['dataset','xai','eps','setting']).margin.mean()
print('cells x settings with mean margin>0:', (c>0).sum(),'/',len(c))
