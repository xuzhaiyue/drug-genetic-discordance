"""Synthetic data only; no biological input is loaded."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
B=Path(__file__).resolve().parent;rng=np.random.default_rng(20261018);n=400;ns=2000
X=np.column_stack([np.ones(n),np.eye(8)[rng.integers(0,8,n)][:,1:],rng.normal(size=(n,3))]);inv=np.linalg.inv(X.T@X);a=inv@X.T;h=np.sum(X*(X@inv),axis=1);df=n-X.shape[1]
# The last feature is the synthetic program, with dependency/G as nuisance.
def fit(Y):
 beta=a@Y;e=Y-X@beta;se=np.sqrt(np.sum((a[-1,:,None]*e/(1-h[:,None]))**2,axis=0));return beta[-1],se
conditions={'null':[0,0,0,0],'heteroskedastic_null':[0,0,0,0],'heavy_tail_null':[0,0,0,0],'one_active':[.8,0,0,0],'opposite':[.8,-.8,0,0],'two_active':[.8,.8,0,0]};rows=[]
for name,signal in conditions.items():
 eps=rng.normal(size=(n,ns,4))
 if name=='heavy_tail_null':eps=rng.standard_t(4,size=(n,ns,4))/np.sqrt(2)
 shared=rng.normal(size=(n,ns,1));eps=.7*eps+.7*shared
 if name in ['heteroskedastic_null','heavy_tail_null']:eps*= (.4+abs(X[:,-1]))[:,None,None]
 Y=eps+X[:,-1,None,None]*np.array(signal)[None,None,:]+.3*X[:,-2,None,None]
 beta,se=fit(Y.reshape(n,-1));T=(beta/se).reshape(ns,4);plus=stats.t.sf(T,df);minus=stats.t.cdf(T,df);two=2*np.minimum(plus,minus)
 pp=np.stack([2*np.minimum(np.maximum(plus[:,i],plus[:,j]),np.maximum(minus[:,i],minus[:,j])) for i,j in itertools.combinations(range(4),2)],axis=1)
 target=np.minimum(1,6*pp.min(axis=1));rates=(two<.05).mean(axis=0);targetrate=float((target<.05).mean())
 nulls=np.array(signal)==0;passed=bool((rates[nulls]<=.07).all() and (name=='two_active' or targetrate<=.07))
 rows.append({'condition':name,'drug_rejection_rates':rates.tolist(),'pair_supported_rejection_rate':targetrate,'status':'PASS' if passed else 'FAIL'})
# Equal nonzero effects: preserve paired-cell dependence when estimating delta.
deltas=[];covered=[]
for rep in range(300):
 x=rng.normal(size=n);shared=rng.normal(size=n);Y=.4*x[:,None]+shared[:,None]*.6+rng.normal(size=(n,4))
 def delta(idx):
  xx=x[idx]-x[idx].mean();slopes=(xx[:,None]*(Y[idx]-Y[idx].mean(axis=0))).sum(axis=0)/(xx@xx);return np.median(slopes[:2])-np.median(slopes[2:])
 d=delta(np.arange(n));draws=np.array([delta(rng.integers(0,n,n)) for _ in range(499)]);lo,hi=np.quantile(draws,[.025,.975]);deltas.append(d);covered.append(lo<=0<=hi)
sp={'replicates':300,'draws':499,'mean_delta':float(np.mean(deltas)),'coverage':float(np.mean(covered))};sp['status']='PASS' if abs(sp['mean_delta'])<=.05 and sp['coverage']>=.90 else 'FAIL'
H=sum(1/i for i in range(1,301));cutoff=.05/(300*H);p=np.r_[1e-8,np.ones(299)];reject,q,_,_=multipletests(p,alpha=.05,method='fdr_by')
report={'status':'PASS' if all(r['status']=='PASS' for r in rows) and sp['status']=='PASS' and reject[0] else 'STOP','seed':20261018,'plan_sha256':hashlib.sha256((B/'CALIBRATION_PLAN.md').read_bytes()).hexdigest(),'n':n,'simulations_per_condition':ns,'conditions':rows,'specificity_equal_nonzero':sp,'BY_rank1_cutoff':cutoff,'permutation_component_floor':.0001,'max_pair_corrected_floor':.0012,'analytic_single_strong_q':float(q[0]),'limitations':'Approximate HC3 inference; nominal-size Monte Carlo is not extreme-tail validation, proof of BY validity, or real-data power. No specificity P values.'}
(B/'SYNTHETIC_CALIBRATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['status']=='PASS'
