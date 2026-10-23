"""Small runtime leakage check; never evaluate predictive performance."""
from pathlib import Path
import json,hashlib,sys,datetime,platform
import pandas as pd
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SPEC=ROOT/'revision/figure3_predictive_audit/2026-10-16/v1_1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((SPEC/'SPEC_FREEZE_MANIFEST.json').read_text())
assert all(sha(SPEC/n)==v for n,v in manifest['files'].items())
gate=json.loads((OUT/'regression_gate.json').read_text());assert gate['status']=='PASS' and gate['checks_passed']==gate['checks_total']==254
cfg=yaml.safe_load((SPEC/'run_config.yaml').read_text());paths={k:Path(v) if Path(v).is_absolute() else ROOT/v for k,v in cfg['sources'].items()}
for p,h in json.loads((SPEC/'INHERITED_INPUT_SHA256.json').read_text()).items():assert sha(p)==h
meta=pd.read_csv(paths['n3']/'n3_drug_results.tsv',sep='\t').set_index('compound_id');cov=pd.read_csv(paths['n3']/'n3_prism_compound_coverage.tsv',sep='\t').set_index('compound_id')
G=pd.read_csv(paths['gdsc2'],index_col=0);raw=pd.read_csv(paths['prism'],usecols=['broad_id','depmap_id','auc']);raw=raw[raw.depmap_id.str.startswith('ACH-',na=False)&raw.auc.notna()].copy();raw['core']=raw.broad_id.str[:13]
P={d:-raw[raw.core.isin(str(cov.at[d,'brd_ids']).split('|'))].groupby('depmap_id').auc.mean() for d in cov.index[cov.status.eq('LEGAL')]}
headers=pd.read_csv(paths['chronos'],nrows=0).columns;lookup={}
for h in headers[1:]:lookup.setdefault(h.split(' ')[0],h)
chron=pd.read_csv(paths['chronos'],usecols=[headers[0]]+[lookup[g] for g in set(meta.target)]).set_index(headers[0]).rename(columns={v:k for k,v in lookup.items()})
lineage=pd.read_csv(paths['models'],usecols=['ModelID','OncotreeLineage']).drop_duplicates('ModelID').set_index('ModelID').OncotreeLineage
cells=pd.read_csv(SPEC/'PRIMARY_TASK_CELLS_v1.1.tsv',sep='\t');bm=pd.read_csv(SPEC/'BENCHMARK_TASK_CELLS_v1.1.tsv',sep='\t');tasks=pd.read_csv(SPEC/'PREDICTION_ELIGIBILITY_v1.1.tsv',sep='\t')
# Pick one task per focal target using only IDs. One outer fold, fixed in advance.
chosen=tasks.sort_values(['recipient_target','donor_id','recipient_id']).groupby('recipient_target',sort=True).head(1)
records=[];fits=0

def canonical(value):
 if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 return value

def digest(value):return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False).encode()).hexdigest()

def predict_only(row,masked_y,cohort,benchmark=False):
 global fits
 train=cohort.loc[cohort.fold.ne(0),'ModelID'].tolist();test=cohort.loc[cohort.fold.eq(0),'ModelID'].tolist();ids=train+test;n=len(train)
 assert n>=150 and len(test)>=30
 # The training API immediately strips all held-out PRISM labels.
 y_train=masked_y.reindex(train).to_numpy(float).copy();assert np.isfinite(y_train).all()
 fitlog=[];art={};D={}
 labels=lineage.reindex(ids);counts=labels.iloc[:n].value_counts();keep=set(counts.index[counts.ge(10)]);mapped=labels.map(lambda s:s if s in keep else 'Other');cats=sorted(set(mapped.iloc[:n]));reference=cats[0];basis=cats[1:]
 L=np.column_stack([np.ones(len(ids))]+[(mapped==c).to_numpy(float) for c in basis]);art['lineage']={'kept':sorted(keep),'basis':basis,'reference':reference}
 def scale(x,key,trainmask=None):
  x=np.asarray(x,dtype=float);v=x[:n] if trainmask is None else x[:n][trainmask];v=v[np.isfinite(v)]
  mu=float(v.mean());sd=float(v.std(ddof=1));assert sd>1e-12 and np.isfinite(sd),(key,sd)
  art[key]={'mu':mu,'sd':sd,'n':len(v)};return (x-mu)/sd
 def design(drug):
  gene=meta.at[drug,'target']
  if gene not in D:D[gene]=scale(-chron[gene].reindex(ids).to_numpy(float),'D:'+gene)
  return np.column_stack([L,D[gene]])
 def ols(X,y,key,mask=None):
  global fits
  good=np.isfinite(y[:n])&np.isfinite(X[:n]).all(axis=1)
  if mask is not None:good&=mask
  assert good.sum()>=50,(key,int(good.sum()))
  beta,res,rank,sv=np.linalg.lstsq(X[:n][good],y[:n][good],rcond=1e-12)
  assert rank==X.shape[1],(key,rank,X.shape[1])
  art[key]={'coef':beta,'rank':int(rank),'singular_values':sv,'train_ids_sha256':digest([train[i] for i in np.flatnonzero(good)])};fits+=1
  return X@beta
 def residual(drug):
  key='E:'+drug
  if key in cache:return cache[key]
  z=design(drug);r=scale(-G[drug].reindex(ids).to_numpy(float),'R:'+drug);e=r-ols(z,r,'nuisance:'+drug)
  cache[key]=scale(e,'Escale:'+drug);return cache[key]
 cache={};z=design(row.recipient_id);panel=row.general_panel_ids.split('|');bg=np.column_stack([residual(d) for d in panel]);need=max(5,int(np.ceil(.8*len(panel))));assert (np.isfinite(bg).sum(axis=1)>=need).all()
 general=np.nanmedian(bg,axis=1);assert np.isfinite(general).all();general=scale(general,'Gscale')
 # Response scaling reads only the explicit training array.
 mu=float(y_train.mean());sd=float(y_train.std(ddof=1));assert sd>1e-12
 art['Y']={'mu':mu,'sd':sd};y=np.r_[(y_train-mu)/sd,np.full(len(test),np.nan)]
 ZG=np.column_stack([z,general])
 if benchmark:
  feature=scale(-G[row.recipient_id].reindex(ids).to_numpy(float),'same_drug_B');specs={'M_BG':np.column_stack([ZG,feature])}
 else:
  ea=residual(row.donor_id);ec=residual(row.selected_control_id);specs={'M0':z,'M1':np.column_stack([z,ea]),'M2':np.column_stack([z,ec]),'M0G':ZG,'M1G':np.column_stack([ZG,ea]),'M2G':np.column_stack([ZG,ec])}
  for k,d in enumerate(row.additional_control_ids.split('|'),1):specs['M2G_C'+str(k)]=np.column_stack([ZG,residual(d)])
 pred={}
 for key,X in specs.items():pred[key]=ols(X,y,'recipient:'+key)[n:]*sd+mu
 return {'parameters':art,'test_ids':test,'predictions':pred,'n_train':n,'n_test':len(test),'model_names':list(specs)}

try:
 for row in chosen.itertuples():
  for isbench in [False,True]:
   cohort=(bm if isbench else cells).loc[lambda x:x.task_id.eq(row.task_id)].copy();held=cohort.loc[cohort.fold.eq(0),'ModelID'];original=P[row.recipient_id].copy();perturbed=original.copy();perturbed.loc[held]=1e6+np.arange(len(held));masked=original.copy();masked.loc[held]=np.nan
   base=predict_only(row,original,cohort,isbench);changed=predict_only(row,perturbed,cohort,isbench);hidden=predict_only(row,masked,cohort,isbench)
   h=digest(base);assert digest(changed)==h and digest(hidden)==h,'STOP: held-out label leakage'
   # A deliberately leaky full-cohort normalizer must respond to the perturbation.
   sentinel_before=float(original.reindex(cohort.ModelID).mean());sentinel_after=float(perturbed.reindex(cohort.ModelID).mean());assert sentinel_before!=sentinel_after
   record={'task_id':row.task_id,'target':row.recipient_target,'fold':0,'benchmark':isbench,'n_train':base['n_train'],'n_test':base['n_test'],'model_names':base['model_names'],'baseline_sha256':h,'perturbed_sha256':digest(changed),'masked_sha256':digest(hidden),'exact_parameter_prediction_invariance':True,'intentional_leak_sentinel_detected':True}
   records.append(record)
   stem=row.task_id+('_benchmark' if isbench else '_primary');(OUT/(stem+'_fit_artifacts.json')).write_text(json.dumps(canonical(base),indent=2)+'\n')
 result={'status':'PASS','scope':'four ID-selected focal tasks, outer fold0 only; no full-run performance evaluation','records':records,'ols_fits_including_nuisance_and_triplicate_invariance_runs':fits,'heldout_performance_evaluated':False,'full_990_started':False,'spec_manifest_sha256':sha(SPEC/'SPEC_FREEZE_MANIFEST.json'),'source_sha256':sha(Path(__file__)),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'runtime_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
except Exception as e:
 result={'status':'STOP','error':repr(e),'records_completed':records,'ols_fits':fits,'full_990_started':False};(OUT/'RUNTIME_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');raise
assert all(sha(SPEC/n)==h for n,h in manifest['files'].items())
(OUT/'RUNTIME_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'task_cohort_checks':len(records),'fits':fits,'performance_evaluated':False},indent=2))
