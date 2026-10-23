"""Accepted training-only prediction function with explicit data context setup.

The original function body is preserved. Call configure_context first.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
fits = 0

def configure_context(compound_metadata, gdsc2_auc, chronos, lineages):
    """Set the original fitter's inputs; response scales remain train-only."""
    global meta, G, chron, lineage, fits
    meta, G, chron, lineage = compound_metadata, gdsc2_auc, chronos, lineages
    fits = 0

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
