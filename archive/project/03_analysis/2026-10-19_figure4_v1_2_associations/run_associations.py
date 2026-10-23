"""Execute the frozen v1.2 association family and conditional paired bootstrap."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,traceback,sys
import numpy as np
import pandas as pd
import scipy
from scipy import linalg
import yaml
B=Path(__file__).resolve().parent;cfg=yaml.safe_load((B/'run_config.yaml').read_text());R=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));A=R/cfg['assets'];P=R/cfg['score_source'];F=R/cfg['preflight']
sys.path.insert(0,str(A));import inference_helpers as frozen

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(name,rows):pd.DataFrame(rows).to_csv(B/name,sep='\t',index=False)
def scaled(x):
 assert np.isfinite(x).all() and np.std(x,axis=0,ddof=0).min()>=cfg['minimum_sd']
 return (x-np.mean(x,axis=0))/np.std(x,axis=0,ddof=0)
def bootstrap_slopes(C,S,Y,w):
 # Weighted Frisch-Waugh-Lovell is exactly OLS on the repeated sampled rows.
 gram=C.T@(w[:,None]*C);cross=C.T@(w[:,None]*np.column_stack([S,Y]));sol=linalg.solve(gram,cross,assume_a='pos',check_finite=False)
 sv=np.linalg.svd(gram,compute_uv=False)
 if np.sqrt(max(0.,sv[-1]))<=cfg['rank_tolerance']:raise RuntimeError('Invalid bootstrap nuisance rank')
 k=S.shape[1];den=np.sum(w[:,None]*S*S,axis=0)-np.sum(cross[:,:k]*sol[:,:k],axis=0)
 if not np.isfinite(den).all() or (den<=1e-10).any():raise RuntimeError('Invalid bootstrap program residual variance/rank')
 num=S.T@(w[:,None]*Y)-cross[:,:k].T@sol[:,k:]
 return num/den[:,None]

source_hashes={};drugrows=[];pairrows=[];targetrows=[];specrows=[];bootci=[];validation=[];bootstrap_status=[]
try:
 lock={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in [Path(__file__),B/'run_config.yaml',B/'EXECUTION_CONTRACT.md']}}
 (B/'PRE_EXECUTION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
 assert json.loads((B/'regression_gate.json').read_text())['checks_passed']==254
 assert json.loads((F/'PREFLIGHT_REPORT.json').read_text())['slots_passed']==300
 for root in [A,P,F]:
  for line in (root/'SHA256SUMS.txt').read_text().splitlines():
   h,p=line.split('  ',1);assert sha(root/p)==h,f'Frozen source changed {root/p}';source_hashes[str(root/p)]=h
 for p,h in json.loads((P/'INPUT_INTEGRITY.json').read_text())['input_sha256'].items():assert sha(Path(p))==h;source_hashes[p]=h
 (B/'INPUT_SHA256.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
 v=pd.read_csv(A/'n3_full_cohort_vectors.tsv.gz',sep='\t');E=v.pivot(index='ModelID',columns='compound_id',values='E');D=v.pivot(index='ModelID',columns='compound_id',values='D')
 masks=pd.read_csv(F/'REVISED_CELL_MASK.tsv',sep='\t');scores=pd.read_csv(P/'HALLMARK_SCORES.tsv.gz',sep='\t',index_col='ModelID');gv=pd.read_csv(P/'GENERAL_SENSITIVITY_VALUES.tsv',sep='\t');lineage=pd.read_csv(cfg['models']).drop_duplicates('ModelID').set_index('ModelID').OncotreeLineage
 drugs=pd.read_csv(R/'03_analysis/2026-09-28_phase1b_target_expansion/n3_drug_results.tsv',sep='\t');controls=pd.read_csv(A/'NEGATIVE_CONTROLS.tsv',sep='\t');pan=pd.read_csv(A/'GENERAL_PANELS.tsv',sep='\t');pairs=pd.read_csv(A/'ADMISSIBLE_PAIRS.tsv',sep='\t');flags=pd.read_csv(A/'PHARMACOLOGY_FLAGS_43.tsv',sep='\t').fillna('').set_index('compound_id')
 programs=sorted(scores.columns);targets=pan.target.tolist();assert len(programs)==50 and len(targets)==6 and pairs.admissible_pair.sum()==17
 rng=np.random.default_rng(cfg['seed']);scaling=[];scalprog=[];sample_rows=[]
 for target_index,t in enumerate(targets):
  cells=sorted(masks.loc[masks.target.eq(t)&masks.keep_v1_2,'ModelID']);n=len(cells);focal=drugs.loc[drugs.target.eq(t),'compound_id'].tolist();nc=controls.loc[controls.target.eq(t),'compound_id'].tolist();all_d=focal+nc;nf=len(focal)
  yraw=E.reindex(cells)[all_d].to_numpy();Y=scaled(yraw);sraw=scores.reindex(cells)[programs].to_numpy();S=scaled(sraw);dep=D.reindex(cells)[focal[0]].to_numpy();G=gv[gv.target.eq(t)].set_index('ModelID').reindex(cells).G.to_numpy();labs=lineage.reindex(cells);coding=pd.get_dummies(labs,dtype=float).reindex(columns=sorted(labs.unique()));L=coding.iloc[:,1:].to_numpy()
  C0=np.column_stack([np.ones(n),dep,L]);C1=np.column_stack([C0,G]);contexts={'G_adjusted':C1,'no_G_same_rows':C0};components={m:[] for m in contexts};actual_beta={m:np.empty((50,len(all_d))) for m in contexts}
  allowed=[(r.drug_a,r.drug_b) for r in pairs[pairs.target.eq(t)&pairs.admissible_pair].itertuples(index=False)];assert len(allowed)>=1
  for i,c in enumerate(cells):sample_rows.append({'target':t,'row_index':i,'ModelID':c,'lineage':labs[c]})
  for j,d in enumerate(all_d):scaling.append({'target':t,'compound_id':d,'mean':float(yraw[:,j].mean()),'sd_ddof0':float(yraw[:,j].std(ddof=0)),'n':n})
  for pi,p in enumerate(programs):
   scalprog.append({'target':t,'program':p,'mean':float(sraw[:,pi].mean()),'sd_ddof0':float(sraw[:,pi].std(ddof=0)),'n':n})
   comp_by_model={}
   for model,C in contexts.items():
    comp={}
    for j,d in enumerate(all_d):
     r=frozen.directional_hc3(Y[:,j],C[:,1:],S[:,pi]);assert np.isfinite(list(r.values())).all()
     if min(r['p_plus'],r['p_minus'])<=0:raise RuntimeError('Component tail underflow; stop before reporting rounded zero')
     comp[d]=r;actual_beta[model][pi,j]=r['beta'];drugrows.append({'target':t,'program':p,'compound_id':d,'compound_name':flags.at[d,'compound_name'],'role':'focal' if j<nf else 'control','model':model,'n':n,'status':'ESTIMATED',**r,'pharmacology_flags':flags.at[d,'flags'],'selectivity_status':flags.at[d,'exclusive_selectivity_status']})
    comp_by_model[model]=comp
    for r in pairs[pairs.target.eq(t)].itertuples(index=False):
     if r.admissible_pair:
      pp=min(1.,2*min(max(comp[r.drug_a]['p_plus'],comp[r.drug_b]['p_plus']),max(comp[r.drug_a]['p_minus'],comp[r.drug_b]['p_minus'])))
     else:pp=np.nan
     pairrows.append({'target':t,'program':p,'model':model,'drug_a':r.drug_a,'drug_b':r.drug_b,'admissible':bool(r.admissible_pair),'pair_p':pp,'beta_a':comp[r.drug_a]['beta'],'beta_b':comp[r.drug_b]['beta'],'same_direction':comp[r.drug_a]['beta']*comp[r.drug_b]['beta']>0})
   for model,comp in comp_by_model.items():
    tp=frozen.pair_supported_p(comp,allowed);aps=[(min(1.,2*min(max(comp[a]['p_plus'],comp[b]['p_plus']),max(comp[a]['p_minus'],comp[b]['p_minus']))),a,b) for a,b in allowed];best=sorted(aps)[0]
    targetrows.append({'target':t,'program':p,'model':model,'n':n,'n_focal':nf,'n_controls':2,'n_admissible_pairs':len(allowed),'target_program_p':tp,'best_pair_a':best[1],'best_pair_b':best[2],'best_pair_same_direction':comp[best[1]]['beta']*comp[best[2]]['beta']>0,'median_beta_focal':float(np.median([comp[d]['beta'] for d in focal])),'median_beta_control':float(np.median([comp[d]['beta'] for d in nc])),'status':'EVALUATED'})
  # Verify the fast bootstrap algebra against the unchanged fitted OLS slopes.
  for model,C in contexts.items():
   b0=bootstrap_slopes(C,S,Y,np.ones(n));err=float(np.max(abs(b0-actual_beta[model])));assert err<=1e-10;validation.append({'target':t,'model':model,'FWL_vs_frozen_helper_max_beta_error':err})
  print(t,'all 50 associations complete; starting 2000 paired within-lineage draws',flush=True)
  boot=np.full((cfg['bootstrap_draws'],2,50,len(all_d)),np.nan);weights=np.zeros((cfg['bootstrap_draws'],n),dtype=np.uint16);strata=[np.flatnonzero(labs.to_numpy()==l) for l in sorted(labs.unique())]
  try:
   for b in range(cfg['bootstrap_draws']):
    idx=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata]);w=np.bincount(idx,minlength=n);weights[b]=w
    for mi,(model,C) in enumerate(contexts.items()):boot[b,mi]=bootstrap_slopes(C,S,Y,w)
    if b%500==499:print(t,'bootstrap',b+1,flush=True)
   assert np.isfinite(boot).all()
  except Exception:
   np.savez_compressed(B/(t+'_bootstrap_partial.npz'),beta=boot,weights=weights,drug_ids=all_d,programs=programs,cells=cells,models=list(contexts));bootstrap_status.append({'target':t,'status':'FAIL','failed_draw':b,'reason':traceback.format_exc()});raise
  np.savez_compressed(B/(t+'_bootstrap.npz'),beta=boot,weights=weights,drug_ids=all_d,programs=programs,cells=cells,models=list(contexts))
  for mi,model in enumerate(contexts):
   bf=np.median(boot[:,mi,:,:nf],axis=2);bc=np.median(boot[:,mi,:,nf:],axis=2);delta=bf-bc
   for pi,p in enumerate(programs):
    for j,d in enumerate(all_d):
     lo,hi=np.quantile(boot[:,mi,pi,j],[.025,.975]);bootci.append({'target':t,'program':p,'model':model,'compound_id':d,'CI_low':lo,'CI_high':hi,'conditional_bootstrap_n':cfg['bootstrap_draws']})
    medf=float(np.median(actual_beta[model][pi,:nf]));medc=float(np.median(actual_beta[model][pi,nf:]));lo,hi=np.quantile(delta[:,pi],[.025,.975]);fl,fh=np.quantile(bf[:,pi],[.025,.975]);cl,ch=np.quantile(bc[:,pi],[.025,.975])
    specrows.append({'target':t,'program':p,'model':model,'n':n,'median_beta_focal':medf,'median_beta_control':medc,'delta':medf-medc,'abs_focal_minus_abs_control':abs(medf)-abs(medc),'focal_CI_low':fl,'focal_CI_high':fh,'control_CI_low':cl,'control_CI_high':ch,'delta_CI_low':lo,'delta_CI_high':hi,'bootstrap_n':cfg['bootstrap_draws'],'specificity_inference':'DESCRIPTIVE_ONLY_NO_P_OR_Q'})
  bootstrap_status.append({'target':t,'status':'PASS','draws':cfg['bootstrap_draws'],'invalid_draws':0})
  write('DRUG_RESULTS_checkpoint.tsv',drugrows);write('TARGET_RESULTS_checkpoint.tsv',targetrows);write('SPECIFICITY_checkpoint.tsv',specrows);write('BOOTSTRAP_STATUS.tsv',bootstrap_status)
  print(t,'COMPLETE',flush=True)
 df=pd.DataFrame(targetrows);assert len(df)==600
 for model in contexts:
  ix=df.model.eq(model);assert ix.sum()==300;df.loc[ix,'BY_q']=frozen.by_300(df.loc[ix,'target_program_p'].values)
 df['corrected_pair_supported']=df.BY_q.le(.05)&df.best_pair_same_direction
 primary=df[df.model.eq('G_adjusted')].copy();sensitivity=df[df.model.eq('no_G_same_rows')].set_index(['target','program'])
 primary['no_G_BY_q']=[sensitivity.at[(r.target,r.program),'BY_q'] for r in primary.itertuples()];primary['no_G_pair_supported']=[bool(sensitivity.at[(r.target,r.program),'corrected_pair_supported']) for r in primary.itertuples()];primary['general_sensitivity_sensitive']=~primary.corrected_pair_supported&primary.no_G_pair_supported
 df.to_csv(B/'TARGET_PROGRAM_ALL_MODELS.tsv',sep='\t',index=False);primary.to_csv(B/'PRIMARY_300_RESULTS.tsv',sep='\t',index=False)
 write('DRUG_LEVEL_RESULTS.tsv',drugrows);write('PAIR_LEVEL_RESULTS.tsv',pairrows);write('DESCRIPTIVE_SPECIFICITY.tsv',specrows);write('DRUG_CONDITIONAL_CI.tsv',bootci);write('BOOTSTRAP_ALGEBRA_CHECK.tsv',validation);write('OUTCOME_SCALING.tsv',scaling);write('PROGRAM_SCALING.tsv',scalprog);write('CELL_ROWS.tsv',sample_rows)
 assert all(sha(Path(p))==h for p,h in source_hashes.items())
 report={'status':'COMPLETED','primary_slots':300,'primary_corrected_pair_supported':int(primary.corrected_pair_supported.sum()),'primary_by_target':primary.groupby('target').corrected_pair_supported.sum().astype(int).to_dict(),'minimum_primary_BY_q':float(primary.BY_q.min()),'no_G_corrected_pair_supported':int(df.loc[df.model.eq('no_G_same_rows'),'corrected_pair_supported'].sum()),'general_sensitivity_sensitive_slots':int(primary.general_sensitivity_sensitive.sum()),'drug_model_fits':len(drugrows),'paired_bootstrap_draws_per_target':cfg['bootstrap_draws'],'targets':len(targets),'invalid_bootstrap_draws':0,'bootstrap_model_coefficient_vectors':int(cfg['bootstrap_draws']*2*50*sum(len(drugs[drugs.target.eq(t)])+2 for t in targets)),'frozen_sources_unchanged':True,'specificity_p_q_computed':False,'evidence_level':'Multiplicity-corrected approximate molecular association only; no causal/functional validation','plots_and_claim_edits':False,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
 (B/'EXECUTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
except Exception:
 write('PARTIAL_DRUG_RESULTS.tsv',drugrows);write('PARTIAL_TARGET_RESULTS.tsv',targetrows);write('BOOTSTRAP_STATUS.tsv',bootstrap_status);(B/'TECHNICAL_STOP.txt').write_text(traceback.format_exc());raise
