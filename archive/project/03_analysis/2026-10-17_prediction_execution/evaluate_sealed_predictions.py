"""Evaluate only sealed v1.1 OOF predictions; never fit or tune a model."""
from pathlib import Path
import json,hashlib,datetime,traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SPEC=ROOT/'revision/figure3_predictive_audit/2026-10-16/v1_1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 seal=json.loads((OUT/'PREDICTION_SEAL.json').read_text());assert seal['status']=='PREDICTIONS_SEALED' and seal['models_completed']==990
 assert sha(OUT/'OOF_PREDICTIONS_ONLY.tsv')==seal['prediction_table_sha256']
 # This is the first performance stage. Raw test outcomes never enter the fitting runner.
 curves=pd.read_csv(ROOT/'raw/prism/secondary-screen-dose-response-curve-parameters.csv',usecols=['broad_id','depmap_id','auc']);curves=curves[curves.depmap_id.str.startswith('ACH-',na=False)&curves.auc.notna()].copy();curves['core']=curves.broad_id.str[:13]
 cov=pd.read_csv(ROOT/'03_analysis/2026-09-28_phase1b_target_expansion/n3_prism_compound_coverage.tsv',sep='\t').set_index('compound_id');pred=pd.read_csv(OUT/'OOF_PREDICTIONS_ONLY.tsv',sep='\t');tasks=pd.read_csv(SPEC/'PREDICTION_ELIGIBILITY_v1.1.tsv',sep='\t').set_index('task_id')
 Y={b:-curves[curves.core.isin(str(cov.at[b,'brd_ids']).split('|'))].groupby('depmap_id').auc.mean() for b in pred.recipient_id.unique()}
 pred['observed_sensitivity']=[float(Y[b].loc[c]) for b,c in zip(pred.recipient_id,pred.ModelID)];assert np.isfinite(pred.observed_sensitivity).all();pred['squared_error']=(pred.observed_sensitivity-pred.prediction_sensitivity)**2;pred['squared_normalized_error']=pred.squared_error/pred.training_response_sd**2
 pred.to_csv(OUT/'OOF_EVALUATION_WITH_TRUTH.tsv',sep='\t',index=False)
 def metrics(frame):
  y=frame.observed_sensitivity.to_numpy();sst=float(((y-y.mean())**2).sum());assert sst>0
  return {'n':len(y),'r2':1-float(frame.squared_error.sum())/sst,'rmse':float(np.sqrt(frame.squared_error.mean())),'nrmse':float(np.sqrt(frame.squared_normalized_error.mean()))}
 rows=[]
 for (task,model),v in pred.groupby(['task_id','model']):rows.append({'task_id':task,'target':v.target.iloc[0],'donor_id':v.donor_id.iloc[0],'recipient_id':v.recipient_id.iloc[0],'model':model,'benchmark':bool(v.benchmark.iloc[0]),**metrics(v)})
 scores=pd.DataFrame(rows);scores.to_csv(OUT/'TASK_MODEL_PERFORMANCE.tsv',sep='\t',index=False)
 primary=scores[~scores.benchmark];models=['M0','M1','M2','M0G','M1G','M2G','M2G_C1','M2G_C2'];tid=sorted(primary.task_id.unique());taskmeta=tasks.loc[tid];pairs={'M1_vs_M0':('M1','M0'),'M1_vs_M2':('M1','M2'),'M1G_vs_M0G':('M1G','M0G'),'M1G_vs_M2G':('M1G','M2G'),'M1G_vs_C1':('M1G','M2G_C1'),'M1G_vs_C2':('M1G','M2G_C2')}
 wide={metric:primary.pivot(index='task_id',columns='model',values=metric).reindex(tid) for metric in ['r2','rmse','nrmse']}
 contrasts=[]
 for task in tid:
  for label,(enhanced,base) in pairs.items():contrasts.append({'task_id':task,'target':tasks.at[task,'recipient_target'],'donor_id':tasks.at[task,'donor_id'],'recipient_id':tasks.at[task,'recipient_id'],'comparison':label,'delta_r2':float(wide['r2'].at[task,enhanced]-wide['r2'].at[task,base]),'rmse_reduction':float(wide['rmse'].at[task,base]-wide['rmse'].at[task,enhanced]),'nrmse_reduction':float(wide['nrmse'].at[task,base]-wide['nrmse'].at[task,enhanced])})
 contrast=pd.DataFrame(contrasts);contrast.to_csv(OUT/'TASK_INCREMENTAL_PERFORMANCE.tsv',sep='\t',index=False)
 def nested(frame,value,by):
  rec=frame.groupby(['target','recipient_id',by])[value].median().reset_index();tar=rec.groupby(['target',by])[value].median().reset_index();overall=tar.groupby(by)[value].median().to_dict();return rec,tar,overall
 for value in ['r2','rmse','nrmse']:
  rec,tar,total=nested(primary,value,'model');rec.to_csv(OUT/f'RECIPIENT_{value.upper()}.tsv',sep='\t',index=False);tar.to_csv(OUT/f'TARGET_{value.upper()}.tsv',sep='\t',index=False)
 rec,tar,total=nested(contrast,'delta_r2','comparison');rec.to_csv(OUT/'RECIPIENT_INCREMENTAL_R2.tsv',sep='\t',index=False);tar.to_csv(OUT/'TARGET_INCREMENTAL_R2.tsv',sep='\t',index=False)
 contrast.groupby(['target','donor_id','comparison'])[['delta_r2','rmse_reduction','nrmse_reduction']].median().reset_index().to_csv(OUT/'DONOR_INCREMENTAL_PERFORMANCE.tsv',sep='\t',index=False)
 primary.groupby(['target','donor_id','model'])[['r2','rmse','nrmse']].median().reset_index().to_csv(OUT/'DONOR_MODEL_PERFORMANCE.tsv',sep='\t',index=False)
 # Preserve per-task pharmacology flags alongside every increment.
 flagcols=[c for c in tasks.columns if c.endswith('_flags') or c.endswith('_evidence_summary')];contrast.merge(tasks[flagcols],left_on='task_id',right_index=True,validate='many_to_one').to_csv(OUT/'TASK_INCREMENTAL_WITH_PHARMACOLOGY_FLAGS.tsv',sep='\t',index=False)
 # Paired ID bootstrap: every model/task shares the same cell multiplicities.
 rawprimary=pred[~pred.benchmark];pool=sorted(rawprimary.ModelID.unique());lookup={c:i for i,c in enumerate(pool)};rng=np.random.default_rng(20261017);B=2000;W=rng.multinomial(len(pool),np.full(len(pool),1/len(pool)),size=B)
 boot={m:np.empty((B,len(tid))) for m in models};bootn={m:np.empty((B,len(tid))) for m in models}
 for j,task in enumerate(tid):
  v=rawprimary[rawprimary.task_id.eq(task)];index=sorted(v.ModelID.unique());wt=W[:,[lookup[c] for c in index]];v0=v[v.model.eq('M0')].set_index('ModelID').reindex(index);y=v0.observed_sensitivity.to_numpy();n=wt.sum(axis=1);sy=wt@y;sst=wt@(y*y)-sy*sy/n;assert (n>1).all() and (sst>0).all()
  for model in models:
   z=v[v.model.eq(model)].set_index('ModelID').reindex(index);boot[model][:,j]=1-(wt@z.squared_error.to_numpy())/sst;bootn[model][:,j]=np.sqrt((wt@z.squared_normalized_error.to_numpy())/n)
 def bootstrap_aggregate(array):
  ts=[]
  for target in sorted(taskmeta.recipient_target.unique()):
   rs=[]
   for recipient in sorted(taskmeta.loc[taskmeta.recipient_target.eq(target),'recipient_id'].unique()):rs.append(np.median(array[:,np.flatnonzero(taskmeta.recipient_id.eq(recipient).to_numpy())],axis=1))
   ts.append(np.median(np.column_stack(rs),axis=1))
  return np.median(np.column_stack(ts),axis=1)
 bootrows=[];ci={}
 for name,(enhanced,base) in pairs.items():
  vals=bootstrap_aggregate(boot[enhanced]-boot[base]);ci[name]=[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
  for k,val in enumerate(vals):bootrows.append({'replicate':k,'comparison':name,'delta_r2_equal_target':float(val)})
 pd.DataFrame(bootrows).to_csv(OUT/'PAIRED_CELL_BOOTSTRAP.tsv',sep='\t',index=False)
 # Benchmark comparisons use the same available-B test subset; original predictions are not refit.
 brows=[]
 for task in tid:
  v=pred[pred.task_id.eq(task)];available=set(v.loc[v.model.eq('M_BG'),'ModelID'])
  for model in models+['M_BG']:
   z=v[v.model.eq(model)&v.ModelID.isin(available)];assert len(z)==len(available)
   brows.append({'task_id':task,'target':tasks.at[task,'recipient_target'],'model':model,'population':'available-GDSC2-B overlap; training sets can differ',**metrics(z)})
 pd.DataFrame(brows).to_csv(OUT/'BENCHMARK_OVERLAP_PERFORMANCE.tsv',sep='\t',index=False)
 absrec,abstar,abstotal=nested(primary,'r2','model');targetwide=tar.pivot(index='target',columns='comparison',values='delta_r2');consistent=int(((targetwide.M1G_vs_M0G>0)&(targetwide.M1G_vs_M2G>0)).sum());supported=total['M1G_vs_M0G']>=.01 and total['M1G_vs_M2G']>0 and consistent>=3 and ci['M1G_vs_M0G'][0]>0 and ci['M1G_vs_M2G'][0]>0
 status='SUPPORTED_IN_TESTED_SET' if supported and abstotal['M1G']>0 else 'RELATIVE_GAIN_WITHOUT_USEFUL_ABSOLUTE_PREDICTION' if supported else 'NOT_SUPPORTED' if total['M1G_vs_M0G']<=0 or total['M1G_vs_M2G']<=0 else 'MODEST_OR_UNCERTAIN'
 loto=[]
 for dropped in sorted(tar.target.unique()):
  for label,value in tar[~tar.target.eq(dropped)].groupby('comparison').delta_r2.median().items():loto.append({'removed_target':dropped,'comparison':label,'median_delta_r2_remaining_targets':float(value)})
 pd.DataFrame(loto).to_csv(OUT/'LEAVE_ONE_TARGET_OUT_DESCRIPTIVE.tsv',sep='\t',index=False)
 result={'engineering_status':'PASS_990_SEALED','scientific_status':status,'equal_target_median_absolute_r2':abstotal,'equal_target_median_delta_r2':total,'conditional_paired_cell_ci95':ci,'targets_positive_for_both_decisive_contrasts':consistent,'target_delta_r2':tar.to_dict('records'),'bootstrap_replicates':B,'bootstrap_seed':20261017,'bootstrap_cell_pool':len(pool),'bootstrap_scope':'conditional on fixed OOF fits/eligibility; no refitting or target/drug-population uncertainty','pragmatic_delta_r2_threshold':.01,'clinically_validated_threshold':False,'prediction_seal_sha256':sha(OUT/'PREDICTION_SEAL.json'),'new_model_fits_in_evaluation':0,'evaluation_source_sha256':sha(Path(__file__)),'evaluation_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'PERFORMANCE_DECISION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
except Exception as e:
 (OUT/'EVALUATION_STOP.json').write_text(json.dumps({'status':'STOP','error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
