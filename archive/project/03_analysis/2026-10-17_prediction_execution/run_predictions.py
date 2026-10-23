"""Full v1.1 prediction runner calling the exact accepted preflight function."""
from pathlib import Path
import ast,json,hashlib,sys,datetime,traceback
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PRE=ROOT/'03_analysis/2026-10-17_prediction_runtime_preflight'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'status':'INITIALIZING','models_completed':0,'contexts_completed':0}

def audit_context_coverage(row,train,test,isbench,ns):
 """Check v1.1 static inputs, never read held-out PRISM B outcomes for fitting.

 Background drug validity uses joint finite GDSC2 response AND target dependency.
 The same cell may miss a minority of background drugs, subject to the locked
 max(5, ceil(0.8 * panel_size)) residual-availability rule.
 """
 ids=train+test
 panel=row.general_panel_ids.split('|')
 controls=[row.donor_id,row.selected_control_id]+row.additional_control_ids.split('|')
 # Primary: test-time GDSC2 B response is NOT required; PRISM B is required above.
 # Benchmark: direct GDSC2 B response is explicitly required on its benchmark mask.
 strict_dependency=[row.recipient_id] if isbench else [row.recipient_id]+controls
 strict_gdsc2=[row.recipient_id] if isbench else controls
 for d in strict_dependency:
  gene=ns['meta'].at[d,'target']
  dep=ns['chron'][gene].reindex(ids).to_numpy(dtype=float)
  assert np.isfinite(dep).all(),(row.task_id,d,'required_dependency')
 for d in strict_gdsc2:
  resp=ns['G'][d].reindex(ids).to_numpy(dtype=float)
  assert np.isfinite(resp).all(),(row.task_id,d,'required_gdsc2_response')
 joint=[];train_observed={}
 for d in panel:
  gene=ns['meta'].at[d,'target']
  dep=ns['chron'][gene].reindex(ids).to_numpy(dtype=float)
  resp=ns['G'][d].reindex(ids).to_numpy(dtype=float)
  good=np.isfinite(dep)&np.isfinite(resp)
  train_n=int(good[:len(train)].sum())
  assert train_n>=50,(row.task_id,d,'background_joint_train_n',train_n)
  joint.append(good)
  train_observed[d]=train_n
 need=max(5,int(np.ceil(.8*len(panel))))
 coverage=np.column_stack(joint).sum(axis=1)
 assert (coverage>=need).all(),(row.task_id,'background_joint_cell_coverage',need,int(coverage.min()))
 return {'background_panel_size':len(panel),'background_required_per_cell':need,
         'min_background_per_cell':int(coverage.min()),
         'min_joint_background_train_n':min(train_observed.values())}

try:
 gate=json.loads((OUT/'regression_gate.json').read_text());assert gate['status']=='PASS' and gate['checks_passed']==gate['checks_total']==254
 source=PRE/'runtime_preflight.py';accepted=json.loads((PRE/'RUNTIME_PREFLIGHT.json').read_text());assert accepted['status']=='PASS' and accepted['source_sha256']==sha(source)
 tree=ast.parse(source.read_text());prefix=[]
 for node in tree.body:
  if isinstance(node,ast.Try):break
  prefix.append(node)
 # Load data/helper definitions only, stopping before preflight execution.
 ns={'__file__':str(source),'__name__':'accepted_preflight_fit_library'}
 exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
 predict=ns['predict_only'];digest=ns['digest'];canonical=ns['canonical'];tasks=ns['tasks'];cells=ns['cells'];bm=ns['bm'];SPEC=ns['SPEC']
 fitnode=next(n for n in prefix if isinstance(n,ast.FunctionDef) and n.name=='predict_only')
 pathcheck={'status':'PASS','accepted_source_sha256':sha(source),'predict_function_ast_sha256':hashlib.sha256(ast.dump(fitnode,include_attributes=False).encode()).hexdigest(),'function_reimplemented':False,'fold_adapter':'For actual fold k, cohort.fold is 0 for k and 1 otherwise; row order and cell IDs unchanged. Exact accepted function still partitions ne(0)/eq(0).','performance_parameters_changed':False}
 (OUT/'PRODUCTION_CODE_PATH_CHECK.json').write_text(json.dumps(pathcheck,indent=2)+'\n')
 maskrows=[];static=[]
 expected_global=pd.read_csv(SPEC/'CELL_FOLDS_v1.1.tsv',sep='\t').set_index('ModelID').proposed_fold.to_dict()
 assert len(tasks)==22 and tasks.recipient_id.nunique()==11 and tasks.recipient_target.nunique()==4
 for row in tasks.itertuples():
  primary=cells[cells.task_id.eq(row.task_id)]
  for isbench,table in [(False,cells),(True,bm)]:
   context=table[table.task_id.eq(row.task_id)].copy();assert context.ModelID.is_unique
   assert set(context.ModelID)<=set(primary.ModelID)
   assert all(expected_global[r.ModelID]==r.fold for r in context.itertuples())
   if not isbench:assert len(context)==row.n_with_general_coverage
   else:assert len(context)==row.same_drug_benchmark_n
   for k in range(5):
    train=context.loc[context.fold.ne(k),'ModelID'].tolist();test=context.loc[context.fold.eq(k),'ModelID'].tolist();assert not set(train)&set(test)
    assert len(train)>=150 and len(test)>=30
    assert np.isfinite(ns['P'][row.recipient_id].reindex(train+test)).all()
    assert ns['lineage'].reindex(train+test).notna().all()
    # Frozen-v1.1 coverage rules differ for primary and same-drug benchmark.
    # Background nuisance drugs may have missing dependency/response entries.
    coverage=audit_context_coverage(row,train,test,isbench,ns)
    for role,ids in [('train',train),('test',test)]:
     for cell in ids:maskrows.append({'task_id':row.task_id,'target':row.recipient_target,'benchmark':isbench,'actual_fold':k,'role':role,'ModelID':cell})
    static.append({'task_id':row.task_id,'target':row.recipient_target,'benchmark':isbench,'fold':k,'n_train':len(train),'n_test':len(test),'train_ids_sha256':digest(train),'test_ids_sha256':digest(test),**coverage,'pass':True})
 pd.DataFrame(static).to_csv(OUT/'ALL_FOLD_STATIC_AUDIT.tsv',sep='\t',index=False);pd.DataFrame(maskrows).to_csv(OUT/'ALL_FOLD_MASKS.tsv',sep='\t',index=False)
 (OUT/'FULL_READINESS_CHECK.json').write_text(json.dumps({'status':'PASS','primary_task_folds':110,'benchmark_task_folds':110,'all_ID_and_count_checks':True,'shared_fitter_verified':True,'models_started_before_static_pass':0},indent=2)+'\n')
 print('STATIC PASS: 110 primary + 110 benchmark contexts; no fits started',flush=True)
 if '--execute-fits' not in sys.argv:
  print('STATIC-ONLY exit; explicit --execute-fits required for future model execution',flush=True)
  raise SystemExit(0)
 print('Explicit execution option detected: beginning frozen prediction-only models',flush=True)
 artdir=OUT/'fit_artifacts';artdir.mkdir(exist_ok=False);predrows=[];logs=[];exact=[]
 accepted_context={(r['task_id'],r['benchmark']):r for r in accepted['records']}
 for row in tasks.sort_values(['recipient_target','donor_id','recipient_id']).itertuples():
  for k in range(5):
   for isbench,table in [(False,cells),(True,bm)]:
    cohort=table[table.task_id.eq(row.task_id)].copy();train=cohort.loc[cohort.fold.ne(k),'ModelID'].tolist();test=cohort.loc[cohort.fold.eq(k),'ModelID'].tolist()
    # Hand the accepted fitter training labels only; no test response values are passed.
    training_only=ns['P'][row.recipient_id].reindex(train).copy()
    adapter=cohort.copy();adapter['fold']=np.where(adapter.fold.eq(k),0,1)
    fitted=predict(row,training_only,adapter,isbench);h=digest(fitted)
    if k==0 and (row.task_id,isbench) in accepted_context:
     assert h==accepted_context[(row.task_id,isbench)]['baseline_sha256'],'STOP: accepted-context fit path mismatch'
     exact.append({'task_id':row.task_id,'benchmark':isbench,'exact_match':True,'sha256':h})
    stem=f'{row.task_id}__fold{k}__'+('benchmark' if isbench else 'primary');af=artdir/(stem+'.json');af.write_text(json.dumps(canonical(fitted),indent=2)+'\n')
    for name,pred in fitted['predictions'].items():
     assert np.isfinite(pred).all()
     for cell,value in zip(fitted['test_ids'],pred):predrows.append({'task_id':row.task_id,'donor_id':row.donor_id,'recipient_id':row.recipient_id,'target':row.recipient_target,'fold':k,'benchmark':isbench,'model':name,'ModelID':cell,'prediction_sensitivity':float(value),'training_response_sd':fitted['parameters']['Y']['sd']})
    logs.append({'task_id':row.task_id,'target':row.recipient_target,'fold':k,'benchmark':isbench,'n_train':fitted['n_train'],'n_test':fitted['n_test'],'n_models':len(fitted['model_names']),'fit_sha256':h,'artifact_sha256':sha(af),'status':'PASS'})
    report['models_completed']+=len(fitted['model_names']);report['contexts_completed']+=1
    (OUT/'RUN_PROGRESS.json').write_text(json.dumps(report,indent=2)+'\n')
  print(f'{row.recipient_target} {row.task_id}: 5 folds completed; cumulative models={report["models_completed"]}',flush=True)
 assert report['models_completed']==990 and report['contexts_completed']==220 and len(exact)==8
 pred=pd.DataFrame(predrows);assert not pred.duplicated(['task_id','model','ModelID']).any()
 pred.to_csv(OUT/'OOF_PREDICTIONS_ONLY.tsv',sep='\t',index=False);pd.DataFrame(logs).to_csv(OUT/'FOLD_EXECUTION_LOG.tsv',sep='\t',index=False)
 (OUT/'ACCEPTED_CONTEXT_REPLAY.json').write_text(json.dumps({'status':'PASS','exact_matches':exact},indent=2)+'\n')
 assert all(sha(SPEC/n)==h for n,h in ns['manifest']['files'].items())
 report.update({'status':'PREDICTIONS_SEALED','prediction_table_sha256':sha(OUT/'OOF_PREDICTIONS_ONLY.tsv'),'ols_fits_including_nuisance':ns['fits'],'heldout_performance_evaluated':False,'runtime_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'spec_manifest_sha256':sha(SPEC/'SPEC_FREEZE_MANIFEST.json'),'source_sha256':sha(Path(__file__))})
 (OUT/'PREDICTION_SEAL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
except Exception as e:
 report.update({'status':'STOP','error':repr(e),'traceback':traceback.format_exc()});(OUT/'STOP_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');raise
