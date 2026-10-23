"""Recover two N3 residual vectors deterministically; verify frozen n and rho."""
import os,sys,json,hashlib,importlib.util,traceback,platform
from pathlib import Path
from datetime import datetime,timezone
import yaml
import numpy as np
import pandas as pd
import scipy
CFG=Path(__file__).with_name('run_config.yaml')
cfg=yaml.safe_load(CFG.read_text());root=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']))
nd=root/cfg['n3_dir'];out=root/cfg['recovery_output']
if out.exists():raise RuntimeError('Use a fresh recovery output; do not overwrite evidence')
out.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(name,obj):
 (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
report={'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'config_sha256':sha(CFG),'new_inference':False,'historical_arena_vectors_used':False}
try:
 provenance=json.loads((nd/'provenance.json').read_text())
 code_checks=[]
 for rel in ['03_analysis/2026-09-28_phase1b_target_expansion/run_n3.py','discordance/revision/rev_common.py']:
  actual=sha(root/rel);expected=provenance['code_sha256'][rel]
  code_checks.append({'path':rel,'expected':expected,'actual':actual,'pass':actual==expected})
 for name in ['n3_pairs_gdsc2.tsv','n3_drug_results.tsv','n3_target_results.tsv']:
  rel=str((nd/name).relative_to(root));expected=provenance['generated_files_sha256'][rel];actual=sha(root/rel)
  code_checks.append({'path':rel,'expected':expected,'actual':actual,'pass':actual==expected})
 save('frozen_source_checks.json',code_checks)
 if not all(c['pass'] for c in code_checks):raise RuntimeError('Frozen source hash mismatch')
 pairs=pd.read_csv(nd/'n3_pairs_gdsc2.tsv',sep='\t');sel=cfg['selection']
 candidates=pairs[pairs.same_target & pairs.target_a.eq(sel['target']) & pairs.target_b.eq(sel['target'])].copy()
 candidates['eligible_display']=candidates.n.ge(sel['minimum_frozen_pair_n']) & candidates.scaffold_tanimoto.lt(sel['maximum_scaffold_tanimoto_exclusive'])
 ranked=candidates[candidates.eligible_display].sort_values(['n','scaffold_tanimoto','drug_a','drug_b'],ascending=[False,True,True,True])
 if len(ranked)==0:raise RuntimeError('No pre-specified eligible example')
 chosen=ranked.iloc[0];candidates['selected']=(candidates.drug_a==chosen.drug_a)&(candidates.drug_b==chosen.drug_b)
 candidates.to_csv(out/'selection_candidates.tsv',sep='\t',index=False)
 save('selection_lock.json',{'rule':sel,'selected':chosen.to_dict(),'lock_time_utc':datetime.now(timezone.utc).isoformat(),'note':'Written before reading raw matrices or fitting either compound'})
 manifest=json.loads((root/cfg['input_manifest']).read_text());raw_checks=[]
 spec=importlib.util.spec_from_file_location('frozen_n3_panel_c',nd/'run_n3.py');n3=importlib.util.module_from_spec(spec);spec.loader.exec_module(n3)
 raw_paths=[n3.GDSC_AUC_PATH,n3.rc.gdsc2.CRISPR_PATH,n3.rc.gdsc2.MODEL_PATH]
 for path in raw_paths:
  records=[r for r in manifest['files'] if Path(r['logical_path'])==Path(path)]
  if len(records)!=1:raise RuntimeError(f'No unique frozen input record: {path}')
  actual=sha(path);expected=records[0]['sha256'];raw_checks.append({'path':str(path),'expected':expected,'actual':actual,'pass':actual==expected})
 save('raw_input_checks.json',raw_checks)
 if not all(r['pass'] for r in raw_checks):raise RuntimeError('Raw input hash mismatch')
 print('Frozen code and raw inputs PASS; selected',chosen.name_a,chosen.name_b,flush=True)
 chronos=n3.rc.gdsc2.load_chronos();_,lineage=n3.rc._models_lineage()
 auc=pd.read_csv(n3.GDSC_AUC_PATH,index_col=0,low_memory=False);auc.index=auc.index.astype(str)
 cells=auc.index.intersection(chronos.index).intersection(lineage.index);lineage=lineage.reindex(cells)
 rows=[];residuals={};fit_checks=[]
 frozen_drugs=pd.read_csv(nd/'n3_drug_results.tsv',sep='\t').set_index('compound_id')
 for drug in [chosen.drug_a,chosen.drug_b]:
  response=n3.rc.gdsc2.response_vector(auc.loc[cells,drug])
  dependency=-chronos.loc[cells,sel['target']].astype(float)
  residual=n3.rc.linear_ols(response,dependency,lineage)
  n=int(residual.notna().sum());expected=int(frozen_drugs.loc[drug,'n'])
  fit_checks.append({'drug':drug,'n_recovered':n,'n_frozen':expected,'pass':n==expected})
  if n!=expected:raise RuntimeError(f'Per-drug complete-case mismatch: {drug}')
  residuals[drug]=residual
  mask=pd.concat([response.rename('R'),dependency.rename('D'),lineage],axis=1).notna().all(axis=1)
  frame=pd.DataFrame({'ModelID':cells,'compound_id':drug,'compound_name':frozen_drugs.loc[drug,'compound_name'],'target':sel['target'],'R':response.to_numpy(),'D':dependency.to_numpy(),'R_hat':(response-residual).to_numpy(),'E':residual.to_numpy(),'complete_case':mask.to_numpy()})
  rows.append(frame)
 full=pd.concat(rows,ignore_index=True)
 full.to_csv(out/'n3_selected_cell_vectors.tsv',sep='\t',index=False,float_format='%.17g')
 rho,n=n3.rc.gdsc2.spearman_pair(residuals[chosen.drug_a],residuals[chosen.drug_b])
 error=abs(float(rho)-float(chosen.rho))
 report.update({'selected_compounds':[chosen.drug_a,chosen.drug_b],'selected_names':[chosen.name_a,chosen.name_b],'rho_recovered':float(rho),'rho_frozen':float(chosen.rho),'rho_absolute_difference':error,'n_recovered':int(n),'n_frozen':int(chosen.n),'per_drug_checks':fit_checks})
 if int(n)!=int(chosen.n) or error>cfg['verification']['rho_absolute_tolerance']:raise RuntimeError('Selected pair verification failed; STOP')
 a=full[full.compound_id.eq(chosen.drug_a)][['ModelID','E']].rename(columns={'E':'E_a'})
 b=full[full.compound_id.eq(chosen.drug_b)][['ModelID','E']].rename(columns={'E':'E_b'})
 joined=a.merge(b,on='ModelID',validate='one_to_one');joined['pair_complete_case']=joined[['E_a','E_b']].notna().all(axis=1)
 joined.to_csv(out/'panel_c_cell_mask.tsv',sep='\t',index=False,float_format='%.17g')
 joined[joined.pair_complete_case].to_csv(out/'panel_c_plot_cells.tsv',sep='\t',index=False,float_format='%.17g')
 report['status']='PASS';report['end_utc']=datetime.now(timezone.utc).isoformat()
 report['code_sources_after_unchanged']=all(sha(root/c['path'])==c['actual'] for c in code_checks)
 if not report['code_sources_after_unchanged']:raise RuntimeError('Frozen sources changed during recovery')
 save('recovery_report.json',report)
 print(json.dumps(report,indent=2),flush=True)
except Exception as exc:
 report.update({'status':'STOP','error':str(exc),'traceback':traceback.format_exc()});save('recovery_report.json',report)
 raise
