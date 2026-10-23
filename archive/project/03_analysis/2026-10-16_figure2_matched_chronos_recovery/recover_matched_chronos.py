"""Recover the unsaved original N3 matched Chronos baseline; no new inference."""
import os,sys,json,hashlib,importlib.util,traceback,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy,yaml
CFG=Path(__file__).with_name('run_config.yaml');cfg=yaml.safe_load(CFG.read_text());root=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));nd=root/cfg['n3_dir'];out=root/cfg['output_dir']
if out.exists():raise RuntimeError('Refuse overwrite: choose a fresh output directory')
out.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
r={'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'config_sha256':sha(CFG),'new_model_selection':False,'new_inference':False,'new_permutations':0,'new_bootstraps':0,'demeter2_residuals_refitted':False}
try:
 prov=json.loads((nd/'provenance.json').read_text());original_cfg=yaml.safe_load((nd/'run_config.yaml').read_text());summary=json.loads((nd/'n3_results.json').read_text())
 source_checks=[]
 for rel in ['03_analysis/2026-09-28_phase1b_target_expansion/run_n3.py','discordance/revision/rev_common.py','discordance/revision/rev_A3_attenuation.py']:
  actual=sha(root/rel);expected=prov['code_sha256'][rel];source_checks.append({'path':str(root/rel),'expected':expected,'actual':actual,'pass':actual==expected})
 for name in ['n3_results.json','n3_demeter2_drug_coverage.tsv','n3_demeter2_pairs.tsv','n3_demeter2_target_results.tsv']:
  path=nd/name;expected=prov['generated_files_sha256'][str(path.relative_to(root))];actual=sha(path);source_checks.append({'path':str(path),'expected':expected,'actual':actual,'pass':actual==expected})
 source_checks.append({'path':str(nd/'run_config.yaml'),'expected':prov['run_config_sha256'],'actual':sha(nd/'run_config.yaml'),'pass':sha(nd/'run_config.yaml')==prov['run_config_sha256']})
 for rel,expected in original_cfg['frozen_table_sha256'].items():
  if rel.startswith('data/phase1b/') or rel=='data/INPUT_MANIFEST.json':
   actual=sha(root/rel);source_checks.append({'path':str(root/rel),'expected':expected,'actual':actual,'pass':actual==expected})
 save('frozen_source_checks.json',source_checks)
 if not all(x['pass'] for x in source_checks):raise RuntimeError('Frozen source hash mismatch')
 spec=importlib.util.spec_from_file_location('frozen_n3_matched_recovery',nd/'run_n3.py');n3=importlib.util.module_from_spec(spec);spec.loader.exec_module(n3)
 if n3.ROOT!=root:raise RuntimeError('Canonical frozen N3 root differs from requested root')
 raw_checks=[]
 for path in [n3.GDSC_AUC_PATH,n3.rc.gdsc2.CRISPR_PATH,n3.rc.gdsc2.MODEL_PATH,n3.DEMETER2_PATH]:
  records=[x for x in original_cfg['input_manifest']['files'] if Path(x['resolved_path'])==Path(path).resolve() or str(x['logical_path'])==str(path) or (not Path(x['logical_path']).is_absolute() and root/x['logical_path']==path)]
  if len(records)!=1:raise RuntimeError(f'No unique frozen input reference: {path}')
  expected=records[0]['sha256_observed'];actual=sha(path);raw_checks.append({'path':str(path),'expected':expected,'actual':actual,'pass':actual==expected})
 save('raw_input_checks.json',raw_checks)
 if not all(x['pass'] for x in raw_checks):raise RuntimeError('Canonical raw input hash mismatch')
 print('Frozen sources and four raw inputs PASS',flush=True)
 included=pd.read_csv(n3.ELIGIBILITY_PATH,sep='\t');included=included[included.status.eq('included')].copy()
 target_map=dict(zip(included.compound_id.astype(str),included.primary_target.astype(str)));names=dict(zip(included.compound_id,included.compound_name))
 meta={str(row.compound_id):{'scaffold_smiles':str(row.scaffold_smiles)} for row in pd.read_csv(n3.SCAFFOLDS_PATH,sep='\t').itertuples()}
 chronos=n3.rc.gdsc2.load_chronos();_,lineage=n3.rc._models_lineage();auc=pd.read_csv(n3.GDSC_AUC_PATH,index_col=0,low_memory=False);auc.index=auc.index.astype(str)
 cells=auc.index.intersection(chronos.index).intersection(lineage.index);lineage=lineage.reindex(cells)
 first=pd.read_csv(n3.DEMETER2_PATH,usecols=[0],dtype=str).iloc[:,0].astype(str);present=[];missing=[]
 for gene in sorted(set(target_map.values())):
  try:n3.a3.gene_column(first,gene);present.append(gene)
  except KeyError:missing.append(gene)
 if missing!=summary['demeter2_sensitivity']['missing_target_genes']:raise RuntimeError('Missing anchor set differs')
 d2_map,mapping=n3.a3.map_external_matrix(n3.DEMETER2_PATH,n3.MODEL_PATH,present)
 if mapping!=summary['demeter2_sensitivity']['mapping_diagnostics']:raise RuntimeError('DEMETER2 cell mapping diagnostics differ')
 oldcov=pd.read_csv(nd/'n3_demeter2_drug_coverage.tsv',sep='\t').set_index('compound_id')
 residuals={};cov=[];all_rows=[]
 for drug,target in target_map.items():
  response=n3.rc.gdsc2.response_vector(auc.loc[cells,drug]);dependency=-chronos.loc[cells,target].astype(float)
  external=d2_map[target].reindex(response.index) if target in d2_map else pd.Series(np.nan,index=response.index)
  frame=pd.concat([response.rename('R'),dependency.rename('Chronos'),external.rename('DEMETER2'),lineage],axis=1).dropna()
  n=int(len(frame));expected=int(oldcov.loc[drug,'n_common_R_Chronos_D2_lineage'])
  if n!=expected:raise RuntimeError(f'Exact matched-cell count mismatch for {drug}: {n}/{expected}')
  if n>=n3.MIN_N:
   idx=frame.index
   # Exactly the original matched Chronos OLS, preserving response standardization and row order.
   residual=n3.rc.linear_ols(response.reindex(idx),dependency.reindex(idx),lineage.reindex(idx)).reindex(response.index)
   status='ANALYZED'
  else:residual=pd.Series(np.nan,index=response.index);status='NA_missing_target_gene' if target not in d2_map else 'NA_n_below_200'
  if status!=oldcov.loc[drug,'status']:raise RuntimeError('Matched coverage status mismatch')
  residuals[drug]=residual;cov.append({'compound_id':drug,'target':target,'n_common':n,'n_frozen':expected,'status':status})
  all_rows.append(pd.DataFrame({'ModelID':response.index,'compound_id':drug,'target':target,'R':response.to_numpy(),'D_chronos':dependency.to_numpy(),'D_demeter2':external.to_numpy(),'E_chronos_matched':residual.to_numpy(),'keep_matched_complete_case':response.index.isin(frame.index)}))
 pd.DataFrame(cov).to_csv(out/'matched_drug_coverage.tsv',sep='\t',index=False)
 pd.concat(all_rows,ignore_index=True).to_csv(out/'matched_chronos_cell_vectors.tsv.gz',sep='\t',index=False,float_format='%.17g',compression='gzip')
 analyzed=sum(x['status']=='ANALYZED' for x in cov)
 if analyzed!=cfg['scope']['expected_analyzed_drugs']:raise RuntimeError('Analyzed-drug count differs')
 pairs=n3.pair_table(residuals,target_map,names)
 oldpairs=pd.read_csv(nd/'n3_demeter2_pairs.tsv',sep='\t')
 checks=pairs[['drug_a','drug_b','n']].merge(oldpairs[['drug_a','drug_b','n']],on=['drug_a','drug_b'],validate='one_to_one',suffixes=('_recovered','_frozen'),how='outer',indicator=True)
 checks['pass']=checks._merge.eq('both')&checks.n_recovered.eq(checks.n_frozen)
 checks.to_csv(out/'pair_cell_count_checks.tsv',sep='\t',index=False)
 if not checks['pass'].all():raise RuntimeError('Matched pair counts differ from frozen DEMETER2 cohort')
 targets=n3.target_estimands(pairs,target_map,cfg['scope']['expected_targets'],meta)
 frozen_targets=pd.read_csv(nd/'n3_demeter2_target_results.tsv',sep='\t')
 for key in ['target','n_drugs','n_scaffolds','within_pair_count','cross_pair_count','compound_ids']:
  if targets[key].tolist()!=frozen_targets[key].tolist():raise RuntimeError(f'Target denominator/membership differs: {key}')
 if not np.isfinite(targets.delta_t).all():raise RuntimeError('Not all six targets have finite values')
 # Do not call summarize_effect: it would rerun a bootstrap. Only recover the original median.
 median=float(np.median(targets.delta_t.to_numpy()));expected=float(summary['demeter2_sensitivity']['primary_median_delta_chronos_same_subset']);error=abs(median-expected)
 if error>cfg['verification']['absolute_tolerance']:raise RuntimeError(f'Overall frozen median mismatch: {median} vs {expected}')
 targets.to_csv(out/'matched_chronos_target_results.tsv',sep='\t',index=False,float_format='%.17g');pairs.to_csv(out/'matched_chronos_pairs.tsv',sep='\t',index=False,float_format='%.17g')
 paired=targets[['target','W_t','C_t','delta_t']].merge(frozen_targets[['target','W_t','C_t','delta_t']],on='target',validate='one_to_one',suffixes=('_chronos_matched','_demeter2_matched'))
 paired.to_csv(out/'panel_d_matched_target_source.tsv',sep='\t',index=False,float_format='%.17g')
 unchanged=all(sha(x['path'])==x['actual'] for x in source_checks)
 if not unchanged:raise RuntimeError('Frozen source mutated during recovery')
 r.update({'status':'PASS','end_utc':datetime.now(timezone.utc).isoformat(),'n_analyzed_drugs':analyzed,'n_focal_targets':len(targets),'pair_cell_count_checks':len(checks),'all_pair_cell_counts_match':True,'median_delta_recovered':median,'median_delta_frozen':expected,'absolute_error':error,'missing_target_genes':missing,'mapping_diagnostics_match':True,'target_denominators_match_demeter2':True,'frozen_sources_unchanged':unchanged,'n_fits_recovered':analyzed})
 save('recovery_report.json',r);print(json.dumps(r,indent=2),flush=True)
except Exception as exc:
 r.update({'status':'STOP','error':str(exc),'traceback':traceback.format_exc()});save('recovery_report.json',r);raise
