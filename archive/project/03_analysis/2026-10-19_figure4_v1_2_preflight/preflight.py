"""Metadata-only singleton amendment and design geometry; no outcome regression."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,traceback,sys
import numpy as np
import pandas as pd
import yaml
B=Path(__file__).resolve().parent;cfg=yaml.safe_load((B/'run_config.yaml').read_text());R=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));A=R/cfg['assets'];P=R/cfg['previous_execution']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,rows):pd.DataFrame(rows).to_csv(B/name,sep='\t',index=False)
lock={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'stage':'before_real_design_preflight','sha256':{p.name:sha(p) for p in [Path(__file__),B/'run_config.yaml',B/'FIGURE4_ANALYSIS_DESIGN_v1.2.md']}}
(B/'PRE_EXECUTION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
try:
 assert json.loads((B/'regression_gate.json').read_text())['checks_passed']==254
 hashes={}
 for root in [A,P]:
  for line in (root/'SHA256SUMS.txt').read_text().splitlines():
   h,p=line.split('  ',1);assert sha(root/p)==h,f'Source hash mismatch: {root/p}';hashes[str(root/p)]=h
 models=Path(cfg['models']);expected=json.loads((P/'INPUT_INTEGRITY.json').read_text())['input_sha256'][str(models)];assert sha(models)==expected
 masks=pd.read_csv(A/'FINAL_CELL_MASKS.tsv',sep='\t');lin=pd.read_csv(models).drop_duplicates('ModelID').set_index('ModelID').OncotreeLineage
 scores=pd.read_csv(P/'HALLMARK_SCORES.tsv.gz',sep='\t',index_col='ModelID');gv=pd.read_csv(P/'GENERAL_SENSITIVITY_VALUES.tsv',sep='\t');v=pd.read_csv(A/'n3_full_cohort_vectors.tsv.gz',sep='\t');E=v.pivot(index='ModelID',columns='compound_id',values='E');D=v.pivot(index='ModelID',columns='compound_id',values='D')
 drugs=pd.read_csv(R/'03_analysis/2026-09-28_phase1b_target_expansion/n3_drug_results.tsv',sep='\t');ctrl=pd.read_csv(A/'NEGATIVE_CONTROLS.tsv',sep='\t');pan=pd.read_csv(A/'GENERAL_PANELS.tsv',sep='\t');pairs=pd.read_csv(A/'ADMISSIBLE_PAIRS.tsv',sep='\t');availability=pd.read_csv(A/'PROGRAM_AVAILABILITY_MASKS.tsv.gz',sep='\t')
 assert len(scores.columns)==50 and pan.target.nunique()==6 and pairs.admissible_pair.sum()==17
 expected_n={'EGFR':487,'MTOR':404,'ESR1':486,'TOP1':420,'ATR':487,'CHEK1':487};expected_removed={'ACH-000976','ACH-001401','ACH-001400'}
 audit=[];outcomes=[];diags=[];slots=[];cell_diags=[];scaling=[];linrows=[];counts=[]
 for row in pan.itertuples(index=False):
  t=row.target;old=sorted(masks.loc[masks.target.eq(t)&masks.primary_eligible,'ModelID']);assert len(old)==len(set(old));labels=lin.reindex(old);assert labels.notna().all();lc=labels.value_counts();keep=[c for c in old if lc[labels[c]]!=1];removed=set(old)-set(keep)
  assert removed<=expected_removed and len(keep)==expected_n[t]
  for c in old:audit.append({'target':t,'ModelID':c,'lineage':labels[c],'n_same_lineage_v1_1':int(lc[labels[c]]),'keep_v1_2':c in keep,'reason':'retained_non_singleton_lineage' if c in keep else 'singleton_lineage_in_frozen_target_mask'})
  counts.append({'target':t,'n_v1_1':len(old),'n_excluded':len(removed),'n_v1_2':len(keep),'excluded_IDs':'|'.join(sorted(removed))})
  labels=labels.reindex(keep);lc2=labels.value_counts();assert (lc2>=2).all();assert sum(lc2>=20)>=2
  for l,n in lc2.items():linrows.append({'target':t,'lineage':l,'n_v1_2':int(n)})
  coding=pd.get_dummies(labels,dtype=float).reindex(columns=sorted(labels.unique()));L=coding.iloc[:,1:].to_numpy()
  focal=drugs.loc[drugs.target.eq(t),'compound_id'].tolist();nc=ctrl.loc[ctrl.target.eq(t),'compound_id'].tolist();all_d=focal+nc
  assert len(focal)>=2 and drugs.loc[drugs.compound_id.isin(focal),'scaffold_smiles'].nunique()>=2 and len(nc)==2
  assert not set(nc)&set(row.compound_ids.split('|')) and not set(focal)&set(row.compound_ids.split('|'))
  subG=gv[gv.target.eq(t)].set_index('ModelID').reindex(keep);assert (subG.n_valid_backgrounds>=row.required_valid_backgrounds).all();G=subG.G;assert np.isfinite(G).all() and G.std(ddof=0)>=cfg['minimum_sd']
  dep=D.reindex(keep)[focal[0]];assert np.isfinite(dep).all() and dep.std(ddof=0)>=cfg['minimum_sd']
  for d in focal:assert np.array_equal(D.reindex(keep)[d].values,dep.values)
  for d in all_d:
   y=E.reindex(keep)[d];ok=bool(np.isfinite(y).all() and y.std(ddof=0)>=cfg['minimum_sd']);outcomes.append({'target':t,'compound_id':d,'role':'focal' if d in focal else 'control','n':len(keep),'finite_count':int(np.isfinite(y).sum()),'E_mean':float(y.mean()),'E_sd_ddof0':float(y.std(ddof=0)),'status':'PASS' if ok else 'FAIL','all_50_programs_same_rows':True});assert ok
  for p in sorted(scores.columns):
   av=availability[availability.target.eq(t)&availability.program.eq(p)].set_index('ModelID').reindex(keep);assert av.gene_coverage_eligible.all() and av.gene_coverage_eligible.notna().all()
   s=scores.reindex(keep)[p];assert np.isfinite(s).all() and s.std(ddof=0)>=cfg['minimum_sd'];z=(s-s.mean())/s.std(ddof=0)
   scaling.append({'target':t,'program':p,'n':len(keep),'score_mean':float(s.mean()),'score_sd_ddof0':float(s.std(ddof=0)),'G_sd':float(G.std(ddof=0)),'D_sd':float(dep.std(ddof=0))})
   base=np.column_stack([np.ones(len(keep)),dep,L]);designs={'G_adjusted':np.column_stack([base,G,z]),'no_G_same_rows':np.column_stack([base,z])};hvals={}
   for model,X in designs.items():
    assert np.isfinite(X).all();rank=int(np.linalg.matrix_rank(X,tol=cfg['rank_tolerance']));Q=np.linalg.qr(X,mode='reduced')[0];h=np.sum(Q*Q,axis=1);hvals[model]=h
    passed=rank==X.shape[1] and len(keep)-rank>0 and max(h)<cfg['leverage_upper_exclusive'] and len(keep)>=cfg['minimum_cells']
    diags.append({'target':t,'program':p,'model':model,'n':len(keep),'columns':X.shape[1],'rank':rank,'df_residual':len(keep)-rank,'max_leverage':float(max(h)),'n_leverage_ge_095':int((h>=.95).sum()),'reference_lineage':coding.columns[0],'status':'PASS' if passed else 'FAIL'})
    assert passed,f'Design gate failed {t}/{p}/{model}'
   for i,c in enumerate(keep):cell_diags.append({'target':t,'program':p,'ModelID':c,'lineage':labels[c],'leverage_G':hvals['G_adjusted'][i],'leverage_no_G':hvals['no_G_same_rows'][i]})
   slots.append({'target':t,'program':p,'n_v1_2':len(keep),'n_focal':len(focal),'n_controls':len(nc),'n_admissible_pairs':int(pairs.loc[pairs.target.eq(t),'admissible_pair'].sum()),'same_cells_all_outcomes_and_models':True,'status':'PASS','association_status':'NOT_RUN_NOT_AUTHORIZED'})
  print(t,len(old),'->',len(keep),'PASS all 50',flush=True)
 assert len(slots)==300 and len(diags)==600
 write('SAMPLE_EXCLUSION_AUDIT.tsv',audit);write('EXCLUDED_SAMPLES.tsv',[r for r in audit if not r['keep_v1_2']]);write('REVISED_CELL_MASK.tsv',audit);write('SAMPLE_COUNTS.tsv',counts);write('LINEAGE_COUNTS.tsv',linrows);write('OUTCOME_QUALITY.tsv',outcomes);write('DESIGN_DIAGNOSTICS.tsv',diags);write('ELIGIBILITY_300.tsv',slots);write('ANALYSIS_SCALING.tsv',scaling)
 pd.DataFrame(cell_diags).to_csv(B/'CELL_LEVERAGE.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
 assert all(sha(Path(p))==h for p,h in hashes.items())
 report={'status':'PASS','preflight_only':True,'slots_passed':len(slots),'designs_passed':len(diags),'excluded_target_cell_rows':sum(not r['keep_v1_2'] for r in audit),'excluded_unique_ids':sorted({r['ModelID'] for r in audit if not r['keep_v1_2']}),'sample_counts':counts,'max_leverage_by_model':{m:max(r['max_leverage'] for r in diags if r['model']==m) for m in ['G_adjusted','no_G_same_rows']},'source_files_verified':len(hashes),'source_assets_unchanged':True,'association_fits':0,'bootstrap_draws':0,'biological_p_q_computed':False,'real_association_execution':'HOLD_this_round','timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
 (B/'PREFLIGHT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');(B/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps(report,indent=2))
except Exception:
 for name,rows in [('PARTIAL_DESIGN_DIAGNOSTICS.tsv',locals().get('diags',[])),('PARTIAL_EXCLUSION_AUDIT.tsv',locals().get('audit',[]))]:write(name,rows)
 (B/'TECHNICAL_STOP.txt').write_text(traceback.format_exc());raise
