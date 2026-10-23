"""Recover frozen full-cohort N3 vectors; no new inferential analyses."""
from pathlib import Path
import os,json,importlib.util,hashlib,traceback
import numpy as np
import pandas as pd
import yaml
B=Path(__file__).resolve().parent
cfg=yaml.safe_load((B/'run_config.yaml').read_text());R=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));S=R/cfg['source_n3']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=[S/x for x in ['run_n3.py','run_config.yaml','n3_drug_results.tsv','n3_pairs_gdsc2.tsv','n3_target_results.tsv','n3_results.json']]
before={str(p):sha(p) for p in protected}
try:
 assert json.loads((B/'regression_gate.json').read_text())['checks_passed']==254
 spec=importlib.util.spec_from_file_location('frozen_n3',S/'run_n3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 inc,groups,targets,chron,lineage,cells,data,oldcfg=m.load_gdsc_inputs()
 tm={d:x['target'] for d,x in data.items()};names={d:x['compound_name'] for d,x in data.items()}
 observed=pd.DataFrame([{k:x[k] for k in ['compound_id','n','concordance_partial']} for x in data.values()])
 old=pd.read_csv(S/'n3_drug_results.tsv',sep='\t');j=observed.merge(old,on='compound_id',suffixes=('_new','_old'),validate='one_to_one')
 assert len(j)==43 and (j.n_new==j.n_old).all()
 drugerr=float(abs(j.concordance_partial_new-j.concordance_partial_old).max());assert drugerr<=cfg['recovery_tolerance']
 pairs=m.pair_table({d:x['E'] for d,x in data.items()},tm,names)
 oldpairs=pd.read_csv(S/'n3_pairs_gdsc2.tsv',sep='\t');j=pairs.merge(oldpairs,on=['drug_a','drug_b'],suffixes=('_new','_old'),validate='one_to_one')
 assert len(j)==903 and (j.n_new==j.n_old).all()
 pairerr=float(abs(j.rho_new-j.rho_old).max());assert pairerr<=cfg['recovery_tolerance']
 out=m.target_estimands(pairs,tm,targets,data);ot=pd.read_csv(S/'n3_target_results.tsv',sep='\t');j=out.merge(ot,on='target',suffixes=('_new','_old'),validate='one_to_one')
 errors={k:float(abs(j[k+'_new']-j[k+'_old']).max()) for k in ['W_t','C_t','delta_t']}
 assert all(e<=cfg['recovery_tolerance'] for e in errors.values())
 for k in ['n_drugs','n_scaffolds','within_pair_count','cross_pair_count']:assert (j[k+'_new']==j[k+'_old']).all()
 vectors=pd.concat([pd.DataFrame({'ModelID':cells,'compound_id':d,'target':x['target'],'R':x['response'].reindex(cells).values,'D':x['chronos'].reindex(cells).values,'E':x['E'].reindex(cells).values}) for d,x in data.items()],ignore_index=True)
 vectors['R_hat']=vectors.R-vectors.E;vectors['complete_case']=np.isfinite(vectors.E)
 oldv=pd.read_csv(R/'03_analysis/2026-10-15_figure1_panel_c_recovery/recovery_run1/n3_selected_cell_vectors.tsv',sep='\t')
 j=vectors.merge(oldv,on=['ModelID','compound_id'],suffixes=('_new','_old'),validate='one_to_one');ve=float(abs(j.E_new-j.E_old).max());assert ve<=cfg['recovery_tolerance']
 vectors.to_csv(B/'n3_full_cohort_vectors.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
 pairs.to_csv(B/'recovered_pairs_gdsc2.tsv',sep='\t',index=False);out.to_csv(B/'recovered_target_results.tsv',sep='\t',index=False);observed.to_csv(B/'recovered_drug_checks.tsv',sep='\t',index=False)
 assert before=={str(p):sha(p) for p in protected}
 report={'status':'PASS','drugs':43,'cells':len(cells),'pairs':903,'targets':len(targets),'tolerance':cfg['recovery_tolerance'],'max_concordance_error':drugerr,'max_pair_rho_error':pairerr,'target_errors':errors,'max_saved_two_drug_E_error':ve,'protected_sha256':before,'real_molecular_associations_run':False}
 (B/'RECOVERY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
except Exception:
 (B/'RECOVERY_STOP.txt').write_text(traceback.format_exc());raise
