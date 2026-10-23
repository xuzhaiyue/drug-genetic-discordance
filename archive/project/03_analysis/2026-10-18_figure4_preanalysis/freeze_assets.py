"""Freeze missingness-only controls and GMT membership; never compute program scores."""
from pathlib import Path
import os,json,hashlib,traceback
import numpy as np
import pandas as pd
import yaml
B=Path(__file__).resolve().parent;cfg=yaml.safe_load((B/'run_config.yaml').read_text());R=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));A=R/cfg['source_audit']
try:
 p=R/'revision/figure3_predictive_audit/2026-10-16/build_eligibility.py';ctx={'__file__':str(p),'__name__':'coverage_only'};exec(compile(p.read_text().split('# Proposed common folds')[0],str(p),'exec'),ctx)
 G=ctx['G'];meta=ctx['meta'];ids=ctx['ids'];review=ctx['review'];known=ctx['known']
 groups=pd.read_csv(R/'data/phase1b/PHASE1B_TARGET_GROUPS.tsv',sep='\t');groups=groups[groups.target_group_eligible]
 rna=pd.read_csv(cfg['rna'],low_memory=False);rna=rna[rna.IsDefaultEntryForModel.eq('Yes')].drop_duplicates('ModelID').set_index('ModelID')
 metadata={'Unnamed: 0','SequencingID','ModelConditionID','IsDefaultEntryForMC','IsDefaultEntryForModel'}
 X=rna[[c for c in rna if c not in metadata]].astype(float);X.columns=[c.split(' ')[0] for c in X];X=X.loc[:,~X.columns.duplicated()]
 rq=pd.read_csv(A/'RNA_ROW_COVERAGE_QC.tsv',sep='\t');rna_ok=set(rq.loc[rq.keep_proposed_QC,'ModelID'])
 assert json.loads((B/'RECOVERY_REPORT.json').read_text())['status']=='PASS'
 n3_cells=set(pd.read_csv(B/'n3_full_cohort_vectors.tsv.gz',sep='\t',usecols=['ModelID']).ModelID)
 assert len(n3_cells)==590
 reference=sorted(n3_cells&known&rna_ok)
 Z=X.reindex(reference);finite=np.isfinite(Z).mean();sd=Z.std(ddof=0);geneok=(finite>=cfg['rna_finite_fraction'])&(sd>=cfg['gene_minimum_sd'])
 geneqc=pd.DataFrame({'symbol':X.columns,'finite_fraction_reference':finite.values,'sd_reference':sd.values,'eligible':geneok.values});geneqc.to_csv(B/'RNA_GENE_QC.tsv',sep='\t',index=False)
 members=[];programs={}
 for line in (B/'assets/h.all.v2026.1.Hs.symbols.gmt').read_text().splitlines():
  fields=line.split('\t');name=fields[0];genes=sorted(set(fields[2:]));ret=[g for g in genes if g in geneok.index and geneok[g]]
  programs[name]=ret
  for g in genes:members.append({'program':name,'symbol':g,'present_in_expression':g in X.columns,'retained':g in ret})
 assert len(programs)==50
 pd.DataFrame(members).to_csv(B/'HALLMARK_GENE_MEMBERSHIP.tsv',sep='\t',index=False)
 pstats=[]
 for name,ret in programs.items():
  original=sum(r['program']==name for r in members);passed=len(ret)>=cfg['program_minimum_genes'] and len(ret)>=cfg['program_minimum_fraction']*original
  pstats.append({'program':name,'n_original':original,'n_retained':len(ret),'eligible':passed})
 assert all(r['eligible'] for r in pstats),'Hallmark membership QC failure; STOP'
 pd.DataFrame(pstats).to_csv(B/'HALLMARK_QC.tsv',sep='\t',index=False)
 controls=[];candidates=[];panels=[];masks=[];pmasks=[];coverage=[]
 def tok(d,k):return set(filter(None,str(review.at[d,k]).split('|')))
 for gr in groups.itertuples(index=False):
  t=gr.primary_target;focal=gr.compound_ids.split('|');common=set.intersection(*(G[d]&known for d in focal));base=common&rna_ok
  pool=[d for d in ids if meta.at[d,'target']!=t and all(not(tok(d,'mechanism_tokens')&tok(a,'mechanism_tokens')) and not(tok(d,'pathway_tokens')&tok(a,'pathway_tokens')) for a in focal)]
  ranked=sorted(pool,key=lambda d:(-len(base&G[d]),d));selected=[];labels=set()
  for d in ranked:
   n=len(base&G[d]);eligible=n>=cfg['minimum_cells'] and n/len(base)>=cfg['control_minimum_coverage']
   take=eligible and len(selected)<2 and meta.at[d,'target'] not in labels
   if take:selected.append(d);labels.add(meta.at[d,'target'])
   candidates.append({'target':t,'compound_id':d,'control_target':meta.at[d,'target'],'n_joint_base':n,'fraction':n/len(base),'eligible':eligible,'selected':take})
  assert len(selected)==2,f'No two qualified distinct controls: {t}'
  panel=[d for d in pool if d not in selected];minimum=max(cfg['general_minimum_drugs'],int(np.ceil(cfg['general_minimum_fraction']*len(panel))))
  assert len(panel)>=5
  valid={c for c in base if sum(c in G[d] for d in panel)>=minimum}
  assert len(valid)>=200
  for d in selected:controls.append({'target':t,'compound_id':d,'compound_name':meta.at[d,'compound_name'],'control_target':meta.at[d,'target'],'selection':'joint finite coverage descending, ID ascending; distinct target labels; no response values','n_on_final_primary':len(valid&G[d]),'n_final_primary':len(valid)})
  panels.append({'target':t,'compound_ids':'|'.join(panel),'n_drugs':len(panel),'required_valid_backgrounds':minimum,'excluded_control_ids':'|'.join(selected)})
  for c in sorted(common):masks.append({'target':t,'ModelID':c,'RNA_QC95':c in rna_ok,'n_valid_background':sum(c in G[d] for d in panel),'primary_eligible':c in valid,'specificity_matched':c in valid and all(c in G[d] for d in selected)})
  for name,ret in programs.items():
   ok=np.isfinite(X.reindex(sorted(valid))[ret]).mean(axis=1)>=cfg['program_per_cell_fraction']
   for c,keep in ok.items():pmasks.append({'target':t,'program':name,'ModelID':c,'gene_coverage_eligible':bool(keep)})
   coverage.append({'target':t,'program':name,'n_primary_before_program_QC':len(valid),'n_after_program_QC':int(ok.sum()),'n_specificity_matched':sum(bool(keep) and all(c in G[d] for d in selected) for c,keep in ok.items())})
 for name,rows in [('NEGATIVE_CONTROLS',controls),('CONTROL_CANDIDATES',candidates),('GENERAL_PANELS',panels),('FINAL_CELL_MASKS',masks),('PROGRAM_COVERAGE',coverage)]:pd.DataFrame(rows).to_csv(B/(name+'.tsv'),sep='\t',index=False)
 pd.DataFrame(pmasks).to_csv(B/'PROGRAM_AVAILABILITY_MASKS.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
 assert min(x['n_after_program_QC'] for x in coverage)>=200
 info={'status':'PASS','programs':50,'reference_cells':len(reference),'feature_QC_only_no_scores':True,'gmt_sha256':hashlib.sha256((B/'assets/h.all.v2026.1.Hs.symbols.gmt').read_bytes()).hexdigest(),'url':cfg['hallmark_url'],'license_url':cfg['license_url'],'acquired_date':'2026-10-18','controls':controls,'coverage_by_target':pd.DataFrame(coverage).groupby('target').n_after_program_QC.agg(['min','max']).reset_index().to_dict('records')}
 (B/'ASSET_FREEZE_REPORT.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
except Exception:
 (B/'ASSET_STOP.txt').write_text(traceback.format_exc());raise
