"""Execute approved R2 masks on validated S0 pairs; no new permutation tests."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();root=args.project_root.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if list(out.iterdir()):raise RuntimeError('Use a new empty output directory')
    state={'status':'RUNNING','scope':'approved R2 S1–S11; S0 baseline unchanged',
           'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed':[]}
    try:
        import numpy as np
        import pandas as pd
        import yaml
        import platform
        from importlib.metadata import version
        run=Path(__file__).parent
        cfg=yaml.safe_load((run/'run_config.yaml').read_text())
        state['environment']={'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','pandas','scipy','PyYAML']}}
        lock=json.loads((run/'input_lock.json').read_text())
        for rel,expected in lock.items():
            if sha(root/rel)!=expected:raise RuntimeError(f'Locked source changed: {rel}')
        r2=root/cfg['r2_directory'];baseline=root/cfg['baseline_directory']
        for name,expected in json.loads((r2/'SHA256.json').read_text()).items():
            if sha(r2/name)!=expected:raise RuntimeError(f'R2 package mismatch: {name}')
        for resource in ['gdsc2','prism']:
            report=json.loads((baseline/'GATE_C_REPORT.json').read_text())
            if report['status']!='PASS_GATE_C_ONLY':raise RuntimeError('Gate C not PASS')
        raw_checks=[]
        for row in json.loads((root/'data/INPUT_MANIFEST.json').read_text())['files']:
            path=Path(row['logical_path']);current=sha(path)
            ok=current==row['sha256'] and path.stat().st_size==row['bytes']
            raw_checks.append({'path':str(path),'resolved_path':str(path.resolve()),'sha256':current,'pass':ok})
            (out/'raw_input_checks.json').write_text(json.dumps(raw_checks,indent=2)+'\n')
            if not ok:raise RuntimeError(f'Raw input hash mismatch: {path}')
        masks=pd.read_csv(r2/'scenario_pair_masks.tsv',sep='\t').fillna('')
        membership=pd.read_csv(r2/'scenario_drug_membership.tsv',sep='\t').fillna('')
        losses=pd.read_csv(root/cfg['loss_reason_metadata'],sep='\t').fillna('')
        targets=cfg['focal_targets'];summaries=[];all_targets=[];loo=[]
        for scenario in cfg['scenario_order']:
            idx=int(scenario.split('_')[0][1:])
            for resource in ['GDSC2','PRISM']:
                label=resource.lower();folder=out/scenario/label;folder.mkdir(parents=True)
                pairs=pd.read_csv(baseline/f'{label}_pairs.tsv',sep='\t')
                s0targets=pd.read_csv(baseline/f'{label}_targets.tsv',sep='\t').set_index('target')
                members=membership[(membership.scenario==scenario)&(membership.resource==resource)].copy()
                members.to_csv(folder/'drug_membership.tsv',sep='\t',index=False)
                retained=set(members.loc[members.retained.astype(str).str.lower().eq('true'),'compound_id'])
                original=set(pairs.drug_a)|set(pairs.drug_b)
                removed=original-retained
                if not retained<=original:raise RuntimeError('Unexpected new drug')
                valid_drugs=pd.read_csv(baseline/f'{label}_drugs.tsv',sep='\t')
                if (valid_drugs[valid_drugs.compound_id.isin(retained)].n<200).any():raise RuntimeError('Drug MIN_N failure')
                component=[];pair_outputs=[]
                for target in targets:
                    block=pairs.copy()
                    within=(block.target_a==target)&(block.target_b==target)
                    cross=(block.target_a==target)^(block.target_b==target)
                    block=block.loc[within|cross].copy()
                    block['pair_role']=np.where((block.target_a==target)&(block.target_b==target),'within','cross')
                    block['identity_removed']=block.drug_a.isin(removed)|block.drug_b.isin(removed)
                    meta=masks[(masks.scenario==scenario)&(masks.resource==resource)&(masks.target==target)]
                    if meta.duplicated(['drug_a','drug_b']).any():raise RuntimeError('Duplicate pair masks')
                    lookup=meta.set_index(['drug_a','drug_b'])
                    reasons=[]
                    for row in block.itertuples():
                        if row.identity_removed:reasons.append('identity_exclusion');continue
                        key=(row.drug_a,row.drug_b)
                        if key not in lookup.index:raise RuntimeError(f'Missing mask: {scenario} {resource} {target} {key}')
                        reasons.append(str(lookup.loc[key,'excluded_reason']))
                    block['metadata_exclusion_reason']=reasons
                    block['metadata_allowed']=block.metadata_exclusion_reason.eq('')
                    block['finite_n30']=np.isfinite(block.rho)&block.n.ge(30)
                    block['allowed']=block.metadata_allowed&block.finite_n30
                    block['final_exclusion_reason']=np.where(block.metadata_allowed & ~block.finite_n30,'no_finite_n30',block.metadata_exclusion_reason)
                    block['focal_target']=target
                    block['scenario']=scenario;block['resource']=resource
                    pair_outputs.append(block)
                    w=block.loc[block.allowed & block.pair_role.eq('within')]
                    c=block.loc[block.allowed & block.pair_role.eq('cross')]
                    meta_loss=losses[(losses.scenario==scenario)&(losses.resource==resource)&(losses.target==target)]
                    if len(meta_loss)!=1:raise RuntimeError('Missing target-loss metadata')
                    structural=meta_loss.iloc[0]
                    reasons=[]
                    for flag in ['n_drugs_lt2','n_distinct_scaffolds_lt2','no_allowed_within_pair']:
                        if str(structural[flag]).lower()=='true':reasons.append(flag)
                    if len(w)==0:reasons.append('no_finite_within_n30')
                    if len(c)==0:reasons.append('no_finite_cross_n30')
                    eligible=not reasons
                    W=float(w.rho.median()) if len(w) else None
                    C=float(c.rho.median()) if len(c) else None
                    delta=W-C if W is not None and C is not None else None
                    old=s0targets.loc[target] if target in s0targets.index else None
                    if scenario in ['S9_narrow_controls','S10_broad_controls'] and old is not None:
                        if not np.isclose(W,old.W_t,atol=1e-8,rtol=0):raise RuntimeError('S9/S10 within sharing changed')
                    row={'scenario':scenario,'resource':resource,'target':target,'eligible':eligible,
                         'n_drugs':int(structural.n_drugs),'n_scaffolds':int(structural.n_scaffolds),
                         'retained_compound_ids':structural.compound_ids,'lost_reason':';'.join(reasons),
                         'n_drugs_lt2':str(structural.n_drugs_lt2).lower()=='true',
                         'n_distinct_scaffolds_lt2':str(structural.n_distinct_scaffolds_lt2).lower()=='true',
                         'no_allowed_within_pair':str(structural.no_allowed_within_pair).lower()=='true',
                         'no_finite_within_n30':len(w)==0,'no_finite_cross_n30':len(c)==0,
                         'W_t':W,'C_t':C,'delta_t':delta,'within_pair_count':len(w),'cross_pair_count':len(c),
                         'S0_W_t':float(old.W_t) if old is not None else None,
                         'S0_C_t':float(old.C_t) if old is not None else None,
                         'S0_delta_t':float(old.delta_t) if old is not None else None,
                         'S0_within_pair_count':int(old.within_pair_count) if old is not None else None,
                         'S0_cross_pair_count':int(old.cross_pair_count) if old is not None else None,
                         'change_W':W-float(old.W_t) if W is not None and old is not None else None,
                         'change_C':C-float(old.C_t) if C is not None and old is not None else None,
                         'change_delta':delta-float(old.delta_t) if delta is not None and old is not None else None,
                         'cross_removed_identity':int((block.pair_role.eq('cross')&block.identity_removed).sum()),
                         'cross_removed_mechanism':int((block.pair_role.eq('cross') & ~block.identity_removed & ~block.metadata_allowed).sum()),
                         'cross_removed_cell_gate':int((block.pair_role.eq('cross')&block.metadata_allowed&~block.finite_n30).sum())}
                    component.append(row)
                pd.concat(pair_outputs,ignore_index=True).to_csv(folder/'complete_target_pair_masks.tsv',sep='\t',index=False)
                target_table=pd.DataFrame(component);target_table.to_csv(folder/'targets.tsv',sep='\t',index=False)
                all_targets.extend(component)
                eligible=target_table[target_table.eligible].copy();deltas=eligible.delta_t.to_numpy(dtype=float)
                if len(deltas) and not np.isfinite(deltas).all():raise RuntimeError('Nonfinite eligible target effect')
                median=float(np.median(deltas)) if len(deltas) else None
                common=s0targets.loc[eligible.target,'delta_t'].to_numpy(dtype=float)
                ref=float(np.median(common)) if len(common) else None
                seed=20260927+1000*idx+(0 if resource=='GDSC2' else 100)
                ci=[None,None]
                if len(deltas)>=4:
                    rng=np.random.default_rng(seed)
                    draws=np.median(deltas[rng.integers(0,len(deltas),size=(10000,len(deltas)))],axis=1)
                    ci=list(map(float,np.quantile(draws,[.025,.975])))
                    pd.DataFrame({'draw':np.arange(1,10001),'median_delta':draws}).to_csv(folder/'target_bootstrap_draws.tsv',sep='\t',index=False)
                    status='DESCRIPTIVELY_CONSISTENT' if ci[0]>0 and float(np.mean(deltas>0))>=.70 else 'DESCRIPTIVELY_INCONSISTENT'
                else:status='COVERAGE_INSUFFICIENT'
                for i,target in enumerate(eligible.target):
                    vals=np.delete(deltas,i)
                    loo.append({'scenario':scenario,'resource':resource,'omitted_target':target,'n_remaining':len(vals),
                                'median_delta':float(np.median(vals)) if len(vals) else None,'meaning':'descriptive_only'})
                summary={'scenario':scenario,'resource':resource,'n_retained_drugs':len(retained),'n_eligible_targets':len(deltas),
                         'targets':';'.join(eligible.target),'within_pairs':int(eligible.within_pair_count.sum()),
                         'cross_pairs_summed_by_target':int(eligible.cross_pair_count.sum()),'median_delta':median,
                         'ci_low':ci[0],'ci_high':ci[1],'positive_targets':int((deltas>0).sum()),
                         'positive_fraction':float(np.mean(deltas>0)) if len(deltas) else None,
                         'S0_common_target_median_delta':ref,'common_set_absolute_change':median-ref if median is not None and ref is not None else None,
                         'common_set_retention_ratio':median/ref if median is not None and ref is not None and ref>0 else None,
                         'seed':seed,'bootstrap_draws':10000 if len(deltas)>=4 else 0,'status':status,
                         'analysis_role':'contrast_definition_stress_test' if idx in [9,10] else 'pharmacology_sensitivity',
                         'lost_targets':';'.join(target_table.loc[~target_table.eligible,'target'])}
                summaries.append(summary);(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(json.dumps(summary),flush=True);state['completed'].append(scenario+'.'+resource)
                pd.DataFrame(summaries).to_csv(out/'scenario_summary.tsv',sep='\t',index=False)
                pd.DataFrame(all_targets).to_csv(out/'all_target_results.tsv',sep='\t',index=False)
                pd.DataFrame(loo).to_csv(out/'leave_one_target_out.tsv',sep='\t',index=False)
        after={rel:sha(root/rel) for rel in lock}
        (out/'source_hashes_after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=lock:raise RuntimeError('Locked original source changed during run')
        state['status']='COMPLETE_ALL_22_SCENARIO_RESOURCE_RUNS'
    except Exception as error:
        state['status']='STOP';state['error']=repr(error);state['traceback']=traceback.format_exc();raise
    finally:
        state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        (out/'execution_report.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
