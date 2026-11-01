"""Draw Figure 2 (panels a–e) from saved outputs; never fit or infer."""
import os,json,hashlib,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
CFG_PATH=Path(__file__).with_name('drawing_config.yaml');cfg=yaml.safe_load(CFG_PATH.read_text())
root=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));out=root/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
nd=root/cfg['sources']['n3_dir'];sd=root/cfg['sources']['sensitivity_dir'];order=cfg['figure']['target_order']
g=pd.read_csv(nd/'n3_target_results.tsv',sep='\t').set_index('target').loc[order]
p=pd.read_csv(nd/'n3_prism_targets.tsv',sep='\t').set_index('target')
all_t=pd.read_csv(sd/'all_target_results.tsv',sep='\t');sc=pd.read_csv(sd/'scenario_summary.tsv',sep='\t')
summary=json.loads((nd/'n3_results.json').read_text());spline=pd.read_csv(nd/'n3_spline_target_results.tsv',sep='\t').set_index('target').loc[order]
strict=all_t[all_t.scenario.eq('S11_strict_combined')].copy();sg=strict[strict.resource.eq('GDSC2')].set_index('target').loc[order]
colors=dict(zip(order,['#2878A6','#D18232','#499783','#9B6C9F','#A68B38','#668396']))
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=np.array(cfg['figure']['size_mm'])/25.4,facecolor='white')
def heading(letter,title,x,y):
 fig.text(x,y,letter,weight='bold',fontsize=10,va='top');fig.text(x+.032,y-.001,title,fontsize=9.2,va='top',weight='medium')
def clean(ax):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.6)

# A: preserve every predefined scenario, resource, and focal target.
heading('a','Sensitivity to pharmacological restrictions',.045,.982)
scenarios=['S0']+list(sc[sc.resource.eq('GDSC2')].scenario)
heat=np.full((13,12),np.nan);heat_rows=[]
for block,res in enumerate(['GDSC2','PRISM']):
 base=g if res=='GDSC2' else p
 for i,t in enumerate(order):
  y=block*7+i
  for j,s in enumerate(scenarios):
   if s=='S0':
    eligible=t in base.index;value=float(base.loc[t,'delta_t']) if eligible else np.nan
    reason='' if eligible else 'baseline_group_not_qualifying'
   else:
    row=all_t[(all_t.resource==res)&(all_t.target==t)&(all_t.scenario==s)].iloc[0]
    eligible=bool(row.eligible);value=float(row.delta_t) if eligible else np.nan;reason=str(row.lost_reason) if not eligible else ''
   heat[y,j]=value
   heat_rows.append({'resource':res,'target':t,'scenario':s,'eligible':eligible,'delta_t':value,'lost_reason':reason})
pd.DataFrame(heat_rows).to_csv(out/'panel_a_source.tsv',sep='\t',index=False)
a=fig.add_axes([.17,.678,.64,.235])
cmap=LinearSegmentedColormap.from_list('delta',['#F4F7F9','#9BBFD0','#2F6988']);cmap.set_bad('#EFF1F3')
norm=Normalize(*cfg['figure']['sensitivity_color_range'])
a.pcolormesh(np.arange(13)-.5,np.arange(14)-.5,np.ma.masked_invalid(heat),cmap=cmap,norm=norm,shading='flat',rasterized=False)
a.set(xlim=(-.5,11.5),ylim=(12.5,-.5))
a.set(xticks=range(12),xticklabels=[f'S{i}\n'+name for i,name in enumerate(['Original','Scaff.','−BEZ','−AZD','Parent','−IRIN','−TAM','Core','Multi','Narrow','Broad','Strict'])],yticks=list(range(6))+list(range(7,13)),yticklabels=order+order)
a.xaxis.tick_top();a.tick_params(axis='both',length=0,pad=4);a.tick_params(axis='y',labelsize=7.2);a.tick_params(axis='x',labelsize=6.5)
for label,t in zip(a.get_yticklabels(),order+order):label.set_color(colors[t])
for sp in a.spines.values():sp.set_visible(False)
for y in list(range(6))+list(range(7,13)):
 for x in range(12):
  a.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='none',edgecolor='white',lw=.6))
  if not np.isfinite(heat[y,x]):a.text(x,y,'—',ha='center',va='center',color='#98A3AB',fontsize=7)
a.add_patch(Rectangle((-.5,5.5),12,1,facecolor='white',edgecolor='none'))
for block,res in enumerate(['GDSC2','PRISM']):
 for j,s in enumerate(scenarios):
  n=(6 if res=='GDSC2' else 4) if s=='S0' else int(sc[(sc.resource==res)&(sc.scenario==s)].iloc[0].n_eligible_targets)
  text=str(n)+('†' if res=='PRISM' and n<4 else '')
  a.text(j,6 if block==0 else 13.3,text,ha='center',va='center',fontsize=6.5,color='#647784',clip_on=False)
fig.text(.045,.856,'GDSC2',rotation=90,fontsize=7.5,va='center',weight='medium')
fig.text(.045,.730,'PRISM',rotation=90,fontsize=7.5,va='center',weight='medium')
a.text(-1.6,6,'Targets',ha='right',va='center',fontsize=6.5,color='#647784',clip_on=False)
a.text(-1.6,13.3,'Targets',ha='right',va='center',fontsize=6.5,color='#647784',clip_on=False)
a.add_patch(Rectangle((7.5,-.5),1,13,facecolor='none',edgecolor='#A2B1BB',lw=.7))
a.add_patch(Rectangle((10.5,-.5),1,13,facecolor='none',edgecolor='#526D7D',lw=1.0))
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.855,.752,.014,.15]),ticks=[0,.2,.4,.6])
cb.solids.set_rasterized(False)
cb.solids.set_edgecolor('face')
cb.ax.tick_params(labelsize=6.5,width=.6,length=2);cb.outline.set_linewidth(.6);cb.set_label('Δ = W − C',fontsize=7)
fig.text(.045,.642,'S0–S11: related sensitivity scenarios, not independent replications.',fontsize=6.5)
fig.text(.045,.628,'PRISM S11: 2 targets · descriptive only',fontsize=7,fontweight='semibold')
fig.text(.045,.615,'—: ineligible. † S5/S6/S7/S11: fewer than 4 PRISM targets; coverage insufficient.',fontsize=6.5,color='#647784')

# B: expose W and C separately rather than only their difference.
heading('b','Strict filtering: within- and cross-target sharing in GDSC2',.045,.596)
fig.text(.132,.558,'Within-target W',fontsize=8,weight='medium');fig.text(.610,.558,'Cross-target C',fontsize=8,weight='medium')
axes=[]
for k,metric in enumerate(['W_t','C_t']):
 ax=fig.add_axes([.13 if k==0 else .61,.370,.24 if k==0 else .27,.172]);clean(ax)
 ax.axhspan(.65,1.35,color='#FBF2E8',zorder=0)
 for i,t in enumerate(order):
  old=float(g.loc[t,metric]);ax.scatter(old,i-.10,s=27,facecolor='white',edgecolor=colors[t],linewidth=.9,zorder=3)
  if bool(sg.loc[t,'eligible']):
   new=float(sg.loc[t,metric]);ax.plot([old,new],[i-.10,i+.10],lw=1.6 if t=='MTOR' else .9,color=colors[t]);ax.scatter(new,i+.10,s=27,color=colors[t],zorder=4)
  else:ax.text(.51,i,'—',color='#98A3AB',ha='center',va='center',fontsize=8)
 ax.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xlim=cfg['figure']['rho_axis_range'],xticks=[0,.2,.4,.6]);ax.set_xlabel('Residual Spearman ρ')
 for lab,t in zip(ax.get_yticklabels(),order):
  lab.set_color(colors[t]);lab.set_weight('semibold' if t=='MTOR' else 'normal')
 axes.append(ax)
wb=axes[0]
wb.text(1.29,-.99,'Pairs\nS0 → S11',ha='center',transform=wb.get_yaxis_transform(),fontsize=6.5,linespacing=1.1)
for i,t in enumerate(order):
 text=f"{int(g.loc[t,'within_pair_count'])} → "+(str(int(sg.loc[t,'within_pair_count'])) if bool(sg.loc[t,'eligible']) else '—')
 wb.text(1.29,i,text,ha='center',va='center',transform=wb.get_yaxis_transform(),fontsize=7.2 if t=='MTOR' else 6.5,fontweight='semibold' if t=='MTOR' else 'normal')
wb.text(.25,1.53,f"W: {g.loc['MTOR','W_t']:.3f} → {sg.loc['MTOR','W_t']:.3f}",fontsize=6.5,color=colors['MTOR'],va='center')
wb.scatter([],[],s=23,facecolor='white',edgecolor='#53616A',label='S0');wb.scatter([],[],s=23,color='#53616A',label='S11')
wb.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(2.05,1.16),fontsize=7,handletextpad=.3,columnspacing=.8)
s11=sc[(sc.resource=='GDSC2')&sc.scenario.eq('S11_strict_combined')].iloc[0]
fig.text(.045,.312,f"Same four targets: median Δ {s11.S0_common_target_median_delta:.3f} → {s11.median_delta:.3f}; TOP1 and CHEK1 lose eligibility.",fontsize=7)
strict.to_csv(out/'panel_b_S11_source.tsv',sep='\t',index=False);g.reset_index().to_csv(out/'panel_b_S0_source.tsv',sep='\t',index=False)

# C: only the N3 GDSC2 spline check, not historical Arena rank/decile checks.
heading('c','Nonlinear residualization',.045,.286)
c=fig.add_axes([.105,.085,.195,.153]);clean(c)
for i,t in enumerate(order):
 old=float(g.loc[t,'delta_t']);new=float(spline.loc[t,'delta_t'])
 c.plot([old,new],[i-.10,i+.10],color=colors[t],lw=.9);c.scatter(old,i-.10,s=26,facecolor='white',edgecolor=colors[t],linewidth=.9,zorder=2);c.scatter(new,i+.10,s=28,color=colors[t],marker='s',zorder=3)
c.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xlim=cfg['figure']['delta_axis_range'],xticks=[0,.2,.4,.6]);c.set_xlabel('Δ = W − C')
c.scatter([],[],s=22,facecolor='white',edgecolor='#53616A',label='Linear');c.scatter([],[],s=24,color='#53616A',marker='s',label='Spline')
c.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.25,1.05),fontsize=6.8,handletextpad=.3,columnspacing=.6)
fig.text(.045,.030,f"GDSC2 median Δ: {summary['gdsc2_main']['median_delta']:.3f} → {summary['spline_sensitivity']['median_delta']:.3f}",fontsize=7.2)
spline.reset_index().to_csv(out/'panel_c_spline_source.tsv',sep='\t',index=False);g.reset_index().to_csv(out/'panel_c_linear_source.tsv',sep='\t',index=False)

# D: recovered original matched Chronos comparator; no fitting in this drawer.
heading('d','Dependency source',.375,.286)
recovery_dir=root/cfg['matched_recovery_dir']
recovery=json.loads((recovery_dir/'recovery_report.json').read_text())
if recovery['status']!='PASS':raise RuntimeError('Matched baseline recovery gate is not PASS')
paired=pd.read_csv(recovery_dir/'panel_d_matched_target_source.tsv',sep='\t').set_index('target').loc[order]
d=fig.add_axes([.435,.085,.195,.153]);clean(d)
for i,t in enumerate(order):
 old=float(paired.loc[t,'delta_t_chronos_matched']);new=float(paired.loc[t,'delta_t_demeter2_matched'])
 d.plot([old,new],[i-.10,i+.10],color=colors[t],lw=.9)
 d.scatter(old,i-.10,s=26,facecolor='white',edgecolor=colors[t],linewidth=.9,zorder=2)
 d.scatter(new,i+.10,s=28,color=colors[t],marker='s',zorder=3)
d.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xlim=cfg['figure']['delta_axis_range'],xticks=[0,.2,.4,.6]);d.set_xlabel('Δ = W − C')
d.scatter([],[],s=22,facecolor='white',edgecolor='#53616A',label='Chronos');d.scatter([],[],s=24,color='#53616A',marker='s',label='DEMETER2')
d.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.30,1.05),fontsize=6.8,handletextpad=.3,columnspacing=.6)
d2=summary['demeter2_sensitivity']
fig.text(.375,.030,f"Median Δ: {d2['primary_median_delta_chronos_same_subset']:.4f} vs {d2['primary_median_delta_demeter2']:.4f}",fontsize=7.2)
fig.text(.375,.016,'42 drugs · 6 targets · matched cells per drug',fontsize=6.5,color='#647784')
paired.reset_index().to_csv(out/'panel_d_source.tsv',sep='\t',index=False,float_format='%.17g')

# E: target-anchor attenuation versus dependency-matched non-target genes (saved outputs only).
heading('e','Target vs matched genes',.705,.286)
dd=root/cfg['decomposition_dir']
e=fig.add_axes([.765,.085,.195,.153]);clean(e)
e_rows=[]
for res,sub,off,marker in [('GDSC2','',-.13,'o'),('PRISM','prism',.13,'s')]:
 at=pd.read_csv(dd/sub/'A_B_target_attenuation.tsv',sep='\t',index_col=0)
 nl=pd.read_csv(dd/sub/'B_null_target_results.tsv',sep='\t')
 for i,t in enumerate(order):
  if t not in at.index:continue
  null=nl.loc[nl.target.eq(t),'A_null'].to_numpy(float)
  lo,mid,hi=np.quantile(null,[.05,.5,.95])
  e.plot([lo,hi],[i+off,i+off],color='#B8C2C9',lw=2.2,solid_capstyle='butt',zorder=1)
  e.plot([mid,mid],[i+off-.08,i+off+.08],color='#7D8A93',lw=.8,zorder=2)
  true=float(at.loc[t,'A_t'])
  e.scatter(true,i+off,s=24,marker=marker,color=colors[t] if res=='GDSC2' else 'white',edgecolor=colors[t],linewidth=.9,zorder=3)
  e_rows.append({'resource':res,'target':t,'A_true':true,'f_t':float(at.loc[t,'f_t']),'P_A':float(at.loc[t,'P_A']),'null_q05':lo,'null_median':mid,'null_q95':hi,'n_null':len(null)})
e.axvline(0,color='#C9D1D6',lw=.6,zorder=0)
e.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xlim=(-.015,.032),xticks=[-.01,0,.01,.02,.03])
e.set_xticklabels(['−0.01','0','0.01','0.02','0.03']);e.set_xlabel('Attenuation of W by anchor')
for lab,t in zip(e.get_yticklabels(),order):lab.set_color(colors[t])
e.scatter([],[],s=20,color='#53616A',marker='o',label='GDSC2');e.scatter([],[],s=20,facecolor='white',edgecolor='#53616A',marker='s',label='PRISM')
e.plot([],[],color='#B8C2C9',lw=2.2,label='Null 5–95%')
e.legend(frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(-.40,1.05),fontsize=6.5,handletextpad=.3,columnspacing=.6,handlelength=1.4)
sg_=json.loads((dd/'summary.json').read_text())['aggregate_attenuation'];sp_=json.loads((dd/'prism'/'summary.json').read_text())['aggregate_attenuation']
def fmt_p(v):
 m,ex=f'{v:.1e}'.split('e');return f'{float(m):.1f}'+r'$\times10^{'+str(int(ex))+'}$' if v<.001 else f'{v:.2f}'
fig.text(.705,.030,f"Median: GDSC2 {sg_['median_A_true']:.3f}, P = {fmt_p(sg_['P'])}",fontsize=7.0)
fig.text(.705,.016,f"PRISM {sp_['median_A_true']:.4f}, P = {fmt_p(sp_['P'])} · 100 genes/target",fontsize=6.5,color='#647784')
pd.DataFrame(e_rows).to_csv(out/'panel_e_source.tsv',sep='\t',index=False,float_format='%.17g')
for ext in ['pdf','svg','png']:fig.savefig(out/f'Figure2.{ext}',dpi=cfg['figure']['dpi'])
plt.close(fig)
paths=[root/cfg['decomposition_dir']/n for n in ['A_B_target_attenuation.tsv','B_null_target_results.tsv','summary.json','prism/A_B_target_attenuation.tsv','prism/B_null_target_results.tsv','prism/summary.json']]+[CFG_PATH,Path(__file__),nd/'n3_results.json',nd/'n3_target_results.tsv',nd/'n3_prism_targets.tsv',nd/'n3_spline_target_results.tsv',nd/'n3_demeter2_target_results.tsv',nd/'n3_demeter2_drug_coverage.tsv',sd/'all_target_results.tsv',sd/'scenario_summary.tsv',root/cfg['sources']['sensitivity_spec']]
paths.extend(recovery_dir/name for name in ['recovery_report.json','panel_d_matched_target_source.tsv','matched_chronos_target_results.tsv','matched_drug_coverage.tsv','pair_cell_count_checks.tsv','raw_input_checks.json','frozen_source_checks.json'])
(out/'source_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
(out/'run_metadata.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'new_analysis':False,'panel_e_source':'03_analysis/2026-10-29_target_anchor_decomposition','new_fits':False,'new_tests':False,'panel_d_only_frozen_overall_comparison':False,'panel_d_baseline_recovery_gate':'PASS','freeze':False},indent=2)+'\n')
print(out)
