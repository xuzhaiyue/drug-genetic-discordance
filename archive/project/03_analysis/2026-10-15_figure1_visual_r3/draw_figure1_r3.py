"""Draw an N3 review figure from saved results; do not fit models or run tests."""
import os,json,hashlib,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CFG_PATH=Path(__file__).with_name('drawing_config.yaml')
cfg=yaml.safe_load(CFG_PATH.read_text())
root=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']))
out=root/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
nd=root/cfg['sources']['n3_dir']
order=cfg['figure']['target_order']
colors=dict(zip(order,['#2878A6','#D18232','#499783','#9B6C9F','#A68B38','#668396']))
g=pd.read_csv(nd/'n3_target_results.tsv',sep='\t').set_index('target').loc[order]
p=pd.read_csv(nd/'n3_prism_targets.tsv',sep='\t').set_index('target')
pairs=pd.read_csv(nd/'n3_pairs_gdsc2.tsv',sep='\t')
elig=pd.read_csv(nd/'PHASE1B_ELIGIBILITY.tsv',sep='\t')
drugs=pd.read_csv(nd/'n3_drug_results.tsv',sep='\t')
summary=json.loads((nd/'n3_results.json').read_text())
sens=pd.read_csv(root/cfg['sources']['sensitivity_summary'],sep='\t')
# Export the actual plotting data without changing the estimands.
g.reset_index().to_csv(out/'panel_d_source.tsv',sep='\t',index=False)
p.reset_index().to_csv(out/'panel_e_prism_source.tsv',sep='\t',index=False)
g.reset_index().to_csv(out/'panel_e_gdsc2_source.tsv',sep='\t',index=False)
elig[['compound_id','status','primary_target','n_complete']].to_csv(out/'panel_b_eligibility_source.tsv',sep='\t',index=False)
g.reset_index()[['target','n_drugs','within_pair_count']].to_csv(out/'panel_b_groups_source.tsv',sep='\t',index=False)

plt.rcParams.update({'font.family':'Arial','font.size':7.0,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=np.array(cfg['figure']['size_mm'])/25.4,facecolor='white')
def heading(letter,title,x,y):
 fig.text(x,y,letter,weight='bold',fontsize=10,va='top')
 fig.text(x+.032,y-.001,title,fontsize=9.2,weight='medium',va='top')
def clean(ax):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.6)
def box(ax,x,y,w,h,text,fc='#F8F9FA',edge='#E0E5E8',size=7.2):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',facecolor=fc,edgecolor=edge,lw=.7))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.35)
def arrow(ax,start,end):
 ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':.85,'color':'#667780','shrinkA':2,'shrinkB':2})

# Reflow the approved artwork for the final 182 mm canvas.
heading('a','Study design and residual definition',.045,.982)
A=fig.add_axes([.045,.755,.91,.19]);A.axis('off');A.set_xlim(0,1);A.set_ylim(0,1)
ink='#26343E';muted='#647784';red='#B96F70';blue='#4F819E'
A.plot([.29,.29],[0,1],color='#DCE3E8',lw=.6);A.plot([.675,.675],[0,1],color='#DCE3E8',lw=.6)
A.text(0,.98,'Data integration',weight='medium',fontsize=7.5,va='top')
A.text(.315,.98,'Residual modelling',weight='medium',fontsize=7.5,va='top')
A.text(.70,.98,'Residual sharing',weight='medium',fontsize=7.5,va='top')
box(A,.005,.65,.145,.22,'GDSC2\nR = −z(AUC)',fc='#F5F8FA',size=7)
box(A,.005,.07,.145,.22,'PRISM\nR = −z(AUC)',fc='#F5F9F7',size=7)
box(A,.12,.365,.155,.24,'DepMap 26Q1\nChronos + lineage',size=7)
A.text(.17,.78,'Match IDs',fontsize=6.5,color=muted)
A.text(.17,.27,'Match IDs',fontsize=6.5,color=muted)
arrow(A,(.15,.69),(.28,.69));arrow(A,(.20,.605),(.28,.69))
arrow(A,(.15,.20),(.28,.20));arrow(A,(.20,.365),(.28,.20))
# Read only the fixed schematic marker coordinates from the user's SVG.
import xml.etree.ElementTree as ET
svg=ET.parse(root/cfg['artwork_svg']).getroot()
ns={'s':'http://www.w3.org/2000/svg'}
coll=svg.find(".//s:g[@id='PathCollection_1']",ns)
xy=np.array([[float(u.attrib['x']),float(u.attrib['y'])] for u in coll.findall('.//s:use',ns)])
xy[:,0]=(xy[:,0]-499.68)/(718.56-499.68);xy[:,1]=(299.52-xy[:,1])/(299.52-109.44)
xy=pd.DataFrame(xy,columns=['schematic_x','schematic_y'])
xy.to_csv(out/'panel_a_user_schematic_coordinates.tsv',sep='\t',index=False)
S=fig.add_axes([.367,.795,.155,.10]);S.set(xlim=(-.04,1.04),ylim=(-.04,1.04));S.axis('off')
S.scatter(xy.schematic_x,xy.schematic_y,s=3,color='#9EABB4',alpha=.4,edgecolors='none')
S.plot([0,1],[0,1],ls='--',lw=.7,color='#98A5AE')
S.annotate('',xy=(1.05,0),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':.6,'color':ink})
S.annotate('',xy=(0,1.05),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':.6,'color':ink})
for xx,yy,col,lab in [(.28,.64,red,'E > 0'),(.74,.43,blue,'E < 0')]:
 S.plot([xx,xx],[xx,yy],color=col,lw=.85);S.scatter([xx],[yy],color=col,s=15)
 S.text(xx+.045,yy,lab,color=col,fontsize=6.5,va='center')
S.text(.05,1.12,'Schematic',fontsize=6.5,fontstyle='italic',color=muted,transform=S.transAxes)
S.text(.5,-.16,r'Predicted $\widehat R$',fontsize=6.5,ha='center',transform=S.transAxes)
S.text(-.13,.5,'Observed R',fontsize=6.5,rotation=90,ha='center',va='center',transform=S.transAxes)
A.text(.545,.78,'Per-compound fit',fontsize=6.5,color=muted)
A.text(.545,.67,r'$R \sim D$ + lineage',fontsize=6.6)
A.text(.545,.48,'Residual',fontsize=6.5,color=muted)
A.text(.545,.36,r'$E = R - \widehat R$',fontsize=8)
A.text(.315,.07,'Higher E: more sensitive than expected',fontsize=6.5,color=red)
A.text(.70,.86,'Schematic · aligned cells',fontsize=6.5,fontstyle='italic',color=muted)
for j,(title,vals,col) in enumerate([('E$_A$',[.35,.65,.30,.72,.42],red),('E$_B$',[.40,.60,.35,.70,.45],red),('E$_C$',[.70,.30,.60,.35,.50],'#8A9BA6')]):
 xx=.735+j*.095
 A.text(xx,.75,title,fontsize=7,ha='center')
 for k,val in enumerate(vals):
  yy=.68-k*.045;A.plot([xx-.026,xx-.026+.05*val],[yy,yy],color=col,lw=1.5,solid_capstyle='round')
A.text(.70,.36,'A–B: same target',fontsize=6.5,color=red)
A.text(.70,.28,'A–C: cross-target, involving t',fontsize=6.5,color=muted)
A.text(.70,.16,r'$W_t$, $C_t$ = median pairwise ρ',fontsize=6.6)
A.text(.70,.05,r'$\Delta_t = W_t - C_t$',fontsize=8)
fig.text(.688,.735,r'Overall effect: median across target $\Delta_t$',fontsize=6.5,color=muted)
heading('b','Cohort and qualifying coverage',.045,.720)
b=fig.add_axes([.045,.434,.405,.239]);b.axis('off');b.set_xlim(0,1);b.set_ylim(0,1)
b.text(0,.99,'286 GDSC2 drugs assessed',fontsize=8,va='top')
x=0
for n,col in zip([43,75,168],['#80A7BB','#D0D7DB','#EDF0F2']):
 w=n/286;b.add_patch(plt.Rectangle((x,.86),w,.045,color=col,lw=0));x+=w
for xx,n,lab in zip([0,.34,.67],[43,75,168],['included','excluded','unresolved']):
 b.text(xx,.81,f'{n} {lab}',fontsize=6.6,va='top')
b.text(0,.68,'43 included drugs · 30 target labels',fontsize=7.5)
b.text(0,.60,'Target',fontsize=7,color='#647784')
b.text(.52,.60,'GDSC2',fontsize=7,ha='center',color='#647784')
b.text(.83,.60,'PRISM',fontsize=7,ha='center',color='#647784')
coverage=[]
for i,t in enumerate(order):
 y=.49-i*.082
 b.text(0,y,t,fontsize=7.5,color=colors[t],va='center')
 for xx,val,col in [(.52,str(int(g.loc[t,'n_drugs'])),'#EDF3F7'),(.83,str(int(p.loc[t,'n_drugs'])) if t in p.index else '—','#EEF5F2' if t in p.index else '#F4F5F6')]:
  b.add_patch(plt.Rectangle((xx-.12,y-.033),.24,.066,facecolor=col,edgecolor='white',lw=.6))
  b.text(xx,y,val,fontsize=8.4,ha='center',va='center',color='#26343E' if val!='—' else '#8A969E')
 coverage.append({'target':t,'gdsc2_drugs':int(g.loc[t,'n_drugs']),'prism_legal_drugs':int(summary['prism']['coverage']['legal_drugs_by_target'][t]),'prism_group_qualifies':t in p.index,'prism_display':str(int(p.loc[t,'n_drugs'])) if t in p.index else '—'})
pd.DataFrame(coverage).to_csv(out/'panel_b_coverage_source.tsv',sep='\t',index=False)
fig.text(.045,.416,'6 target groups · 18 focal compounds',fontsize=6.7)
fig.text(.045,.403,'—: below group eligibility, not absence of drugs.',fontsize=6.5,color='#647784')
heading('c','Cell-line residual concordance',.545,.720)
c=fig.add_axes([.605,.455,.295,.225]);clean(c)
recovery_dir=root/cfg['example']['recovery_dir']
recovery=json.loads((recovery_dir/'recovery_report.json').read_text())
if recovery['status']!='PASS':raise RuntimeError('Panel c recovery gate did not pass')
lock=json.loads((recovery_dir/'selection_lock.json').read_text())
r=pd.Series(lock['selected'])
pts=pd.read_csv(recovery_dir/'panel_c_plot_cells.tsv',sep='\t')
if len(pts)!=int(r.n):raise RuntimeError('Panel c point count differs from frozen n')
lim=float(np.ceil(pts[['E_a','E_b']].abs().to_numpy().max()*2)/2)
c.axhline(0,color='#DCE2E5',lw=.6,zorder=0);c.axvline(0,color='#DCE2E5',lw=.6,zorder=0)
c.scatter(pts.E_a,pts.E_b,s=7,color=colors['EGFR'],alpha=.36,edgecolors='none',rasterized=False)
c.set(xlim=(-lim,lim),ylim=(-lim,lim));c.set_aspect('equal',adjustable='box')
c.set_xlabel(r.name_a.title()+' residual E',fontsize=7.5);c.set_ylabel(r.name_b.title()+' residual E',fontsize=7.5)
c.text(.03,.97,'EGFR',transform=c.transAxes,va='top',fontsize=7)
fig.text(.545,.690,f'Spearman ρ = {r.rho:.3f}; n = {int(r.n)} cell lines',fontsize=7.2)
pts.to_csv(out/'panel_c_cell_source.tsv',sep='\t',index=False,float_format='%.17g')
pd.read_csv(recovery_dir/'selection_candidates.tsv',sep='\t').to_csv(out/'panel_c_selection_candidates.tsv',sep='\t',index=False)
example={**recovery,'selection':lock['rule'],'role':'Illustrative example; all six targets contribute to the discovery estimand'}


heading('d','GDSC2 discovery',.045,.385)
d=fig.add_axes([.12,.095,.225,.245]);clean(d)
for i,t in enumerate(order):
 row=g.loc[t];d.plot([row.C_t,row.W_t],[i,i],color=colors[t],lw=.9)
 d.scatter(row.C_t,i,s=34,facecolors='white',edgecolors=colors[t],linewidths=.9,zorder=2)
 d.scatter(row.W_t,i,s=34,color=colors[t],zorder=3)
 d.text(1.14,i,f'{row.delta_t:.3f}',transform=d.get_yaxis_transform(),ha='center',va='center',fontsize=7.5)
 d.text(1.47,i,str(int(row.within_pair_count)),transform=d.get_yaxis_transform(),ha='center',va='center',fontsize=7.5)
d.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xlim=(-.025,.56),xticks=[0,.2,.4]);d.set_xlabel('Residual Spearman ρ',fontsize=8)
d.text(1.14,-1.02,'Δ',transform=d.get_yaxis_transform(),ha='center',fontsize=7.5)
d.text(1.47,-1.02,'Pairs',transform=d.get_yaxis_transform(),ha='center',fontsize=7.5)
d.scatter([],[],s=24,color='#53616A',label='Same-target W')
d.scatter([],[],s=24,facecolor='white',edgecolor='#53616A',linewidth=.9,label='Cross-target C')
d.legend(loc='lower left',bbox_to_anchor=(-.32,1.04),ncol=2,frameon=False,fontsize=7.0,handletextpad=.3,columnspacing=.6)
d.plot([1.01,1.60],[-.68,-.68],transform=d.get_yaxis_transform(),clip_on=False,color='#E1E6E9',lw=.6)
fig.text(.045,.038,'Median Δ = +0.326',fontsize=8,fontweight='semibold')
fig.text(.045,.020,'6 target groups · 20 within-target pairs',fontsize=6.5,color='#647784')
heading('e','Support in PRISM',.545,.385)
e=fig.add_axes([.635,.095,.24,.245]);clean(e)
for i,t in enumerate(order):
 e.scatter(g.loc[t,'delta_t'],i,color=colors[t],s=34,marker='o')
 if t in p.index:
  e.scatter(p.loc[t,'delta_t'],i,facecolor='white',edgecolor=colors[t],linewidth=.9,s=37,marker='s')
 else:
  e.text(.035,i,'NA',fontsize=7.5,color='#98A3AB',va='center')
e.axvline(0,lw=.6,color='#AAB5BC');e.set(yticks=range(6),yticklabels=order,ylim=(5.5,-.7),xticks=[0,.2,.4,.6],xlim=(-.015,.61));e.set_xlabel('Δ = W − C',fontsize=8)
e.scatter([],[],s=26,color='#53616A',label='GDSC2');e.scatter([],[],s=34,facecolor='white',edgecolor='#53616A',marker='s',label='PRISM')
e.tick_params(axis='y',labelsize=8)
d.tick_params(axis='y',labelsize=8)
e.text(1.27,-1.00,'PRISM\npairs',transform=e.get_yaxis_transform(),ha='center',fontsize=7,linespacing=1.1)
for i,t in enumerate(order):
 e.text(1.27,i,str(int(p.loc[t,'within_pair_count'])) if t in p.index else '—',transform=e.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,color='#26343E' if t in p.index else '#8A969E')
e.legend(loc='lower left',bbox_to_anchor=(-.16,1.04),ncol=2,frameon=False,fontsize=7.0,handletextpad=.3,columnspacing=.7)
fig.text(.545,.038,'PRISM: 30 drugs · 4 targets; median Δ = +0.296',fontsize=7.0)
fig.text(.545,.020,'NA: insufficient qualifying PRISM coverage.',fontsize=7.0,color='#647784')
for ext in ['pdf','svg','png']:
 fig.savefig(out/f'Figure1_visual_r3.{ext}',dpi=cfg['figure']['dpi'])
plt.close(fig)
(out/'panel_c_selection.json').write_text(json.dumps(example,indent=2)+'\n')
source_paths=[root/cfg['artwork_pdf'],root/cfg['artwork_svg'],CFG_PATH,Path(__file__),nd/'n3_target_results.tsv',nd/'n3_prism_targets.tsv',nd/'n3_results.json',nd/'PHASE1B_ELIGIBILITY.tsv',nd/'n3_drug_results.tsv',nd/'n3_pairs_gdsc2.tsv',root/cfg['sources']['sensitivity_summary']]
source_paths.extend(recovery_dir/name for name in ['panel_c_plot_cells.tsv','n3_selected_cell_vectors.tsv','selection_candidates.tsv','selection_lock.json','recovery_report.json','raw_input_checks.json','frozen_source_checks.json'])
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
(out/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'run_metadata.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'new_statistics':False,'new_fits':False,'deterministic_recovery_of_original_two_fits':False,'reuses_previously_verified_N3_recovery':True,'example':example,'review_only':True},indent=2)+'\n')
print(out)
