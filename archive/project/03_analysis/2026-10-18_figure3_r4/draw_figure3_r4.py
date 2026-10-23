"""Draw approved Figure3 R3 from frozen sources only; no analysis or fitting."""
from pathlib import Path
import os,json,hashlib
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;cfg=yaml.safe_load((BASE/'run_config.yaml').read_text());root=Path(os.environ.get('PROJECT_ROOT',cfg['project_root']));out=root/cfg['output_dir'];src=root/cfg['source_run'];nd=root/cfg['source_n3'];layout=root/cfg['source_layout']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,h in json.loads((layout/'SOURCE_MANIFEST.json').read_text()).items():assert sha(root/n)==h
D=json.loads((src/'PERFORMANCE_DECISION.json').read_text());assert D['scientific_status']=='MODEST_OR_UNCERTAIN'
X=pd.read_csv(nd/'n3_cross_resource_xs_xc.tsv',sep='\t');S=json.loads((nd/'n3_results.json').read_text())['prism']['cross_resource_xS_xC'];T=pd.read_csv(src/'TARGET_INCREMENTAL_R2.tsv',sep='\t');BM=pd.read_csv(src/'BENCHMARK_OVERLAP_TARGET_PERFORMANCE.tsv',sep='\t');registry=pd.read_csv(layout/'FIGURE3_NUMERIC_EVIDENCE_MAP.tsv',sep='\t');order=['EGFR','MTOR','ESR1','TOP1'];counts={'EGFR':6,'MTOR':12,'ESR1':2,'TOP1':2};C={'EGFR':'#2878A6','MTOR':'#D18232','ESR1':'#499783','TOP1':'#9B6C9F','FKBP1A':'#7B858B'}
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=np.array(cfg['size_mm'])/25.4,facecolor='white')
def title(letter,label,x,y):fig.text(x,y,letter,fontsize=10,weight='bold',va='top');fig.text(x+.032,y-.001,label,fontsize=9.2,va='top')
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.6)
# a: all435 reciprocal associations, including the nonfocal FKBP1A pair.
title('a','Cross-screen correspondence',.045,.965)
a=fig.add_axes([.115,.655,.255,.245]);clean(a)
for i,is_same in enumerate([True,False]):
 v=X[X.same_target.eq(is_same)];offset=((np.arange(len(v))*7)%len(v))/max(1,len(v)-1)-.5
 for off,r in zip(offset,v.itertuples()):
  extra=is_same and r.target_a=='FKBP1A';a.scatter(i+off*(.26 if is_same else .40),r.x_pair,s=26 if is_same else 7,marker='^' if extra else 'o',facecolor='white' if extra else C[r.target_a] if is_same else '#A9B4BC',edgecolor=C[r.target_a] if extra else 'none',lw=.9,alpha=.9 if is_same else .38,zorder=3 if is_same else 1)
 med=S['xS_median'] if is_same else S['xC_median'];a.plot([i-.26,i+.26],[med,med],lw=1.7,color='#334A58',zorder=4);a.text(i,.625,'n = '+str(len(v)),ha='center',fontsize=7)
for target in order+['FKBP1A']:a.scatter([],[],s=20,marker='^' if target=='FKBP1A' else 'o',facecolor='white' if target=='FKBP1A' else C[target],edgecolor=C[target],label=target+('†' if target=='FKBP1A' else ''))
a.legend(loc='upper left',bbox_to_anchor=(1.04,1),fontsize=6.5,frameon=False,handlelength=.8,handletextpad=.3,borderaxespad=0)
a.axhline(0,color='#D5DDE2',lw=.6,zorder=0);a.set(xlim=(-.55,1.55),ylim=(-.2,.68),xticks=[0,1],xticklabels=['Same label','Cross label'],yticks=[-.2,0,.2,.4,.6]);a.set_ylabel('Mean reciprocal cross-screen ρ')
fig.text(.045,.605,'Frozen medians: 0.124 vs 0.016',fontsize=7.5)
fig.text(.045,.575,'† Additional FKBP1A pair outside focal groups.',fontsize=6.5,color='#647784')
fig.text(.045,.550,'Descriptive association; pairs share compounds.',fontsize=6.5,color='#647784')
# b: two predeclared decisive comparisons, with frozen conditional intervals.
title('b','Incremental predictive performance',.535,.965)
b=fig.add_axes([.650,.700,.305,.160]);clean(b)
comparisons=['M1G_vs_M0G','M1G_vs_M2G'];brows=[]
for i,key in enumerate(comparisons):
 y=1-i;value=D['equal_target_median_delta_r2'][key];lo,hi=D['conditional_paired_cell_ci95'][key];b.plot([lo,hi],[y,y],color='#536A78',lw=1.1);b.scatter(value,y,s=28,marker='o' if i==0 else 's',facecolor='#334A58' if i==0 else 'white',edgecolor='#334A58',lw=.9,zorder=3);b.text(value,y+.29,f'{value:+.5f}',ha='center',fontsize=7,color='#334A58');brows.append({'comparison':key,'delta_r2':value,'ci_low':lo,'ci_high':hi})
b.axvline(0,color='#A8B5BE',lw=.8,zorder=0);b.set(xlim=(-.01,.04),ylim=(-.5,1.65),yticks=[1,0],yticklabels=['vs M0G','vs M2G'],xticks=[-.01,0,.02,.04]);b.set_xlabel('Paired incremental R²')
fig.text(.535,.910,'M1G: same-target donor + D + lineage + G',fontsize=6.5,color='#647784')
fig.text(.535,.635,'22 tasks · 11 recipients · 4 targets · 5-fold OOF',fontsize=6.5)
fig.text(.535,.605,'95% conditional paired-cell bootstrap CI',fontsize=6.5,color='#647784')
fig.text(.535,.575,'Overall support criterion not met',fontsize=7.3,weight='semibold',color='#53616B')
fig.text(.535,.550,'Baseline gain < 0.01; both intervals cross zero.',fontsize=6.5,color='#647784')
# c: separate axes for paired increments and absolute benchmark reference.
title('c','Target-specific estimates of predictive gain',.045,.485)
c=fig.add_axes([.140,.205,.460,.205]);clean(c)
crows=[];bmrows=[]
for i,target in enumerate(order):
 y=3-i;values=[]
 for j,key in enumerate(comparisons):
  value=float(T[T.target.eq(target)&T.comparison.eq(key)].delta_r2.iloc[0]);values.append(value);c.scatter(value,y+(.10 if j==0 else -.10),s=27,marker='o' if j==0 else 's',facecolor=C[target] if j==0 else 'white',edgecolor=C[target],lw=.9,zorder=3);crows.append({'target':target,'comparison':key,'delta_r2':value,'directed_tasks':counts[target]})
 c.plot(values,[y+.10,y-.10],color=C[target],lw=.65,alpha=.65);c.text(1.15,y,str(counts[target]),ha='center',va='center',transform=c.get_yaxis_transform(),fontsize=7,color='#647784')
 for j,value in enumerate(values):c.text(1.40 if j==0 else 1.71,y,f'{value:+.5f}',ha='center',va='center',transform=c.get_yaxis_transform(),fontsize=7.2,color='#334A58')
c.axvline(0,color='#A8B5BE',lw=.8,zorder=0);c.set(xlim=(-.025,.18),ylim=(-.55,3.55),yticks=[3,2,1,0],yticklabels=order,xticks=[0,.05,.10,.15]);c.set_xlabel('Paired incremental R²')
for tick,target in zip(c.get_yticklabels(),order):tick.set_color(C[target])
c.text(1.15,4,'Tasks',ha='center',transform=c.get_yaxis_transform(),fontsize=6.5)
c.scatter([],[],s=25,color='#536A78',label='M1G − M0G');c.scatter([],[],s=25,marker='s',facecolor='white',edgecolor='#536A78',label='M1G − M2G');c.legend(loc='lower left',bbox_to_anchor=(-.23,1.10),ncol=2,frameon=False,fontsize=6.5,handletextpad=.35,columnspacing=1)
c.text(1.40,4,'vs M0G',ha='center',transform=c.get_yaxis_transform(),fontsize=6.5);c.text(1.71,4,'vs M2G',ha='center',transform=c.get_yaxis_transform(),fontsize=6.5)
fig.text(.780,.455,'Paired ΔR²',fontsize=7.2,weight='semibold')
fig.text(.045,.135,'TOP1: two reciprocal irinotecan–topotecan tasks; exploratory, context-limited signal.',fontsize=6.5,color='#647784')
fig.text(.045,.109,'Gains are medians of paired task increments, aggregated within recipient and target; no target-level CIs are shown.',fontsize=6.5,color='#647784')
fig.text(.045,.083,'Prediction uses measured GDSC2 donor responses in test cells. The recipient drug is observed during training.',fontsize=6.5,color='#647784')
fig.text(.045,.057,'Same-drug benchmark: Supplementary Figure Sy; available-B subsets, potentially different training samples.',fontsize=6.5,color='#647784')
for ext in ['pdf','svg','png']:fig.savefig(out/f'Figure3_R4.{ext}',dpi=cfg['dpi'])
plt.close(fig)
X.to_csv(out/'PANEL_A_ALL435_SOURCE.tsv',sep='\t',index=False);pd.DataFrame(brows).to_csv(out/'PANEL_B_SOURCE.tsv',sep='\t',index=False);pd.DataFrame(crows).to_csv(out/'PANEL_C_PRIMARY_SOURCE.tsv',sep='\t',index=False);BM[BM.model.isin(['M1G','M_BG'])].to_csv(out/'SUPPLEMENT_BENCHMARK_SOURCE.tsv',sep='\t',index=False)
(out/'SOURCE_SHA256.json').write_text(json.dumps({str(p):sha(p) for p in [nd/'n3_cross_resource_xs_xc.tsv',nd/'n3_results.json',src/'PERFORMANCE_DECISION.json',src/'TARGET_INCREMENTAL_R2.tsv',src/'BENCHMARK_OVERLAP_TARGET_PERFORMANCE.tsv',BASE/'run_config.yaml',Path(__file__)]},indent=2)+'\n')

fig,ax=plt.subplots(figsize=(100/25.4,100/25.4));fig.subplots_adjust(left=.23,right=.94,bottom=.30,top=.76);clean(ax)
fig.text(.07,.94,'Same-drug benchmark',fontsize=9.2,va='top');fig.text(.07,.87,'Absolute OOF R² on available-B test subsets',fontsize=7)
for i,target in enumerate(order):
 for j,model in enumerate(['M1G','M_BG']):
  value=float(BM[BM.target.eq(target)&BM.model.eq(model)].r2.iloc[0]);ax.scatter(value,3-i+(.10 if j==0 else -.10),s=27,marker='o' if j==0 else 'D',facecolor='white' if j==0 else C[target],edgecolor=C[target],lw=.9)
ax.axvline(0,color='#B9C5CD',lw=.7,zorder=0);ax.set(xlim=(-.12,.30),ylim=(-.55,3.55),yticks=[3,2,1,0],yticklabels=order,xticks=[-.1,0,.1,.2,.3]);ax.set_xlabel('Absolute OOF R²')
for tick,target in zip(ax.get_yticklabels(),order):tick.set_color(C[target])
ax.scatter([],[],s=25,facecolor='white',edgecolor='#536A78',label='M1G');ax.scatter([],[],s=25,marker='D',color='#536A78',label='Same drug (M_BG)');ax.legend(loc='lower left',bbox_to_anchor=(-.05,1.04),frameon=False,ncol=2,fontsize=6.5,handletextpad=.3,columnspacing=.7)
fig.text(.07,.17,'9 full-coverage / 13 partial tasks.',fontsize=6.5,color='#647784');fig.text(.07,.125,'Matched test subsets; training samples can differ.',fontsize=6.5,color='#647784');fig.text(.07,.080,'Practical reference; not an upper bound or superiority test.',fontsize=6.5,color='#647784')
for ext in ['pdf','svg','png']:fig.savefig(out/f'Supplementary_Figure_Sy_same_drug_benchmark.{ext}',dpi=cfg['dpi'])
plt.close(fig)
