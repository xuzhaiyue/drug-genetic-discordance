"""Reviewed candidate primitives, not a biological analysis runner."""
import itertools
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

def directional_hc3(y, nuisance, program):
    x=np.column_stack([np.ones(len(y)),nuisance,program])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite association input; do not drop rows silently')
    if np.linalg.matrix_rank(x,tol=1e-10)!=x.shape[1]:
        raise ValueError('Rank-deficient design; report NOT_EVALUABLE')
    result=sm.OLS(y,x).fit(cov_type='HC3',use_t=True)
    leverage=np.sum(x*(x@np.linalg.inv(x.T@x)),axis=1)
    if np.max(leverage)>=.95:
        raise ValueError('Leverage >=0.95; report NOT_EVALUABLE')
    t=result.params[-1]/result.bse[-1]
    return {'beta':float(result.params[-1]),'se_HC3':float(result.bse[-1]),'df':float(result.df_resid),'p_plus':float(stats.t.sf(t,result.df_resid)),'p_minus':float(stats.t.cdf(t,result.df_resid)),'max_leverage':float(max(leverage))}

def pair_supported_p(components,allowed_pairs):
    if not allowed_pairs:return 1.0
    pair_p=[min(1.,2*min(max(components[i]['p_plus'],components[j]['p_plus']),max(components[i]['p_minus'],components[j]['p_minus']))) for i,j in allowed_pairs]
    return min(1.,len(allowed_pairs)*min(pair_p))

def by_300(pvalues):
    if len(pvalues)!=300:raise ValueError('Preserve planned family of 300')
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid P values')
    return multipletests(p,alpha=.05,method='fdr_by')[1]
