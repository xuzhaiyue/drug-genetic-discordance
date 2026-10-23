"""Original response transform and OLS routines, extracted without modification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
MIN_N = 200

def response_vector(auc: pd.Series) -> pd.Series:
    """Higher = more sensitive (lower AUC)."""
    x = auc.astype(float)
    mu, sd = x.mean(), x.std() or 1.0
    return -(x - mu) / sd

def partial_corr(y: pd.Series, x: pd.Series, lin: pd.DataFrame) -> float:
    df = pd.concat([y.rename("y"), x.rename("x"), lin], axis=1).dropna()
    if len(df) < MIN_N:
        return float("nan")
    lin_cols = list(lin.columns)
    Y = df["y"].to_numpy()
    X = df["x"].to_numpy()
    Z = df[lin_cols].to_numpy()
    # residualize y and x on Z + intercept
    Z1 = np.column_stack([np.ones(len(df)), Z])
    by, *_ = np.linalg.lstsq(Z1, Y, rcond=None)
    bx, *_ = np.linalg.lstsq(Z1, X, rcond=None)
    ry = Y - Z1 @ by
    rx = X - Z1 @ bx
    r, _ = stats.pearsonr(ry, rx)
    return float(r)

def ols_residual(y: pd.Series, x: pd.Series, lin: pd.DataFrame) -> pd.Series:
    df = pd.concat([y.rename("y"), x.rename("x"), lin], axis=1).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if len(df) < MIN_N:
        return out
    lin_cols = list(lin.columns)
    Xmat = np.column_stack([np.ones(len(df)), df["x"].to_numpy(), df[lin_cols].to_numpy()])
    beta, *_ = np.linalg.lstsq(Xmat, df["y"].to_numpy(), rcond=None)
    out.loc[df.index] = df["y"].to_numpy() - Xmat @ beta
    return out

def spearman_pair(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna()
    if int(m.sum()) < 30:
        return float("nan"), int(m.sum())
    r, _ = stats.spearmanr(a[m], b[m])
    return float(r), int(m.sum())
