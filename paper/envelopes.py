"""Quantile residuals r_i = Phi^{-1}{F(y_i; theta_hat)} with simulated envelopes (B refits), MLE and LMDPDE, tau = 0.5."""
import os, sys, json, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
P = Path(__file__).resolve().parent; ROOT = P.parent
sys.path.insert(0, str(ROOT / 'modal_run'))
SPEC = {'water': ('phpws', 'mhdi + np.log(incpc) + region + np.log(pop)', 'UW', .5),
        'HIC': ('HIC', 'URB + np.log(GDP)', 'KW', .4), 'Firm': ('FIRMCOST', 'SIZELOG + INDCOST', 'KW', .1)}
B = 100


def design(ds):
    import patsy
    resp, form, fam, g = SPEC[ds]; d = pd.read_csv(ROOT / 'data_real/csv' / f'{ds}.csv'); y = d[resp].to_numpy(float)
    Xd = patsy.dmatrix('1 + ' + form, d, return_type='dataframe')
    for c in Xd.columns:
        if c != 'Intercept' and Xd[c].nunique() > 2: Xd[c] = (Xd[c] - Xd[c].mean()) / Xd[c].std(ddof=0)
    return Xd.to_numpy(), y, fam, g


def resid(v, X, y, fam):
    import dpd
    from scipy.stats import norm
    a, lb = dpd.pars(v, X, fam); b = np.exp(lb)
    if fam == 'UW': F = np.exp(-b * (-np.log(y)) ** a)
    else: F = -np.expm1(b * np.log1p(-y ** a))
    return norm.ppf(np.clip(F, 1e-15, 1 - 1e-15))


def fit_both(X, y, fam, g, start):
    import dpd
    from models import UnitWeibullQR, KumaraswamyQR
    M = {'UW': UnitWeibullQR, 'KW': KumaraswamyQR}[fam]
    m = M(.5).fit(X, y, start[:-1]); f = dpd.fit(X, y, fam, g, 'logit', start[:-1], m.v)
    return m.v, f.x


def sim_one(args):
    ds, which, v, seed = args
    import dpd
    dpd.TAU = .5
    X, _, fam, g = design(ds); rng = np.random.default_rng(seed)
    a, lb = dpd.pars(v, X, fam); b = np.exp(lb)
    with np.errstate(all='ignore'):
        if fam == 'UW': ys = np.exp(-(rng.exponential(size=len(X)) / b) ** (1 / a))
        else: ys = (-np.expm1(np.log1p(-rng.uniform(size=len(X))) / b)) ** (1 / a)
    ys = np.clip(ys, 1e-10, 1 - 1e-10)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        vm, vl = fit_both(X, ys, fam, g, v)
    return np.sort(resid(vm if which == 'mle' else vl, X, ys, fam))


def main():
    import dpd
    dpd.TAU = .5; out = {}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        fits = {}
        for ds in SPEC:
            X, y, fam, g = design(ds)
            from models import LogitQR
            b0 = LogitQR(.5).fit(X, y, None).v
            vm, vl = fit_both(X, y, fam, g, np.r_[b0, 0.])
            fits[ds] = (vm, vl); out[ds] = {'mle': np.sort(resid(vm, X, y, fam)).tolist(), 'lmdpde': np.sort(resid(vl, X, y, fam)).tolist()}
    jobs = [(ds, w, fits[ds][0 if w == 'mle' else 1], 1000 * i + j) for i, ds in enumerate(SPEC) for j, w in enumerate(('mle', 'lmdpde')) for _ in range(1)]
    jobs = [(ds, w, v, s * 1000 + r) for ds, w, v, s in jobs for r in range(B)]
    with ProcessPoolExecutor(9) as pool:
        res = list(pool.map(sim_one, jobs, chunksize=4))
    k = 0
    for ds in SPEC:
        for w in ('mle', 'lmdpde'):
            S = np.array(res[k:k + B]); k += B
            out[ds][w + '_lo'] = np.percentile(S, 2.5, axis=0).tolist(); out[ds][w + '_hi'] = np.percentile(S, 97.5, axis=0).tolist()
            out[ds][w + '_mid'] = np.median(S, axis=0).tolist()
            obs = np.array(out[ds][w]); out[ds][w + '_outside'] = int(np.sum((obs < S.min(0) * 0 + np.percentile(S, 2.5, axis=0)) | (obs > np.percentile(S, 97.5, axis=0))))
    json.dump(out, open(P / 'tables' / 'envelopes.json', 'w'))
    for ds in SPEC: print(ds, 'points outside envelope: MLE', out[ds]['mle_outside'], 'LMDPDE', out[ds]['lmdpde_outside'], 'n', len(out[ds]['mle']))


if __name__ == '__main__':
    main()
