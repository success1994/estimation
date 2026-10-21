"""Model-based robust competitor: ML for the rectangular mixture (1-lam) f + lam U(0,1) of the true family,
quantile-parameterised so that the tau-quantile of the mixture is expit(x'beta) (cf. Castro et al. 2024).
Same data sets as results_full (tau = 0.5, n = 300).  Run: python3 code/sim_rect.py"""
import os, sys, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT / 'modal_run'))
TAU, N, R = .5, 300, 300
FAM = {0: 'UW', 1: 'UW', 2: 'KW', 3: 'KW'}


def negll(v, X, y, fam):
    p = X.shape[1]; q = expit(np.clip(X @ v[:p], -30, 30)); s = np.exp(v[p]); lam = .5 * expit(v[p + 1])
    pt = np.clip((TAU - lam * q) / (1 - lam), 1e-12, 1 - 1e-12)
    with np.errstate(all='ignore'):
        if fam == 'KW':
            b = np.log1p(-pt) / np.log1p(-q ** s)
            lf = np.log(s * b) + (s - 1) * np.log(y) + (b - 1) * np.log1p(-y ** s)
        else:
            t = -np.log(y); a = -np.log(pt) / (-np.log(q)) ** s
            lf = np.log(a * s) - np.log(y) + (s - 1) * np.log(t) - a * t ** s
        ll = np.logaddexp(np.log1p(-lam) + lf, np.log(lam))
    z = -np.sum(ll)
    return z if np.isfinite(z) else 1e10


def one(task):
    import simulate_full as sf
    from models import LogitQR
    si, ci, rep = task; fam = FAM[si]
    X, y, _, _ = sf.generate(si, N, ci, rep)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        b0 = LogitQR(TAU).fit(X, y, None).v
        best = None
        for s0 in (-1., 0., 1.):
            for r0 in (-4., -1.):
                r = minimize(negll, np.r_[b0, s0, r0], args=(X, y, fam), method='L-BFGS-B',
                             bounds=[(-30, 30)] * 3 + [(-6, 6), (-12, 12)], options={'maxiter': 2000})
                if best is None or r.fun < best.fun: best = r
    return dict(setting=si, scenario=f'C{ci}', rep=rep, beta1=best.x[1], beta2=best.x[2], lam=.5 * expit(best.x[4]), converged=bool(best.success))


if __name__ == '__main__':
    tasks = [(si, ci, rep) for si in range(4) for ci in range(6) for rep in range(R)]
    with ProcessPoolExecutor(9) as pool: res = list(pool.map(one, tasks, chunksize=20))
    d = pd.DataFrame(res); (ROOT / 'results_rect').mkdir(exist_ok=True); d.to_csv(ROOT / 'results_rect/sim_rect.csv', index=False)
    print(d.groupby(['setting', 'scenario']).agg(rmse1=('beta1', lambda x: np.sqrt(np.mean((x - .8) ** 2))),
                                                 rmse2=('beta2', lambda x: np.sqrt(np.mean((x + .6) ** 2))), conv=('converged', 'mean'), lam=('lam', 'median')).round(3))
