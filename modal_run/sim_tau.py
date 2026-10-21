"""Tau = 0.1 / 0.9 simulation on Modal, in parallel.

Same design as code/simulate_full.py, but the truth is the tau-quantile parameterisation
(q_i = expit(x_i' beta) is the conditional tau-quantile) and every estimator targets tau.

Run:   .venv-modal/bin/modal run modal_run/sim_tau.py                     (full run)
       .venv-modal/bin/modal run modal_run/sim_tau.py --reps300 1 --reps100 1 --output smoke_tau.csv
Output: results_tau/<output> (resumable: completed dataset blocks are skipped).
"""
import os, sys, json, time, csv
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('robust-unit-qr-tau')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.4.4', 'scipy==1.17.1', 'statsmodels==0.14.6', 'pandas==3.0.3')
         .env({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_file(HERE / 'dpd.py', '/root/app/dpd.py')
         .add_local_file(HERE / 'models.py', '/root/app/models.py'))

BETA = [-.5, .8, -.6]
SETTINGS = [('UW', .6), ('UW', 1.5), ('KW', .6), ('KW', 2.)]
GRID = [.1, .2, .3, .4, .5, .6, .7]
NAMES = (['MLE'] + [f'LMDPDE_{g}' for g in GRID] + ['LMDPDE_auto', 'MDPDE_0.3', 'LogitQR', 'BetaMLE', 'BetaLMDPDE_0.3'])


def run_dataset(si, n, ci, rep, tau):
    import warnings
    import numpy as np
    from scipy.special import expit
    from scipy.stats import beta as beta_dist
    for p in ('/root/app', str(HERE)):
        if p not in sys.path: sys.path.insert(0, p)
    import dpd
    from models import LogitQR, UnitWeibullQR, KumaraswamyQR, BetaReg
    dpd.TAU = tau
    B = np.array(BETA); family, shape = SETTINGS[si]
    rng = np.random.default_rng(20261018)
    TEST = np.c_[np.ones(1000), rng.normal(size=1000), rng.binomial(1, .5, 1000)]; TRUE = expit(TEST @ B)

    def qfun(u, q):
        if family == 'UW':
            alpha = -np.log(tau) / (-np.log(q)) ** shape
            return np.exp(-(-np.log(u) / alpha) ** (1 / shape))
        b = np.log(1 - tau) / np.log1p(-q ** shape)
        return (-np.expm1(np.log1p(-u) / b)) ** (1 / shape)

    rng = np.random.default_rng(np.random.SeedSequence([20261017, si, n, ci, rep, int(round(tau * 10))]))
    X = np.c_[np.ones(n), rng.normal(size=n), rng.binomial(1, .5, n)]; q = expit(X @ B)
    with np.errstate(all='ignore'): y = qfun(rng.uniform(size=n), q)
    clipped = int(np.sum((y < 1e-12) | (y > 1 - 1e-12))); y = np.clip(y, 1e-12, 1 - 1e-12); oc = 0
    if ci:
        rate = .1 if ci in (2, 5) else .05
        idx = rng.choice(n, size=int(np.ceil(rate * n)), replace=False)
        if ci == 3: X[idx, 1] = 3.; y[idx] = rng.uniform(.005, .05, len(idx))
        elif ci in (1, 2): y[idx] = np.where(q[idx] < .5, rng.uniform(.95, .995, len(idx)), rng.uniform(.005, .05, len(idx)))
        else:
            up = q[idx] < .5
            u = np.where(up, rng.uniform(.999, .9999, len(idx)), rng.uniform(1e-4, 1e-3, len(idx)))
            with np.errstate(all='ignore'): yo = qfun(u, q[idx])
            oc = int(np.sum((yo < 1e-12) | (yo > 1 - 1e-12))); y[idx] = np.clip(yo, 1e-12, 1 - 1e-12)

    rows = []; grid = {}; mle = {}

    def R(name, v, se, success, message, obj, runtime, boundary, gsel=float('nan')):
        v = np.asarray(v, float); se = np.asarray(se, float)
        se = np.r_[se, np.full(4 - len(se), np.nan)] if len(se) < 4 else se
        if name.startswith('Beta'):
            a, b = dpd.pars(v, TEST, 'Beta'); pred = beta_dist.ppf(tau, a, b)
        else: pred = expit(TEST @ v[:3])
        rows.append(dict(setting=si, family=family, shape=shape, n=n, tau=tau, scenario=f'C{ci}', rep=rep, estimator=name,
            converged=bool(success), message=str(message)[:200], estimate=json.dumps(v.tolist()), standard_errors=json.dumps(se.tolist()),
            objective=obj, runtime=runtime, boundary=boundary, gamma_selected=gsel,
            slope_squared_error=float(np.sum((v[1:3] - B[1:3]) ** 2)), beta1=float(v[1]), beta2=float(v[2]),
            hit1=bool(abs(v[1] - B[1]) <= 1.96 * se[1]), hit2=bool(abs(v[2] - B[2]) <= 1.96 * se[2]),
            se_valid=bool(np.all(np.isfinite(se[1:3]))), quantile_rmse=float(np.sqrt(np.mean((pred - TRUE) ** 2))),
            clipped=clipped, outlier_clipped=oc))

    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        t = time.monotonic()
        try:
            qr = LogitQR(tau).fit(X, y, None); b0 = qr.v
            R('LogitQR', qr.v, np.asarray(qr.se), qr.success, qr.message, np.nan, time.monotonic() - t, False)
        except Exception as ex:
            b0 = np.linalg.lstsq(X, np.log(y / (1 - y)), rcond=None)[0]
            R('LogitQR', b0, np.full(3, np.nan), False, repr(ex), np.nan, time.monotonic() - t, False)
        for name, fam in (('MLE', family), ('BetaMLE', 'Beta')):
            t = time.monotonic()
            try:
                m = {'UW': UnitWeibullQR, 'KW': KumaraswamyQR, 'Beta': BetaReg}[fam](tau).fit(X, y, b0)
                se = dpd.sandwich(m.v, X, y, fam, 0., 'logit'); mle[fam] = m.v
                R(name, m.v, se, m.success, m.message, -m.loglik / n, time.monotonic() - t, False)
                if fam == family: grid[0.] = (m.v, se)
            except Exception as ex:
                R(name, np.r_[b0, 0.], np.full(4, np.nan), False, repr(ex), np.nan, time.monotonic() - t, False)
        for name in [f'LMDPDE_{g}' for g in GRID] + ['MDPDE_0.3', 'BetaLMDPDE_0.3']:
            fam = 'Beta' if name.startswith('Beta') else family; scale = 'original' if name.startswith('MDPDE') else 'logit'
            g = float(name.rsplit('_', 1)[1]); t = time.monotonic(); boundary = False
            try:
                f = dpd.fit(X, y, fam, g, scale, b0, mle.get(fam)); v = f.x
                se = dpd.sandwich(v, X, y, fam, g, scale) if f.fun < dpd.PENALTY else np.full(4, np.nan)
                if name == 'MDPDE_0.3':
                    if fam == 'UW': boundary = bool(abs(v[-1]) < .01)
                    else:
                        a, lb = dpd.pars(v, X, fam); th = g / (1 + g)
                        boundary = bool(abs(v[-1] - np.log(th)) < .01 or np.min(abs(lb - np.log(th))) < .01)
                R(name, v, se, f.success, f.message, f.fun, time.monotonic() - t, boundary)
                if scale == 'logit' and fam == family: grid[g] = (v, se)
            except Exception as ex:
                R(name, np.r_[b0, 0.], np.full(4, np.nan), False, repr(ex), np.nan, time.monotonic() - t, False)
        # Iterated Warwick-Jones gamma selection over {0} + GRID (pilot 0.5), on the regression coefficients.
        usable = {g: (v, se) for g, (v, se) in grid.items() if np.all(np.isfinite(se[:3]))}
        gs = float('nan')
        if usable:
            pilot = .5 if .5 in usable else min(usable, key=lambda g: abs(g - .5))
            for _ in range(10):
                vp = usable[pilot][0][:3]
                crit = {g: float(np.sum((v[:3] - vp) ** 2) + np.sum(se[:3] ** 2)) for g, (v, se) in usable.items()}
                new = min(crit, key=crit.get)
                if new == pilot: break
                pilot = new
            gs = pilot
        src = next((r for r in rows if r['estimator'] == ('MLE' if gs == 0 else f'LMDPDE_{gs}')), None) if gs == gs else None
        if src is not None:
            auto = dict(src); auto['estimator'] = 'LMDPDE_auto'; auto['gamma_selected'] = gs; rows.append(auto)
        else:
            R('LMDPDE_auto', np.r_[b0, 0.], np.full(4, np.nan), False, 'no usable gamma', np.nan, 0., False)
    return rows


@app.function(image=image, cpu=1.0, memory=1024, timeout=3600, retries=2)
def run_batch(tasks):
    out = []
    for t in tasks: out.extend(run_dataset(*t))
    return out


@app.local_entrypoint()
def main(reps300: int = 300, reps100: int = 200, batch: int = 10, output: str = 'sim_tau.csv'):
    import pandas as pd
    out_dir = HERE.parent / 'results_tau'; out_dir.mkdir(exist_ok=True); path = out_dir / output; done = set()
    if path.exists():
        old = pd.read_csv(path)
        for key, g in old.groupby(['tau', 'setting', 'n', 'scenario', 'rep']):
            if len(g) == len(NAMES): done.add((float(key[0]), int(key[1]), int(key[2]), int(key[3][1:]), int(key[4])))
        old = old[old.set_index(['tau', 'setting', 'n', 'scenario', 'rep']).index.map(
            lambda k: (float(k[0]), int(k[1]), int(k[2]), int(k[3][1:]), int(k[4])) in done)]
        old.to_csv(path, index=False)  # drop any incomplete block before resuming
    tasks = [(si, n, ci, rep, tau) for tau in (.1, .9) for n in (300, 100) for si in range(4) for ci in range(6)
             for rep in range(reps300 if n == 300 else reps100) if (tau, si, n, ci, rep) not in done
             and not (si == 0 and tau == .9)]  # UW k=0.6 at tau=0.9 is degenerate: 48% of responses clip at 1e-12
    tasks.sort(key=lambda t: (t[3], t[4], t[1], t[0], t[2]))
    chunks = [tasks[i:i + batch] for i in range(0, len(tasks), batch)]
    print(f'START datasets={len(tasks)} batches={len(chunks)} already_done={len(done)}', flush=True)
    tick = time.monotonic(); n_done = 0; writer = None
    with path.open('a', newline='') as handle:
        for rows in run_batch.map(chunks, order_outputs=False):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                if handle.tell() == 0: writer.writeheader()
            writer.writerows(rows); handle.flush(); n_done += len(rows) // len(NAMES)
            el = time.monotonic() - tick; rate = n_done / el
            print(f'{n_done}/{len(tasks)} datasets  elapsed {el/60:.1f} min  ETA {(len(tasks)-n_done)/max(rate,1e-9)/60:.1f} min', flush=True)
    print(f'COMPLETE datasets={n_done} minutes={(time.monotonic()-tick)/60:.1f}', flush=True)
