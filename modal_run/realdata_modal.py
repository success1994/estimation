"""Bootstrap and cross-validation for the real-data showcase, in parallel on Modal.

Bootstrap: nonparametric case resampling; MLE and LMDPDE at the gamma selected on the full data
           (water UW 0.5; HIC KW 0.4; Firm KW 0.1). Gives percentile CIs to check the sandwich-based conclusions.
CV:        2 x 5-fold, held-out pinball loss at tau; models UW/KW MLE, UW/KW LMDPDE (gamma re-selected in each fold),
           LogitQR and beta LMDPDE(0.3). Predictors standardised with training-fold moments.

Run: .venv-modal/bin/modal run modal_run/realdata_modal.py          Output: results_real/boot.csv, results_real/cv.csv
"""
import sys, json, time, csv
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent; DATA = HERE.parent / 'data_real' / 'csv'
app = modal.App('robust-unit-qr-realdata')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.4.4', 'scipy==1.17.1', 'statsmodels==0.14.6', 'pandas==3.0.3', 'patsy==1.0.2')
         .env({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
         .add_local_file(HERE / 'dpd.py', '/root/app/dpd.py').add_local_file(HERE / 'models.py', '/root/app/models.py')
         .add_local_file(DATA / 'water.csv', '/root/data/water.csv').add_local_file(DATA / 'HIC.csv', '/root/data/HIC.csv')
         .add_local_file(DATA / 'Firm.csv', '/root/data/Firm.csv'))
SPECS = {'water': ('phpws', 'mhdi + np.log(incpc) + region + np.log(pop)'), 'HIC': ('HIC', 'URB + np.log(GDP)'),
         'Firm': ('FIRMCOST', 'SIZELOG + INDCOST')}
BOOT = {'water': ('UW', .5), 'HIC': ('KW', .4), 'Firm': ('KW', .1)}
GRID = [0., .1, .2, .3, .4, .5, .6, .7]


def _setup():
    for p in ('/root/app', str(HERE)):
        if p not in sys.path: sys.path.insert(0, p)


def _raw(ds):
    import numpy as np, pandas as pd, patsy
    root = Path('/root/data') if Path('/root/data').exists() else DATA
    d = pd.read_csv(root / f'{ds}.csv'); resp, form = SPECS[ds]
    X = patsy.dmatrix('1 + ' + form, d, return_type='dataframe'); return d[resp].to_numpy(float), X


def _std(X, rows):
    X = X.copy()
    for c in X.columns:
        if c != 'Intercept' and X[c].nunique() > 2:
            m, s = X[c].iloc[rows].mean(), X[c].iloc[rows].std(ddof=0); X[c] = (X[c] - m) / s
    return X.to_numpy()


def _select(fits):
    import numpy as np
    usable = {g: v for g, v in fits.items() if np.all(np.isfinite(v[1][:-1]))}
    pilot = .5 if .5 in usable else max(usable)
    for _ in range(10):
        vp = usable[pilot][0][:-1]
        crit = {g: float(np.sum((v[0][:-1] - vp) ** 2) + np.sum(v[1][:-1] ** 2)) for g, v in usable.items()}
        new = min(crit, key=crit.get)
        if new == pilot: break
        pilot = new
    return pilot


@app.function(image=image, cpu=1.0, memory=2048, timeout=3600, retries=2)
def boot_batch(tasks):
    _setup(); import numpy as np, warnings, dpd
    from models import UnitWeibullQR, KumaraswamyQR, LogitQR
    out = []
    for ds, tau, b in tasks:
        fam, g = BOOT[ds]; dpd.TAU = tau; y, Xdf = _raw(ds); n = len(y)
        rng = np.random.default_rng(np.random.SeedSequence([20261020, ['water', 'HIC', 'Firm'].index(ds), int(tau * 10), b]))
        idx = rng.integers(0, n, n); X = _std(Xdf, np.arange(n))[idx]; yy = y[idx]
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            b0 = LogitQR(tau).fit(X, yy, None).v
            m = {'UW': UnitWeibullQR, 'KW': KumaraswamyQR}[fam](tau).fit(X, yy, b0)
            f = dpd.fit(X, yy, fam, g, 'logit', b0, m.v)
        out.append(dict(dataset=ds, family=fam, tau=tau, gamma=g, b=b, mle=json.dumps(m.v[:-1].tolist()),
                        dpd=json.dumps(f.x[:-1].tolist()), ok=bool(m.success and f.success)))
    return out


@app.function(image=image, cpu=1.0, memory=2048, timeout=7200, retries=2)
def cv_task(task):
    _setup(); import numpy as np, warnings, dpd
    from scipy.special import expit
    from scipy.stats import beta as beta_dist
    from models import UnitWeibullQR, KumaraswamyQR, BetaReg, LogitQR
    ds, tau, rep, fold = task; dpd.TAU = tau; y, Xdf = _raw(ds); n = len(y)
    rng = np.random.default_rng(np.random.SeedSequence([20261017, ['water', 'HIC', 'Firm'].index(ds), rep]))
    part = np.array_split(rng.permutation(n), 5)[fold]; te = np.sort(part); tr = np.setdiff1d(np.arange(n), te)
    X = _std(Xdf, tr); Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]; preds = {}; gam = {}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        q = LogitQR(tau).fit(Xtr, ytr, None); preds['LogitQR'] = expit(Xte @ q.v)
        for fam in ('UW', 'KW'):
            m = {'UW': UnitWeibullQR, 'KW': KumaraswamyQR}[fam](tau).fit(Xtr, ytr, q.v)
            fits = {0.: (m.v, dpd.sandwich(m.v, Xtr, ytr, fam, 0., 'logit'))}
            for g in GRID[1:]:
                f = dpd.fit(Xtr, ytr, fam, g, 'logit', q.v, m.v); fits[g] = (f.x, dpd.sandwich(f.x, Xtr, ytr, fam, g, 'logit'))
            g = _select(fits); gam[fam] = g
            preds[f'{fam}-MLE'] = expit(Xte @ m.v[:-1]); preds[f'{fam}-LMDPDE_auto'] = expit(Xte @ fits[g][0][:-1])
        bm = BetaReg(tau).fit(Xtr, ytr, q.v); fb = dpd.fit(Xtr, ytr, 'Beta', .3, 'logit', q.v, bm.v)
        mu = expit(Xte @ fb.x[:-1]); phi = np.exp(fb.x[-1]); preds['Beta-LMDPDE_0.3'] = beta_dist.ppf(tau, mu * phi, (1 - mu) * phi)
    rows = []
    for name, p in preds.items():
        u = yte - p; loss = u * (tau - (u < 0))
        for i, l in zip(te, loss): rows.append(dict(dataset=ds, tau=tau, rep=rep, fold=fold, record=int(i), model=name, loss=float(l),
                                                    gamma_uw=gam['UW'], gamma_kw=gam['KW']))
    return rows


@app.local_entrypoint()
def main(boot_reps: int = 300):
    out = HERE.parent / 'results_real'; out.mkdir(exist_ok=True)
    boot_tasks = [(ds, tau, b) for ds in BOOT for tau in (.1, .5, .9) for b in range(boot_reps)]
    chunks = []; i = 0  # water fits are slow: 2 per batch; HIC/Firm: 25 per batch
    while i < len(boot_tasks):
        size = 2 if boot_tasks[i][0] == 'water' else 25; chunks.append(boot_tasks[i:i + size]); i += size
    cv_tasks = [(ds, tau, rep, fold) for ds in SPECS for tau in (.1, .5, .9) for rep in range(2) for fold in range(5)]
    print(f'START boot_tasks={len(boot_tasks)} in {len(chunks)} batches; cv_tasks={len(cv_tasks)}', flush=True)
    tick = time.monotonic()
    boot_calls = boot_batch.map(chunks, order_outputs=False)
    cv_handle = [cv_task.spawn(t) for t in cv_tasks]
    with (out / 'boot.csv').open('w', newline='') as h:
        w = None; done = 0
        for rows in boot_calls:
            if w is None: w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader()
            w.writerows(rows); h.flush(); done += 1
            if done % 25 == 0 or done == len(chunks):
                print(f'boot {done}/{len(chunks)} batches, {(time.monotonic() - tick) / 60:.1f} min', flush=True)
    with (out / 'cv.csv').open('w', newline='') as h:
        w = None
        for k, c in enumerate(cv_handle, 1):
            rows = c.get()
            if w is None: w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader()
            w.writerows(rows); h.flush()
            if k % 10 == 0 or k == len(cv_handle): print(f'cv {k}/{len(cv_handle)}, {(time.monotonic() - tick) / 60:.1f} min', flush=True)
    print(f'COMPLETE {(time.monotonic() - tick) / 60:.1f} min', flush=True)
