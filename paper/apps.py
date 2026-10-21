"""Applications (HIC and water, unit-Weibull, chosen by AIC): tables and figures for the paper."""
import os, sys, json, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit

P = Path(__file__).resolve().parent; ROOT = P.parent; FIG = P / 'figures'; TAB = P / 'tables'
sys.path.insert(0, str(ROOT / 'modal_run'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})
GREY, BLUE, ORANGE, RED = '#6b6b6b', '#1f5fa8', '#d9822b', '#b8323a'
SPEC = {'HIC': ('HIC', 'URB + np.log(GDP)', {.1: .5, .5: .7, .9: .7}),
        'water': ('phpws', 'mhdi + np.log(incpc) + region + np.log(pop)', {.1: .5, .5: .5, .9: .5})}
NAMES = {'HIC': ['Intercept', 'URB', 'log(GDP)'], 'water': ['Intercept', 'MHDI', 'log(income)', 'Region (NE)', 'log(population)']}
TITLE = {'HIC': 'Health insurance', 'water': 'Piped water'}


def design(ds, rows=None):
    import patsy
    resp, form, _ = SPEC[ds]; d = pd.read_csv(ROOT / 'data_real/csv' / f'{ds}.csv'); y = d[resp].to_numpy(float)
    Xd = patsy.dmatrix('1 + ' + form, d, return_type='dataframe')
    for c in Xd.columns:
        if c != 'Intercept' and Xd[c].nunique() > 2: Xd[c] = (Xd[c] - Xd[c].mean()) / Xd[c].std(ddof=0)
    return d, Xd.to_numpy(), y


def fits(X, y, tau, g):
    import dpd
    from models import UnitWeibullQR, LogitQR
    dpd.TAU = tau
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        b0 = LogitQR(tau).fit(X, y, None).v
        m = UnitWeibullQR(tau).fit(X, y, b0); f = dpd.fit(X, y, 'UW', g, 'logit', b0, m.v)
    return m.v, f.x, b0


def weights(v, X, y, g):
    import dpd
    lh = dpd.logpdf(v, X, y, 'UW') + np.log(y) + np.log1p(-y)
    return np.exp(g * (lh - np.median(lh)))


def boot_one(args):
    ds, tau, b = args
    d, X, y = design(ds); g = SPEC[ds][2][tau]; n = len(y)
    rng = np.random.default_rng(np.random.SeedSequence([20261020, 7, int(tau * 10), b])); idx = rng.integers(0, n, n)
    vm, vg, _ = fits(X[idx], y[idx], tau, g)
    return ds, tau, b, vm[:-1].tolist(), vg[:-1].tolist()


def env_one(args):
    ds, which, v, seed = args
    import dpd
    dpd.TAU = .5; d, X, y = design(ds); g = SPEC[ds][2][.5]; rng = np.random.default_rng(seed)
    a, lb = dpd.pars(np.array(v), X, 'UW'); al = np.exp(lb)
    with np.errstate(all='ignore'): ys = np.clip(np.exp(-(rng.exponential(size=len(X)) / al) ** (1 / a)), 1e-10, 1 - 1e-10)
    vm, vg, _ = fits(X, ys, .5, g)
    return np.sort(resid(vm if which == 'mle' else vg, X, ys))


def resid(v, X, y):
    import dpd
    a, lb = dpd.pars(np.asarray(v), X, 'UW')
    return norm.ppf(np.clip(np.exp(-np.exp(lb) * (-np.log(y)) ** a), 1e-15, 1 - 1e-15))


def main():
    import dpd
    out = {}
    # ---- fits at tau = 0.5: MLE, MLE without flagged, LMDPDE
    lines = []
    for ds in ('HIC', 'water'):
        d, X, y = design(ds); g = SPEC[ds][2][.5]
        vm, vg, b0 = fits(X, y, .5, g); w = weights(vg, X, y, g); fl = np.flatnonzero(w < .1); keep = np.setdiff1d(np.arange(len(y)), fl)
        from models import UnitWeibullQR
        dpd.TAU = .5
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            vd = UnitWeibullQR(.5).fit(X[keep], y[keep], b0).v
            se = {'m': dpd.sandwich(vm, X, y, 'UW', 0., 'logit'), 'd': dpd.sandwich(vd, X[keep], y[keep], 'UW', 0., 'logit'),
                  'g': dpd.sandwich(vg, X, y, 'UW', g, 'logit')}
        V = {'m': vm, 'd': vd, 'g': vg}
        out[ds] = dict(flagged=fl.tolist(), w=w.tolist(), vm=vm.tolist(), vg=vg.tolist(), vd=vd.tolist(),
                       se={k: s.tolist() for k, s in se.items()}, gamma=g, n=len(y))
        lines.append(f'\\multicolumn{{10}}{{l}}{{\\textit{{{TITLE[ds]} ($n={len(y)}$, $\\hat\\gamma={g}$, {len(fl)} flagged observations)}}}}\\\\')
        for j, t in enumerate(NAMES[ds] + ['$\\log k$']):
            cells = []
            for k in 'mdg':
                z = V[k][j] / se[k][j]; p = 2 * norm.sf(abs(z))
                cells += [f'{V[k][j]:.3f}', f'{se[k][j]:.3f}', '$<$0.001' if p < .001 else f'{p:.3f}']
            lines.append(f'\\quad {t} & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\addlinespace')
    (TAB / 'app_real.tex').write_text('\n'.join(lines[:-1]) + '\n')

    # ---- descriptive statistics
    rows = []
    h = pd.read_csv(ROOT / 'data_real/csv/HIC.csv'); wd = pd.read_csv(ROOT / 'data_real/csv/water.csv')
    for ds, lab, s in (('HIC', 'HIC (response)', h.HIC), ('HIC', 'URB', h.URB), ('HIC', 'GDP per capita (R\\$)', h.GDP),
                       ('water', 'Piped-water proportion (response)', wd.phpws), ('water', 'MHDI', wd.mhdi),
                       ('water', 'Income per capita (R\\$)', wd.incpc), ('water', 'Population', wd['pop'])):
        fmt = (lambda x: f'{x:,.0f}') if s.max() > 50 else (lambda x: f'{x:.3f}')
        rows.append(f'{lab} & ' + ' & '.join(fmt(v) for v in (s.mean(), s.median(), s.std(), s.min(), s.max())) + r'\\')
    rows.insert(3, r'\addlinespace')
    rows.insert(0, r'\multicolumn{6}{l}{\textit{Health insurance ($n=80$)}}\\'); rows.insert(5, r'\multicolumn{6}{l}{\textit{Piped water ($n=3457$; 52\% of municipalities in the Northeast)}}\\')
    (TAB / 'app_desc.tex').write_text('\n'.join(rows) + '\n')

    # ---- CV table (two data sets)
    c = pd.read_csv(ROOT / 'results_real/cv.csv'); c = c[c.dataset.isin(['HIC', 'water'])]
    rec = c.groupby(['dataset', 'tau', 'model', 'record']).loss.mean().reset_index()
    tab = rec.groupby(['dataset', 'tau', 'model']).loss.mean().unstack('model') * 1000
    order = ['UW-MLE', 'UW-LMDPDE_auto', 'KW-MLE', 'KW-LMDPDE_auto', 'LogitQR', 'Beta-LMDPDE_0.3']
    L = []
    for (ds, t), r in tab.iterrows():
        best = r[order].idxmin()
        L.append(f'{TITLE[ds]} & {t} & ' + ' & '.join((r'\textbf{%.2f}' if m == best else '%.2f') % r[m] for m in order) + r'\\')
    (TAB / 'app_cv.tex').write_text('\n'.join(L) + '\n')

    # ---- bootstrap (HIC recomputed under UW; water from the Modal run, UW, gamma 0.5)
    with ProcessPoolExecutor(9) as pool:
        bh = list(pool.map(boot_one, [('HIC', t, b) for t in (.1, .5, .9) for b in range(300)], chunksize=10))
    boot = {('HIC', t): ([], []) for t in (.1, .5, .9)}
    for ds, t, b, m, g in bh: boot[(ds, t)][0].append(m); boot[(ds, t)][1].append(g)
    bw = pd.read_csv(ROOT / 'results_real/boot.csv'); bw = bw[bw.dataset == 'water']
    for t in (.1, .5, .9):
        gg = bw[bw.tau == t]; boot[('water', t)] = ([json.loads(x) for x in gg.mle], [json.loads(x) for x in gg.dpd])
    bsum = {}
    for (ds, t), (m, g) in boot.items():
        m, g = np.array(m), np.array(g)
        bsum[f'{ds}_{t}'] = {'mle': np.percentile(m, [2.5, 50, 97.5], axis=0).tolist(), 'lmdpde': np.percentile(g, [2.5, 50, 97.5], axis=0).tolist()}
    out['boot'] = bsum

    # ---- envelopes (HIC under UW; water reuses the UW envelopes already computed)
    env = json.load(open(TAB / 'envelopes.json')); B = 100
    jobs = [('HIC', w, out['HIC']['vm' if w == 'mle' else 'vg'], 5000 + 100 * i + r) for i, w in enumerate(('mle', 'lmdpde')) for r in range(B)]
    with ProcessPoolExecutor(9) as pool:
        res = list(pool.map(env_one, jobs, chunksize=5))
    d, X, y = design('HIC'); envH = {}
    for i, w in enumerate(('mle', 'lmdpde')):
        S = np.array(res[i * B:(i + 1) * B]); obs = np.sort(resid(out['HIC']['vm' if w == 'mle' else 'vg'], X, y))
        lo, hi = np.percentile(S, 2.5, axis=0), np.percentile(S, 97.5, axis=0)
        envH[w] = obs.tolist(); envH[w + '_lo'] = lo.tolist(); envH[w + '_hi'] = hi.tolist(); envH[w + '_mid'] = np.median(S, axis=0).tolist()
        envH[w + '_outside'] = int(np.sum((obs < lo) | (obs > hi)))
    env['HIC_UW'] = envH; out['env_outside'] = {w: envH[w + '_outside'] for w in ('mle', 'lmdpde')}
    json.dump(env, open(TAB / 'envelopes.json', 'w')); json.dump(out, open(TAB / 'apps.json', 'w'))
    print('HIC envelope outside (MLE, LMDPDE):', out['env_outside'])
    figures(out, env)


def figures(out, env):
    hic = pd.read_csv(ROOT / 'data_real/csv/HIC.csv'); wd = pd.read_csv(ROOT / 'data_real/csv/water.csv')
    flH = np.array(out['HIC']['flagged']); flW = np.array(out['water']['flagged'])
    # Exploratory plot for HIC
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    m = np.zeros(len(hic), bool); m[flH] = True
    for a, x, xl in ((ax[0], hic.URB, 'urbanisation rate (URB)'), (ax[1], np.log(hic.GDP), 'log GDP per capita')):
        a.scatter(x[~m], hic.HIC[~m], s=12, color='#9db8d9', label='cities'); a.scatter(x[m], hic.HIC[m], s=26, marker='x', color=RED, label='flagged by LMDPDE')
        a.set_xlabel(xl)
        for i in flH:
            if hic.HIC[i] > .5 or hic.GDP[i] > 150000: a.annotate(hic.CITY[i].title().replace(' Do ', ' do '), (x[i], hic.HIC[i]), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax[0].set_ylabel('health insurance coverage'); ax[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(FIG / 'app_hic_scatter.pdf'); fig.savefig(FIG / 'png/app_hic_scatter.png', dpi=110); plt.close(fig)
    # Weights
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a, ds in zip(ax, ('HIC', 'water')):
        d, X, y = design(ds); w = np.array(out[ds]['w']); q = expit(X @ np.array(out[ds]['vg'])[:-1]); f = w < .1
        a.scatter(q[~f], w[~f], s=4 if ds == 'water' else 10, color=BLUE, alpha=.6); a.scatter(q[f], w[f], s=16, marker='x', color=RED)
        a.set_yscale('log'); a.set_xlabel('fitted median'); a.axhline(.1, color='k', lw=.6, ls=':')
        a.set_title(f'{"(a)" if ds == "HIC" else "(b)"} {TITLE[ds]} ($\\hat\\gamma$={out[ds]["gamma"]}; {f.sum()} flagged)')
        if ds == 'HIC':
            for i in np.flatnonzero(f):
                if d.CITY[i] in ('CAMPINA DO MONTE ALEGRE', 'JAMBEIRO', 'ARIRANHA', 'SANTO ANTONIO DO ARACANGUA'):
                    a.annotate(d.CITY[i].title().replace(' Do ', ' do '), (q[i], w[i]), fontsize=6, xytext=(3, -2), textcoords='offset points')
    ax[0].set_ylabel('relative weight $w_i$')
    fig.tight_layout(); fig.savefig(FIG / 'app_weights.pdf'); fig.savefig(FIG / 'png/app_weights.png', dpi=110); plt.close(fig)
    # Envelopes 2x2
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for j, key in enumerate(('HIC_UW', 'water')):
        E = env[key]
        for i, (w, lab, col) in enumerate((('mle', 'MLE', '#4d4d4d'), ('lmdpde', 'LMDPDE', BLUE))):
            r = np.array(E[w]); n = len(r); qq = norm.ppf((np.arange(1, n + 1) - .375) / (n + .25)); a = ax[i, j]
            a.fill_between(qq, E[w + '_lo'], E[w + '_hi'], color='#c9d9ee', lw=0); a.plot(qq, E[w + '_mid'], color=GREY, lw=.7, ls='--')
            a.scatter(qq, r, s=3 if n > 500 else 9, color=col, zorder=3)
            a.set_title(f'{TITLE["HIC" if j == 0 else "water"]}: {lab}', fontsize=8.5)
            lim = max(4, min(9, np.nanmax(np.abs(r)) + .5)); a.set_ylim(-lim, lim)
            if i == 1: a.set_xlabel('standard normal quantiles')
            if j == 0: a.set_ylabel('quantile residuals')
    fig.tight_layout(); fig.savefig(FIG / 'app_envelopes.pdf'); fig.savefig(FIG / 'png/app_envelopes.png', dpi=110); plt.close(fig)
    # Bootstrap forest (water)
    fig, a = plt.subplots(figsize=(5.2, 3.3))
    ds = 'water'; ypos, ticks, tl = 0, [], []
    for j, t in enumerate(NAMES[ds][1:]):
        for tau in (.1, .5, .9):
            b = out['boot'][f'{ds}_{tau}']
            for est, off, col in (('mle', .15, GREY), ('lmdpde', -.15, BLUE)):
                lo, md, hi = (b[est][k][j + 1] for k in range(3))
                a.plot([lo, hi], [ypos + off] * 2, color=col, lw=1.4); a.plot(md, ypos + off, 'o', color=col, ms=2.5)
            ticks.append(ypos); tl.append(f'{t}, $\\tau$={tau}'); ypos -= 1
        ypos -= .6
    a.axvline(0, color='k', lw=.6, ls=':'); a.set_yticks(ticks); a.set_yticklabels(tl, fontsize=7); a.set_xlabel('coefficient')
    from matplotlib.lines import Line2D
    a.legend([Line2D([], [], color=GREY), Line2D([], [], color=BLUE)], ['MLE', 'LMDPDE'], frameon=False, loc='lower right', fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'app_forest.pdf'); fig.savefig(FIG / 'png/app_forest.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--figs':
        figures(json.load(open(TAB / 'apps.json')), json.load(open(TAB / 'envelopes.json')))
    else:
        main()
