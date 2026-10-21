"""Per-dataset application figures: diagnostics (weights + envelopes) and coefficient intervals across tau."""
import os, sys, json, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
P = Path(__file__).resolve().parent; sys.path.insert(0, str(P)); sys.path.insert(0, str(P.parent / 'modal_run'))
import apps
from apps import design, fits, NAMES, TITLE, SPEC, GREY, BLUE, RED, FIG, TAB
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
out = json.load(open(TAB / 'apps.json')); env = json.load(open(TAB / 'envelopes.json'))

def diag(ds, envkey, fname):
    d, X, y = design(ds); w = np.array(out[ds]['w']); q = expit(X @ np.array(out[ds]['vg'])[:-1]); f = w < .1
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a = ax[0]; a.scatter(q[~f], w[~f], s=4 if ds == 'water' else 10, color=BLUE, alpha=.6); a.scatter(q[f], w[f], s=16, marker='x', color=RED)
    a.set_yscale('log'); a.axhline(.1, color='k', lw=.6, ls=':'); a.set_xlabel('fitted median'); a.set_ylabel('relative weight $w_i$')
    a.set_title(f'(a) Weights ({f.sum()} flagged)')
    if ds == 'HIC':
        for i in np.flatnonzero(f):
            if d.CITY[i] in ('CAMPINA DO MONTE ALEGRE', 'ARIRANHA'):
                a.annotate(d.CITY[i].title().replace(' Do ', ' do '), (q[i], w[i]), fontsize=6, xytext=(-4, 4), textcoords='offset points', ha='right' if d.CITY[i] == 'ARIRANHA' else 'left')
    E = env[envkey]
    for k, (wk, lab, col) in enumerate((('mle', 'MLE', '#4d4d4d'), ('lmdpde', 'LMDPDE', BLUE)), start=1):
        r = np.array(E[wk]); n = len(r); qq = norm.ppf((np.arange(1, n + 1) - .375) / (n + .25)); a = ax[k]
        a.fill_between(qq, E[wk + '_lo'], E[wk + '_hi'], color='#c9d9ee', lw=0); a.plot(qq, E[wk + '_mid'], color=GREY, lw=.7, ls='--')
        a.scatter(qq, r, s=3 if n > 500 else 9, color=col, zorder=3)
        a.set_title(f'({"bc"[k-1]}) Residuals, {lab}'); a.set_xlabel('standard normal quantiles')
        lim = max(4, min(9, np.nanmax(np.abs(r)) + .5)); a.set_ylim(-lim, lim)
    ax[1].set_ylabel('quantile residuals')
    fig.tight_layout(); fig.savefig(FIG / f'{fname}.pdf'); fig.savefig(FIG / f'png/{fname}.png', dpi=110); plt.close(fig)

def hic_coef():
    import dpd
    d, X, y = design('HIC'); res = {}
    for tau in (.1, .5, .9):
        g = SPEC['HIC'][2][tau]; vm, vg, _ = fits(X, y, tau, g); dpd.TAU = tau
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            sm = dpd.sandwich(vm, X, y, 'UW', 0., 'logit'); sg = dpd.sandwich(vg, X, y, 'UW', g, 'logit')
        res[tau] = (vm, sm, vg, sg)
    fig, a = plt.subplots(figsize=(5.2, 2.6)); ypos, ticks, tl = 0, [], []
    for j, t in enumerate(NAMES['HIC'][1:], start=1):
        for tau in (.1, .5, .9):
            vm, sm, vg, sg = res[tau]
            for v, s, off, col in ((vm, sm, .15, GREY), (vg, sg, -.15, BLUE)):
                a.plot([v[j] - 1.96 * s[j], v[j] + 1.96 * s[j]], [ypos + off] * 2, color=col, lw=1.4); a.plot(v[j], ypos + off, 'o', color=col, ms=2.5)
            ticks.append(ypos); tl.append(f'{t}, $\\tau$={tau}'); ypos -= 1
        ypos -= .6
    a.axvline(0, color='k', lw=.6, ls=':'); a.set_yticks(ticks); a.set_yticklabels(tl, fontsize=7); a.set_xlabel('coefficient')
    from matplotlib.lines import Line2D
    a.legend([Line2D([], [], color=GREY), Line2D([], [], color=BLUE)], ['MLE', 'LMDPDE'], frameon=False, loc='lower right', fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'app_hic_coef.pdf'); fig.savefig(FIG / 'png/app_hic_coef.png', dpi=110); plt.close(fig)
    json.dump({str(t): {'mle': [float(x) for x in r[0][:-1]], 'mle_se': [float(x) for x in r[1][:-1]], 'lmd': [float(x) for x in r[2][:-1]], 'lmd_se': [float(x) for x in r[3][:-1]]} for t, r in res.items()}, open(TAB / 'hic_coef.json', 'w'), indent=1)

diag('HIC', 'HIC_UW', 'app_hic_diag'); diag('water', 'water', 'app_water_diag'); hic_coef(); print('done')
