"""Revised application figures: diagnostics with flagged points marked and common residual scales,
water scatter with common axes, and bootstrap intervals for the difference LMDPDE - MLE (water, gamma fixed at 0.5)."""
import os, sys, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
P = Path(__file__).resolve().parent; sys.path.insert(0, str(P)); sys.path.insert(0, str(P.parent / 'modal_run'))
from apps import design, resid, NAMES, GREY, BLUE, RED, FIG, TAB, ROOT
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
out = json.load(open(TAB / 'apps.json')); env = json.load(open(TAB / 'envelopes.json'))


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf'); fig.savefig(FIG / f'png/{name}.png', dpi=110); plt.close(fig)


def diag(ds, envkey, fname):
    d, X, y = design(ds); w = np.array(out[ds]['w']); q = expit(X @ np.array(out[ds]['vg'])[:-1]); f = w < .1
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    a = ax[0]; a.scatter(q[~f], w[~f], s=4 if ds == 'water' else 10, color=BLUE, alpha=.6); a.scatter(q[f], w[f], s=16, marker='x', color=RED)
    a.set_yscale('log'); a.axhline(.1, color='k', lw=.6, ls=':'); a.set_xlabel('fitted median'); a.set_ylabel('relative weight $w_i$')
    a.set_title(f'(a) Weights ({f.sum()} flagged)')
    if ds == 'HIC':
        for i in np.flatnonzero(f):
            if d.CITY[i] == 'CAMPINA DO MONTE ALEGRE':
                a.annotate('Campina do\nMonte Alegre', (q[i], w[i]), fontsize=6, xytext=(8, -2), textcoords='offset points', va='center')
            if d.CITY[i] == 'ARIRANHA':
                a.annotate('Ariranha', (q[i], w[i]), fontsize=6, xytext=(-6, 0), textcoords='offset points', ha='right', va='center')
    E = env[envkey]; lim = 0
    for k, (wk, key, lab, col) in enumerate((('mle', 'vm', 'MLE', '#4d4d4d'), ('lmdpde', 'vg', 'LMDPDE', BLUE)), start=1):
        r_obs = resid(out[ds][key], X, y); order = np.argsort(r_obs); fl_sorted = f[order]
        r = np.array(E[wk]); n = len(r); qq = norm.ppf((np.arange(1, n + 1) - .375) / (n + .25)); a = ax[k]
        lo, hi = np.array(E[wk + '_lo']), np.array(E[wk + '_hi']); outside = (r < lo) | (r > hi)
        a.fill_between(qq, lo, hi, color='#c9d9ee', lw=0); a.plot(qq, E[wk + '_mid'], color=GREY, lw=.7, ls='--')
        a.scatter(qq[~fl_sorted], r[~fl_sorted], s=3 if n > 500 else 9, color=col, zorder=3)
        a.scatter(qq[fl_sorted], r[fl_sorted], s=14 if n > 500 else 18, marker='x', color=RED, zorder=4, label='flagged')
        a.set_title(f'({"bc"[k-1]}) Residuals, {lab}'); a.set_xlabel('standard normal quantiles')
        a.text(.97, .05, f'{outside.sum()} of {n} outside', transform=a.transAxes, ha='right', fontsize=6.5)
        lim = max(lim, np.nanmax(np.abs(r)) + .5)
    for a in ax[1:]: a.set_ylim(-min(lim, 9), min(lim, 9))
    ax[1].set_ylabel('quantile residuals'); ax[2].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(); save(fig, fname)


def water_scatter():
    d = pd.read_csv(ROOT / 'data_real/csv/water.csv'); y = d.phpws.to_numpy(); flag = np.zeros(len(d), bool); flag[out['water']['flagged']] = True
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for k, (reg, nm) in enumerate(((0, 'Southeast'), (1, 'Northeast'))):
        m = d.region.to_numpy() == reg
        ax[k].scatter(np.log(d.incpc[m & ~flag]), y[m & ~flag], s=3, color='#9db8d9', alpha=.5, label='municipalities')
        ax[k].scatter(np.log(d.incpc[m & flag]), y[m & flag], s=14, marker='x', color=RED, label='flagged by LMDPDE')
        ax[k].set_xlabel('log per-capita income'); ax[k].set_title(f'{nm} ({m.sum()} municipalities)')
    ax[0].set_ylabel('proportion with piped water'); ax[0].legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout(); save(fig, 'fig10_water_scatter')


def water_diff():
    b = pd.read_csv(ROOT / 'results_real/boot.csv'); b = b[b.dataset == 'water']
    fig, a = plt.subplots(figsize=(5.2, 3.1)); ypos, ticks, tl, res = 0, [], [], {}
    for j, t in enumerate(NAMES['water'][1:], start=1):
        for tau in (.1, .5, .9):
            g = b[b.tau == tau]; D = np.array([json.loads(x)[j] for x in g.dpd]) - np.array([json.loads(x)[j] for x in g.mle])
            lo, md, hi = np.percentile(D, [2.5, 50, 97.5]); res[f'{t}_{tau}'] = [lo, md, hi]
            col = BLUE if (lo > 0 or hi < 0) else GREY
            a.plot([lo, hi], [ypos] * 2, color=col, lw=1.6); a.plot(md, ypos, 'o', color=col, ms=3)
            ticks.append(ypos); tl.append(f'{t}, $\\tau$={tau}'); ypos -= 1
        ypos -= .6
    a.axvline(0, color='k', lw=.6, ls=':'); a.set_yticks(ticks); a.set_yticklabels(tl, fontsize=7)
    a.set_xlabel('difference LMDPDE $-$ MLE')
    fig.tight_layout(); save(fig, 'app_forest'); json.dump(res, open(TAB / 'water_diff.json', 'w'), indent=1)
    for k, v in res.items(): print(k, np.round(v, 3))


diag('HIC', 'HIC_UW', 'app_hic_diag'); diag('water', 'water', 'app_water_diag'); water_scatter(); water_diff()
