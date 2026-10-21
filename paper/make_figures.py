"""Figures and LaTeX tables for the paper. Run with system python3 from the paper/ folder:  python3 make_figures.py"""
import os, sys, json, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.special import expit, betaln

P = Path(__file__).resolve().parent; ROOT = P.parent; FIG = P / 'figures'; TAB = P / 'tables'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'modal_run'))
import dpd
from models import UnitWeibullQR, KumaraswamyQR, LogitQR

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
COL = {'MLE': '#6b6b6b', 'LMDPDE_auto': '#1f5fa8', 'LMDPDE_0.3': '#7fa7d6', 'MDPDE_0.3': '#d9822b', 'LogitQR': '#3a9a5b',
       'BetaLMDPDE_0.3': '#b8323a'}
LAB = {'MLE': 'MLE', 'LMDPDE_auto': r'LMDPDE ($\hat\gamma$)', 'LMDPDE_0.3': r'LMDPDE ($\gamma$=0.3)', 'MDPDE_0.3': r'MDPDE ($\gamma$=0.3)',
       'LogitQR': 'Logit-QR', 'BetaLMDPDE_0.3': 'Beta LMDPDE'}
SET = {0: 'UW, $k$=0.6', 1: 'UW, $k$=1.5', 2: 'KW, $a$=0.6', 3: 'KW, $a$=2'}
SC = {'C0': 'C0 clean', 'C1': 'C1 5% gross', 'C2': 'C2 10% gross', 'C3': 'C3 leverage', 'C4': 'C4 5% moderate', 'C5': 'C5 10% moderate'}
EST = ['MLE', 'LMDPDE_auto', 'LMDPDE_0.3', 'MDPDE_0.3', 'LogitQR']


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf'); (FIG / 'png').mkdir(exist_ok=True); fig.savefig(FIG / 'png' / f'{name}.png', dpi=110)
    plt.close(fig); print('figure', name)


# ---------- Figure 1: nonexistence of the original-scale DPD integral ----------
def fig_nonexistence():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    g = .3; th = g / (1 + g)
    a = np.linspace(0, 1.5, 400); b = np.linspace(0, 1.5, 400); A, B = np.meshgrid(a, b)
    ok = (A > th) & (B > th)
    ax[0].contourf(A, B, ok, levels=[-.5, .5, 1.5], colors=['#e39a9a', '#cfdff2'])
    ax[0].axvline(th, color='k', lw=.6, ls='--'); ax[0].axhline(th, color='k', lw=.6, ls='--')
    ax[0].text(.85, .85, 'finite', ha='center'); ax[0].text(th / 2, .85, 'infinite', rotation=90, va='center', ha='center', fontsize=7.5, fontweight='bold')
    ax[0].text(.85, th / 2, 'infinite', va='center', ha='center', fontsize=7.5, fontweight='bold')
    ax[0].set_xlim(0, 1.5); ax[0].set_ylim(0, 1.5)
    ax[0].set_xlabel('$a$'); ax[0].set_ylabel('$b$'); ax[0].set_title(r'(a) Kumaraswamy, $\int f^{1+\gamma}<\infty$ ($\gamma$=0.3)')
    # (b) unit-Weibull: truncated original-scale integral vs cut-off in t = -log y
    T = np.linspace(1, 80, 200)
    for k, c in ((.6, '#b8323a'), (1.0, '#d9822b'), (1.5, '#1f5fa8')):
        al = 1.
        t = np.linspace(1e-6, 80, 400001); dt = t[1] - t[0]
        integrand = (al * k) ** (1 + g) * t ** ((k - 1) * (1 + g)) * np.exp(g * t - al * (1 + g) * t ** k)
        cum = np.cumsum(integrand) * dt
        ax[1].plot(T, np.interp(T, t, cum), color=c, label=f'$k$={k}')
    ax[1].set_yscale('log'); ax[1].set_xlabel('upper limit of $t=-\\log y$'); ax[1].set_ylabel('truncated integral')
    ax[1].set_title(r'(b) Unit-Weibull, original scale'); ax[1].legend(frameon=False)
    # (c) same on logit scale: converges for all k
    for k, c in ((.6, '#b8323a'), (1.0, '#d9822b'), (1.5, '#1f5fa8')):
        mu = np.exp(-(np.log(2)) ** (1 / k)); v = np.array([np.log(mu / (1 - mu)), np.log(k)])
        dpd.TAU = .5
        vals = []
        for Z in T:
            z = np.linspace(-Z, Z, 20001); y = expit(z)
            with np.errstate(all='ignore'):
                lh = dpd.logpdf(v, np.ones((len(y), 1)), y, 'UW') + np.log(y) + np.log1p(-y)
            vals.append(np.trapezoid(np.exp((1 + g) * np.nan_to_num(lh, nan=-np.inf)), z))
        ax[2].plot(T, vals, color=c, label=f'$k$={k}')
    ax[2].set_xlabel('half-width of logit range'); ax[2].set_ylabel('truncated integral'); ax[2].set_ylim(0, 1)
    ax[2].set_title(r'(c) Unit-Weibull, logit scale')
    fig.tight_layout(); save(fig, 'fig1_nonexistence')


# ---------- Figure 2: simulation designs ----------
def fig_designs():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1.05]))
    y = np.linspace(1e-4, 1 - 1e-4, 2000); X1 = np.ones((len(y), 1))
    dpd.TAU = .5
    for si, (fam, shape) in enumerate([('UW', .6), ('UW', 1.5), ('KW', .6), ('KW', 2.)]):
        v = np.array([np.log(.35 / .65), np.log(shape)])
        with np.errstate(all='ignore'): f = np.exp(dpd.logpdf(v, X1, y, fam))
        ax[0].plot(y, f, label=SET[si], lw=1.2)
    ax[0].set_ylim(0, 5); ax[0].set_xlabel('$y$'); ax[0].set_ylabel('density'); ax[0].legend(frameon=False)
    ax[0].set_title('(a) Densities with conditional median 0.35')
    sys.path.insert(0, str(ROOT / 'code')); import simulate_full as sf
    X, yy, _, _ = sf.generate(3, 300, 0, 0); mu = expit(X @ np.array([-.5, .8, -.6])); r = np.random.default_rng(7)
    idx = r.choice(300, 30, replace=False); keep = np.setdiff1d(np.arange(300), idx)
    yo = np.where(mu[idx] < .5, r.uniform(.95, .995, 30), r.uniform(.005, .05, 30))
    ax[1].scatter(X[keep, 1], yy[keep], s=5, color='#6b6b6b', alpha=.6, label='clean observations')
    ax[1].scatter(X[idx, 1], yo, s=14, marker='x', color='#b8323a', label='10% gross outliers (C2)')
    lev = r.uniform(.005, .05, 15); ax[1].scatter(np.full(15, 3.), lev, s=14, marker='^', color='#d9822b', label='bad leverage, $x_1$=3 (C3)')
    xs = np.linspace(-3, 3.2, 100); ax[1].plot(xs, expit(-.5 + .8 * xs), color='#1f5fa8', lw=1.2, label='true median ($x_2$=0)')
    ax[1].set_xlabel('$x_1$'); ax[1].set_ylabel('$y$'); ax[1].legend(frameon=False, loc='upper left', fontsize=7, bbox_to_anchor=(1.0, 1.0))
    ax[1].set_title('(b) One sample, KW $a$=2, $n$=300')
    fig.tight_layout(); save(fig, 'fig2_designs')


def load_sim():
    a = pd.read_csv(ROOT / 'results_full/sim_fits.csv', usecols=['setting', 'n', 'scenario', 'rep', 'estimator', 'beta1', 'beta2',
            'slope_squared_error', 'hit1', 'hit2', 'converged', 'gamma_selected'])
    a['tau'] = .5
    b = pd.read_csv(ROOT / 'results_tau/sim_tau.csv', usecols=['setting', 'n', 'tau', 'scenario', 'rep', 'estimator', 'beta1', 'beta2',
            'slope_squared_error', 'hit1', 'hit2', 'converged', 'gamma_selected'])
    return pd.concat([a, b], ignore_index=True)


# ---------- Figure 3: boxplots of beta1 ----------
def fig_boxplots(d):
    sub = d[(d.n == 300) & (d.tau == .5) & d.estimator.isin(EST)]
    scs = ['C0', 'C2', 'C3', 'C5']
    fig, ax = plt.subplots(4, 4, figsize=(7.2, 7.4), sharey='row')
    for i in range(4):
        for j, sc in enumerate(scs):
            c = sub[(sub.setting == i) & (sub.scenario == sc)]
            ok = {e: not (e == 'MDPDE_0.3' and 1 - c[c.estimator == e].converged.mean() > .5) for e in EST}
            data = [c[c.estimator == e].beta1.to_numpy() if ok[e] else [np.nan] for e in EST]
            bp = ax[i, j].boxplot(data, widths=.6, patch_artist=True, showfliers=False, medianprops=dict(color='k', lw=.8))
            for patch, e in zip(bp['boxes'], EST): patch.set_facecolor(COL[e]); patch.set_alpha(.85)
            for k, e in enumerate(EST):
                if not ok[e]: ax[i, j].text(k + 1, .5, 'failed', rotation=90, ha='center', va='center', fontsize=7, color='#d9822b', transform=ax[i, j].get_xaxis_transform())
            ax[i, j].axhline(.8, color='k', ls='--', lw=.8, zorder=0)
            ax[i, j].set_xticks(range(1, 6)); ax[i, j].set_xticklabels(['MLE', r'L-$\hat\gamma$', 'L-.3', 'M-.3', 'LQR'], fontsize=7.5, rotation=45)
            if i == 0: ax[i, j].set_title(SC[sc])
            if j == 0: ax[i, j].set_ylabel(SET[i] + '\n' + r'$\hat\beta_1$')
    fig.tight_layout(); save(fig, 'fig3_boxplots_beta1')


# ---------- Figure 4: relative MSE heatmaps ----------
def fig_heatmap(d):
    sub = d[(d.n == 300)]
    mse = sub.groupby(['tau', 'setting', 'scenario', 'estimator']).slope_squared_error.mean().unstack('estimator')
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.8))
    norm = TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
    for k, ref in enumerate(['MLE', 'LogitQR']):
        r = np.log2(mse[ref] / mse['LMDPDE_auto']).unstack('scenario')
        rows = [(t, s) for t in (.1, .5, .9) for s in range(4) if (t, s) in r.index]
        M = np.array([[r.loc[rw, sc] for sc in SC] for rw in rows])
        im = ax[k].imshow(M, cmap='RdBu', norm=norm, aspect='auto')
        for (ii, jj), val in np.ndenumerate(M):
            ax[k].text(jj, ii, f'{2 ** val:.2f}', ha='center', va='center', fontsize=6, color='w' if abs(val) > 2.3 else 'k')
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(list(SC.values()), rotation=35, ha='right')
        ax[k].set_yticks(range(len(rows))); ax[k].set_yticklabels([rf'$\tau$={t}: {SET[s]}' for t, s in rows] if k == 0 else [], fontsize=7)
        ax[k].set_title(f'MSE ratio: {LAB[ref]} / Ours')
        ax[k].spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=.8, ticks=[-4, -2, 0, 2, 4]); cb.ax.set_yticklabels(['1/16', '1/4', '1', '4', '16'])
    cb.set_label('ratio (log scale); blue = ours more accurate')
    save(fig, 'fig4_mse_ratio_heatmap')
    return mse


# ---------- Figure 5: coverage ----------
def fig_coverage(d):
    sub = d[(d.n == 300) & d.estimator.isin(['MLE', 'LMDPDE_auto', 'LogitQR'])].copy()
    sub['cover'] = (sub.hit1.astype(float) + sub.hit2.astype(float)) / 2
    cv = sub.groupby(['tau', 'scenario', 'estimator']).cover.mean().unstack('estimator')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for k, t in enumerate((.1, .5, .9)):
        c = cv.loc[t]; x = np.arange(len(c))
        for off, e in zip((-.25, 0, .25), ['MLE', 'LMDPDE_auto', 'LogitQR']):
            ax[k].bar(x + off, c[e], width=.25, color=COL[e], label=LAB[e])
        ax[k].axhline(.95, color='k', ls='--', lw=.7); ax[k].set_ylim(.3, 1)
        ax[k].set_xticks(x); ax[k].set_xticklabels([SC[s] for s in c.index], rotation=35, ha='right', fontsize=7)
        ax[k].set_title(rf'$\tau$ = {t}')
    ax[0].set_ylabel('coverage of 95% intervals')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.08))
    fig.tight_layout(); save(fig, 'fig5_coverage')
    return cv


# ---------- Figure 6: selected gamma ----------
def fig_gamma(d):
    sub = d[(d.n == 300) & (d.estimator == 'LMDPDE_auto')]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    grid = [0, .1, .2, .3, .4, .5, .6, .7]
    for k, t in enumerate((.1, .5, .9)):
        s = sub[sub.tau == t]
        M = np.array([[np.mean(np.isclose(s[s.scenario == sc].gamma_selected, g)) for sc in SC] for g in grid])
        im = ax[k].imshow(M, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(list(SC.values()), rotation=35, ha='right', fontsize=7)
        ax[k].set_yticks(range(8)); ax[k].set_yticklabels(grid); ax[k].set_title(rf'$\tau$ = {t}')
    ax[0].set_ylabel(r'selected $\gamma$')
    fig.colorbar(im, ax=ax, shrink=.85, label='proportion of replications')
    save(fig, 'fig6_gamma_selected')


# ---------- Figure 7: original-scale MDPDE failure ----------
def fig_mdpde_fail(d):
    sub = d[(d.estimator == 'MDPDE_0.3') & (d.n == 300)]
    f = 1 - sub.groupby(['tau', 'setting']).converged.mean()
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    labels = [rf'$\tau$={t}' + '\n' + SET[s] for t, s in f.index]
    ax.bar(range(len(f)), f.values, color=['#b8323a' if v > .5 else '#7fa7d6' for v in f.values])
    ax.set_xticks(range(len(f))); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel('failure rate'); ax.set_ylim(0, 1.05); ax.set_title('Original-scale MDPDE ($\\gamma$=0.3), $n$=300, all scenarios')
    for i, v in enumerate(f.values): ax.text(i, v + .02, f'{v:.2f}', ha='center', fontsize=6)
    save(fig, 'fig7_mdpde_failure')
    return f


# ---------- Real data ----------
SPECS = {'water': ('phpws', 'mhdi + np.log(incpc) + region + np.log(pop)', 'UW', .5),
         'HIC': ('HIC', 'URB + np.log(GDP)', 'KW', .4), 'Firm': ('FIRMCOST', 'SIZELOG + INDCOST', 'KW', .1)}


def fit_real(ds, tau):
    import patsy
    resp, form, fam, g = SPECS[ds]; d = pd.read_csv(ROOT / 'data_real/csv' / f'{ds}.csv'); y = d[resp].to_numpy(float)
    Xd = patsy.dmatrix('1 + ' + form, d, return_type='dataframe')
    for c in Xd.columns:
        if c != 'Intercept' and Xd[c].nunique() > 2: Xd[c] = (Xd[c] - Xd[c].mean()) / Xd[c].std(ddof=0)
    X = Xd.to_numpy(); dpd.TAU = tau
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        b0 = LogitQR(tau).fit(X, y, None).v
        m = {'UW': UnitWeibullQR, 'KW': KumaraswamyQR}[fam](tau).fit(X, y, b0)
        f = dpd.fit(X, y, fam, g, 'logit', b0, m.v)
        se0 = dpd.sandwich(m.v, X, y, fam, 0., 'logit'); seg = dpd.sandwich(f.x, X, y, fam, g, 'logit')
        lh = dpd.logpdf(f.x, X, y, fam) + np.log(y) + np.log1p(-y)
    w = np.exp(g * (lh - np.median(lh)))
    return d, y, X, list(Xd.columns), m.v, se0, f.x, seg, w, fam, g


def fig_water_and_tables():
    boot = pd.read_csv(ROOT / 'results_real/boot.csv')
    names = {'water': ['mhdi', 'log(incpc)', 'region (NE)', 'log(pop)'], 'HIC': ['URB', 'log(GDP)'], 'Firm': ['SIZELOG', 'INDCOST']}
    # Forest plot of bootstrap CIs for all three datasets and taus
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[2, 1, 1]))
    for ax, ds in zip(axes, ['water', 'HIC', 'Firm']):
        g = boot[boot.dataset == ds]; terms = names[ds]; ypos = 0; ticks = []; tl = []
        for j, t in enumerate(terms):
            for tau in (.1, .5, .9):
                gg = g[g.tau == tau]
                for est, off, col in (('mle', .15, COL['MLE']), ('dpd', -.15, COL['LMDPDE_auto'])):
                    v = np.array([json.loads(x)[j + 1] for x in gg[est]]); lo, md_, hi = np.percentile(v, [2.5, 50, 97.5])
                    ax.plot([lo, hi], [ypos + off] * 2, color=col, lw=1.4); ax.plot(md_, ypos + off, 'o', color=col, ms=2.5)
                ticks.append(ypos); tl.append(rf'{t}, $\tau$={tau}'); ypos -= 1
            ypos -= .6
        ax.axvline(0, color='k', lw=.6, ls=':'); ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=6.5)
        ax.set_title({'water': '(a) Piped water', 'HIC': '(b) Health insurance', 'Firm': '(c) Firm cost'}[ds])
        ax.set_xlabel('coefficient')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=COL['MLE']), Line2D([], [], color=COL['LMDPDE_auto'])], ['MLE', r'LMDPDE ($\hat\gamma$)'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(); save(fig, 'fig9_bootstrap_forest')

    # Weights plots (tau = 0.5)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    out = {}
    for k, ds in enumerate(['water', 'HIC', 'Firm']):
        d, y, X, cols, v0, se0, vg, seg, w, fam, g = fit_real(ds, .5); out[ds] = (d, y, cols, v0, se0, vg, seg, w, fam, g)
        q = expit(X @ vg[:-1]); flag = w < .1
        ax[k].scatter(q[~flag], w[~flag], s=4, color=COL['LMDPDE_auto'], alpha=.5)
        ax[k].scatter(q[flag], w[flag], s=12, color='#b8323a', marker='x', label='_nolegend_')
        ax[k].set_xlabel('fitted median'); ax[k].set_yscale('log'); ax[k].text(*((.04, .08) if ds == 'water' else (.98, .55)), f'{flag.sum()} flagged\n($w_i<0.1$)', transform=ax[k].transAxes, ha='left' if ds == 'water' else 'right', fontsize=7, color='#b8323a')
        ax[k].set_title({'water': f'(a) Piped water (UW, $\\gamma$={g})', 'HIC': f'(b) Health insurance (KW, $\\gamma$={g})',
                         'Firm': f'(c) Firm cost (KW, $\\gamma$={g})'}[ds])
        if ds == 'HIC':
            for i in np.flatnonzero(flag): ax[k].annotate(d.CITY.iloc[i].title(), (q[i], w[i]), fontsize=6, xytext=(3, 3), textcoords='offset points')
        if ds == 'Firm':
            for i in np.flatnonzero(flag): ax[k].annotate(f'firm {i + 1} (y={y[i]:.3f})', (q[i], w[i]), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax[0].set_ylabel('relative DPD weight')
    fig.tight_layout(); save(fig, 'fig8_weights')

    # Water: observed vs income coloured by weight
    d, y, cols, v0, se0, vg, seg, w, fam, g = out['water']
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    flag = w < .1
    for k, (reg, nm) in enumerate(((0, 'Southeast'), (1, 'Northeast'))):
        m = d.region.to_numpy() == reg
        ax[k].scatter(np.log(d.incpc[m & ~flag]), y[m & ~flag], s=3, color='#9db8d9', alpha=.5, label='municipalities')
        ax[k].scatter(np.log(d.incpc[m & flag]), y[m & flag], s=14, marker='x', color='#b8323a', label='flagged by LMDPDE')
        ax[k].set_xlabel('log per-capita income'); ax[k].set_title(f'{nm} ({m.sum()} municipalities)')
    ax[0].set_ylabel('proportion with piped water'); ax[0].legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout(); save(fig, 'fig10_water_scatter')

    # Coefficient tables (tau = 0.5) with bootstrap CIs
    lines = []
    for ds in ['water', 'HIC', 'Firm']:
        d, y, cols, v0, se0, vg, seg, w, fam, g = out[ds]; gb = boot[(boot.dataset == ds) & (boot.tau == .5)]
        Mb = np.array([json.loads(x) for x in gb.mle]); Db = np.array([json.loads(x) for x in gb.dpd])
        nm = {'water': 'Piped water (UW)', 'HIC': 'Health insurance (KW)', 'Firm': 'Firm cost (KW)'}[ds]
        lines.append(rf'\multicolumn{{7}}{{l}}{{\textit{{{nm}}}, $n={len(y)}$, $\hat\gamma={g}$}}\\')
        labels = ['Intercept'] + names[ds]
        for j, t in enumerate(labels):
            ml, mh = np.percentile(Mb[:, j], [2.5, 97.5]); dl, dh = np.percentile(Db[:, j], [2.5, 97.5])
            lines.append(f'\\quad {t} & {v0[j]:.3f} & ({se0[j]:.3f}) & [{ml:.3f}, {mh:.3f}] & {vg[j]:.3f} & ({seg[j]:.3f}) & [{dl:.3f}, {dh:.3f}]\\\\')
        lines.append(r'\addlinespace')
    (TAB / 'tab_realdata_coef.tex').write_text('\n'.join(lines[:-1]) + '\n'); print('table realdata coef')
    flagged = {ds: int((out[ds][7] < .1).sum()) for ds in out}
    json.dump(flagged, open(TAB / 'flagged_counts.json', 'w'))


def tables_sim(d):
    sub = d[d.n == 300]
    mse = sub.groupby(['tau', 'setting', 'scenario', 'estimator']).slope_squared_error.mean().unstack('estimator')
    cov = sub.assign(c=(sub.hit1.astype(float) + sub.hit2.astype(float)) / 2).groupby(['tau', 'setting', 'scenario', 'estimator']).c.mean().unstack('estimator')
    fail = 1 - sub[sub.estimator == 'MDPDE_0.3'].groupby(['tau', 'setting', 'scenario']).converged.mean()
    mc = sub.groupby(['tau', 'setting', 'scenario', 'estimator']).slope_squared_error.agg(lambda x: x.std() / np.sqrt(len(x)))
    rel = (mc / sub.groupby(['tau', 'setting', 'scenario', 'estimator']).slope_squared_error.mean()).unstack('estimator')[EST]
    print('relative MCSE slope MSE tau .5: median %.3f max %.3f' % (rel.loc[.5].stack().median(), rel.loc[.5].stack().max()))
    for t in (.1, .5, .9):
        lines = []
        for s in range(4):
            if (t, s) not in mse.index.droplevel(2): continue
            first = True
            for sc in SC:
                if (t, s, sc) not in mse.index: continue
                r = mse.loc[(t, s, sc)]; c = cov.loc[(t, s, sc)]
                failed = fail.loc[(t, s, sc)] > .5
                best = r[[e for e in EST if not (failed and e == 'MDPDE_0.3')]].idxmin()
                cells = []
                for e in EST:
                    txt = '--' if (failed and e == 'MDPDE_0.3') else f'{100 * r[e]:.2f}'
                    if e == best: txt = r'\textbf{' + txt + '}'
                    cells.append(txt)
                lines.append((SET[s] if first else '') + f' & {sc} & ' + ' & '.join(cells) +
                             f' & {c["MLE"]:.2f} & {c["LMDPDE_auto"]:.2f}\\\\'); first = False
            lines.append(r'\addlinespace')
        (TAB / f'tab_sim_tau{int(t * 10)}.tex').write_text('\n'.join(lines[:-1]) + '\n'); print('table sim', t)
    # n = 100 compact: ratio vs MLE and vs LogitQR for LMDPDE_auto
    sub = d[d.n == 100]; m1 = sub.groupby(['tau', 'setting', 'scenario', 'estimator']).slope_squared_error.mean().unstack('estimator')
    lines = []
    for (t, s), g in m1.groupby(level=[0, 1]):
        r1 = [f'{g.loc[(t, s, sc), "MLE"] / g.loc[(t, s, sc), "LMDPDE_auto"]:.2f}' for sc in SC]
        r2 = [f'{g.loc[(t, s, sc), "LogitQR"] / g.loc[(t, s, sc), "LMDPDE_auto"]:.2f}' for sc in SC]
        lines.append(f'{t} & {SET[s]} & ' + ' & '.join(r1) + ' & ' + ' & '.join(r2) + r'\\')
    (TAB / 'tab_sim_n100.tex').write_text('\n'.join(lines) + '\n'); print('table n100')


def table_cv():
    c = pd.read_csv(ROOT / 'results_real/cv.csv')
    rec = c.groupby(['dataset', 'tau', 'model', 'record']).loss.mean().reset_index()
    tab = rec.groupby(['dataset', 'tau', 'model']).loss.mean().unstack('model') * 1000
    order = ['UW-MLE', 'UW-LMDPDE_auto', 'KW-MLE', 'KW-LMDPDE_auto', 'LogitQR', 'Beta-LMDPDE_0.3']
    lines = []
    for (ds, t), r in tab.iterrows():
        best = r[order].idxmin()
        cells = [(r'\textbf{%.2f}' if m == best else '%.2f') % r[m] for m in order]
        lines.append(f'{ {"water": "Piped water", "HIC": "Health insurance", "Firm": "Firm cost"}[ds]} & {t} & ' + ' & '.join(cells) + r'\\')
    (TAB / 'tab_cv.tex').write_text('\n'.join(lines) + '\n'); print('table cv')


# ---------- Figure 11: MSE as a function of the tuning constant ----------
def fig_gamma_path(d):
    sub = d[(d.n == 300) & (d.tau == .5)]
    grid = [0, .1, .2, .3, .4, .5, .6, .7]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), sharex=True)
    for si in range(4):
        for sc, c, ls in (('C0', '#6b6b6b', '-'), ('C1', '#7fa7d6', '-'), ('C2', '#1f5fa8', '-'), ('C3', '#d9822b', '--')):
            s = sub[(sub.setting == si) & (sub.scenario == sc)]
            m = [s[s.estimator == ('MLE' if g == 0 else f'LMDPDE_{g}')].slope_squared_error.mean() for g in grid]
            ax[si].plot(grid, np.array(m) * 100, color=c, ls=ls, marker='o', ms=2.5, label=SC[sc])
            ma = s[s.estimator == 'LMDPDE_auto'].slope_squared_error.mean() * 100
            ax[si].axhline(ma, color=c, lw=.6, ls=':')
        ax[si].set_yscale('log'); ax[si].set_title(SET[si]); ax[si].set_xlabel(r'$\gamma$')
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        ax[si].yaxis.set_major_formatter(ScalarFormatter()); ax[si].yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel(r'slope MSE ($\times100$, log scale)')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.1))
    fig.tight_layout(); save(fig, 'fig11_gamma_path')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--paper':
    fig_nonexistence(); fig_designs(); d = load_sim(); fig_boxplots(d); fig_gamma(d); fig_gamma_path(d); tables_sim(d); sys.exit()

if __name__ == '__main__':
    fig_nonexistence(); fig_designs()
    d = load_sim(); print('sim rows', len(d))
    fig_boxplots(d); fig_heatmap(d); fig_coverage(d); fig_gamma(d); fig_mdpde_fail(d); fig_gamma_path(d); tables_sim(d)
    fig_water_and_tables(); table_cv()
