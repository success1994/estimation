"""Conventional simulation tables: RMSE and bias per coefficient, and empirical levels of Wald tests."""
from pathlib import Path
import numpy as np, pandas as pd

P = Path(__file__).resolve().parent; ROOT = P.parent; TAB = P / 'tables'
B1, B2 = .8, -.6
SET = {0: 'UW, $k=0.6$', 1: 'UW, $k=1.5$', 2: 'KW, $a=0.6$', 3: 'KW, $a=2$'}
SC = {'C0': 'C0', 'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'C5': 'C5'}
EST = ['MLE', 'MDPDE_0.3', 'LMDPDE_auto', 'LogitQR']

cols = ['setting', 'n', 'scenario', 'rep', 'estimator', 'beta1', 'beta2', 'hit1', 'hit2', 'converged', 'boundary']
a = pd.read_csv(ROOT / 'results_full/sim_fits.csv', usecols=cols); a['tau'] = .5
b = pd.read_csv(ROOT / 'results_tau/sim_tau.csv', usecols=cols + ['tau'])
d = pd.concat([a, b], ignore_index=True); d = d[d.estimator.isin(EST + ['LMDPDE_0.3'])]
g = d.groupby(['tau', 'n', 'setting', 'scenario', 'estimator'])
S = g.agg(bias1=('beta1', lambda x: x.mean() - B1), bias2=('beta2', lambda x: x.mean() - B2),
          rmse1=('beta1', lambda x: np.sqrt(np.mean((x - B1) ** 2))), rmse2=('beta2', lambda x: np.sqrt(np.mean((x - B2) ** 2))),
          lev1=('hit1', lambda x: 100 * (1 - x.mean())), lev2=('hit2', lambda x: 100 * (1 - x.mean())),
          fail=('converged', lambda x: 1 - x.mean()), bnd=('boundary', 'mean'),
          mc1=('beta1', lambda x: np.std((x - B1) ** 2, ddof=1) / (2 * np.sqrt(np.mean((x - B1) ** 2)) * np.sqrt(len(x)))),
          mc2=('beta2', lambda x: np.std((x - B2) ** 2, ddof=1) / (2 * np.sqrt(np.mean((x - B2) ** 2)) * np.sqrt(len(x))))).reset_index()
S.to_csv(TAB / 'standard_summary.csv', index=False)


def failed(r): return r is not None and r.estimator == 'MDPDE_0.3' and r.fail > .5


def boundary(r): return r is not None and r.estimator == 'MDPDE_0.3' and not failed(r) and r.bnd > .5


def cell(r, key, fmt):
    if r is None: return ''
    if failed(r): return '--'
    return fmt % r[key] + ('$^\\dagger$' if boundary(r) else '')


WINS = []


def table(tau, n, keys, fmt, fname, bold=True):
    lines = []
    for si in range(4):
        sub = S[(S.tau == tau) & (S.n == n) & (S.setting == si)]
        if sub.empty: continue
        first = True
        for sc in SC:
            row = []
            for k in keys:
                vals = []
                for e in EST:
                    r = sub[(sub.scenario == sc) & (sub.estimator == e)]
                    r = r.iloc[0] if len(r) else None
                    vals.append((cell(r, k, fmt), None if r is None or failed(r) or boundary(r) else abs(r[k])))
                ok = [round(v, 3) for _, v in vals if v is not None]
                m = min(ok) if ok else None
                if bold and keys[0] == 'rmse1' and k == 'rmse1' and m is not None:
                    WINS.append((tau, n, si, sc, round(vals[2][1], 3) == m if vals[2][1] is not None else False))
                for txt, v in vals:
                    row.append(r'\textbf{' + txt + '}' if bold and v is not None and round(v, 3) == m else txt)
            lines.append((f'\\multirow{{6}}{{*}}{{{SET[si]}}}' if first else '') + f' & {SC[sc]} & ' + ' & '.join(row) + r'\\'); first = False
        lines.append(r'\midrule')
    (TAB / fname).write_text('\n'.join(lines[:-1]) + '\n'); print(fname)


for t in (.1, .5, .9):
    table(t, 300, ['rmse1', 'rmse2'], '%.3f', f'std_rmse_tau{int(t * 10)}.tex')
    table(t, 300, ['bias1', 'bias2'], '%.3f', f'std_bias_tau{int(t * 10)}.tex', bold=False)
    table(t, 100, ['rmse1', 'rmse2'], '%.3f', f'std_rmse_n100_tau{int(t * 10)}.tex')

# Empirical level (%) of 5% Wald tests of H0: beta_j = true value, averaged over response distributions, n = 300
lines = []
for t in (.1, .5, .9):
    first = True
    for sc in SC:
        sub = S[(S.tau == t) & (S.n == 300) & (S.scenario == sc)]
        row = []
        for k in ('lev1', 'lev2'):
            vals = [sub[sub.estimator == e][k].mean() for e in ('MLE', 'LMDPDE_auto', 'LogitQR')]
            dist = [round(abs(round(v, 1) - 5), 1) for v in vals]; bd = min(dist)
            row += [(r'\textbf{%.1f}' if dist[i] == bd else '%.1f') % v for i, v in enumerate(vals)]
        lines.append((f'\\multirow{{6}}{{*}}{{{t}}}' if first else '') + f' & {sc} & ' + ' & '.join(row) + r'\\'); first = False
    lines.append(r'\midrule')
(TAB / 'std_wald.tex').write_text('\n'.join(lines[:-1]) + '\n'); print('std_wald.tex')

# Numbers used in the text
L = S[(S.n == 300) & (S.estimator == 'LMDPDE_auto')].set_index(['tau', 'setting', 'scenario'])
M = S[(S.n == 300) & (S.estimator == 'MLE')].set_index(['tau', 'setting', 'scenario'])
Q = S[(S.n == 300) & (S.estimator == 'LogitQR')].set_index(['tau', 'setting', 'scenario'])
print('RMSE1 ratio MLE/LMDPDE clean range', (M.rmse1 / L.rmse1).xs('C0', level=2).round(2).agg(['min', 'max']).tolist())
print('Wald levels C0 (MLE,LMDPDE,LQR):', [round(S[(S.n == 300) & (S.scenario == 'C0') & (S.estimator == e)][['lev1', 'lev2']].values.mean(), 1) for e in ('MLE', 'LMDPDE_auto', 'LogitQR')])
for sc in ('C1', 'C2', 'C3'):
    print('Wald levels', sc, [round(S[(S.n == 300) & (S.scenario == sc) & (S.estimator == e)][['lev1', 'lev2']].values.mean(), 1) for e in ('MLE', 'LMDPDE_auto', 'LogitQR')])

# Win counts for beta_1 (LMDPDE with data-driven gamma attains or ties the smallest RMSE)
W = pd.DataFrame(WINS, columns=['tau', 'n', 'setting', 'scenario', 'win']); W = W[W.n == 300]
print('wins all', W.win.sum(), len(W), 'contaminated', W[W.scenario != 'C0'].win.sum(), (W.scenario != 'C0').sum())
for t, g in W.groupby('tau'): print(' tau', t, 'all', g.win.sum(), len(g), 'contam', g[g.scenario != 'C0'].win.sum(), (g.scenario != 'C0').sum())
X = S[S.estimator.isin(['MLE', 'LMDPDE_auto', 'LogitQR', 'MDPDE_0.3']) & (S.fail <= .5)]
rel = pd.concat([X.mc1 / X.rmse1, X.mc2 / X.rmse2])
print('relative MCSE of RMSE: median %.3f, 95%% %.3f, max %.3f' % (rel.median(), rel.quantile(.95), rel.max()))

# Original-scale MDPDE: failure and boundary rates, n = 300, pooled over scenarios
m = S[(S.estimator == 'MDPDE_0.3') & (S.n == 300)].groupby(['tau', 'setting'])[['fail', 'bnd']].mean()
lines = []
for (t, si), r in m.iterrows():
    lines.append(f'{t} & {SET[si]} & {100 * r.fail:.1f} & {100 * r.bnd:.1f}\\\\')
(TAB / 'mdpde_fail.tex').write_text('\n'.join(lines) + '\n'); print('mdpde_fail.tex')

# Rectangular-mixture ML (model-based robust competitor), tau = 0.5, n = 300
R = pd.read_csv(ROOT / 'results_rect/sim_rect.csv')
R = R.groupby(['setting', 'scenario']).agg(r1=('beta1', lambda x: np.sqrt(np.mean((x - B1) ** 2))), r2=('beta2', lambda x: np.sqrt(np.mean((x - B2) ** 2))))
lines = []; nw = 0
for si in range(4):
    for i, sc in enumerate(SC):
        row = []
        for k, rk in (('rmse1', 'r1'), ('rmse2', 'r2')):
            v = {e: S[(S.tau == .5) & (S.n == 300) & (S.setting == si) & (S.scenario == sc) & (S.estimator == e)][k].iloc[0] for e in ('MLE', 'LMDPDE_auto')}
            v['Rect'] = R.loc[(si, sc), rk]; order = ['MLE', 'Rect', 'LMDPDE_auto']; mn = min(round(v[e], 3) for e in order)
            if k == 'rmse1' and round(v['LMDPDE_auto'], 3) <= round(v['Rect'], 3): nw += 1
            row += [(r'\textbf{%.3f}' if round(v[e], 3) == mn else '%.3f') % v[e] for e in order]
        lines.append((f'\\multirow{{6}}{{*}}{{{SET[si]}}}' if i == 0 else '') + f' & {sc} & ' + ' & '.join(row) + r'\\')
    lines.append(r'\midrule')
(TAB / 'rect.tex').write_text('\n'.join(lines[:-1]) + '\n'); print('rect.tex; LMDPDE <= Rect for beta1 in', nw, 'of 24')
