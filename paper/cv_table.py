"""Table 7: held-out pinball loss (x1000), long format. Bold: better of MLE/LMDPDE within a family; underline: smallest in the column."""
from pathlib import Path
import pandas as pd
P = Path(__file__).resolve().parent; ROOT = P.parent
c = pd.read_csv(ROOT / 'results_real/cv.csv'); c = c[c.dataset.isin(['HIC', 'water'])]
rec = c.groupby(['dataset', 'tau', 'model', 'record']).loss.mean().reset_index()
tab = (rec.groupby(['dataset', 'tau', 'model']).loss.mean() * 1000).unstack('tau')
ROWS = [('UW-MLE', 'Unit-Weibull MLE'), ('UW-LMDPDE_auto', 'Unit-Weibull LMDPDE'), ('KW-MLE', 'Kumaraswamy MLE'),
        ('KW-LMDPDE_auto', 'Kumaraswamy LMDPDE'), ('LogitQR', 'Logit-QR'), ('Beta-LMDPDE_0.3', 'Beta LMDPDE')]
L = []
for ds, title in (('HIC', 'Health insurance'), ('water', 'Piped water')):
    T = tab.loc[ds].round(2)
    for i, (m, lab) in enumerate(ROWS):
        cells = []
        for t in T.columns:
            v = T.loc[m, t]; s = f'{v:.2f}'
            fam = m.split('-')[0]
            if fam in ('UW', 'KW') and v == min(T.loc[f'{fam}-MLE', t], T.loc[f'{fam}-LMDPDE_auto', t]): s = r'\mathbf{' + s + '}'
            if v == T[t].min(): s = r'\underline{' + s + '}'
            cells.append(f'${s}$')
        L.append((title if i == 0 else '') + f' & {lab} & ' + ' & '.join(cells) + r'\\')
    L.append(r'\midrule')
(P / 'tables/app_cv.tex').write_text('\n'.join(L[:-1]) + '\n'); print('\n'.join(L))
