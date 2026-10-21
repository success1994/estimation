# Robust parametric quantile regression for bounded responses

Code, data and simulation results for the paper

> Sumit Kumar and Rishi Kumar Jha. *Robust parametric quantile regression for bounded responses via a logit-scale density power divergence.* Department of Mathematics and Computing Technology, National Institute of Technology Patna. (Manuscript submitted for publication.)

The paper shows that the original-scale minimum density power divergence estimator (MDPDE) is undefined over large parts of the parameter space of the Kumaraswamy and unit-Weibull quantile regression models. It proposes the logit-scale MDPDE (LMDPDE), whose objective is finite for every parameter value, together with a stable quadrature scheme, complex-step gradients and a data-driven choice of the tuning constant.

## Repository layout

| Folder | Contents |
|---|---|
| `code/` | Estimators and the median-regression simulation (`dpd.py`, `models.py`, `simulate_full.py`), the rectangular-mixture competitor (`sim_rect.py`) and numerical checks of the quadrature and gradients (`validate.py`) |
| `modal_run/` | Version of the estimators with a configurable quantile level (`dpd.TAU`), the simulation at $\tau=0.1, 0.9$ (`sim_tau.py`) and the bootstrap / cross-validation for the applications (`realdata_modal.py`). These two scripts run on [Modal](https://modal.com) for parallelism; the estimator modules run locally |
| `data_real/csv/` | Application data: `HIC.csv` (health insurance coverage, from the R package `robustbetareg`) and `water.csv` (piped water, from the R package `unitquantreg`). `Firm.csv` is used only by the cross-validation script |
| `results_full/`, `results_tau/` | Raw simulation output, one row per fitted estimator and data set (gzip-compressed) |
| `results_real/` | Bootstrap and cross-validation output for the applications |
| `results_rect/` | Rectangular-mixture competitor, $\tau=0.5$, $n=300$ |
| `results/` | Output of the numerical checks in `code/validate.py` |
| `paper/` | Scripts that produce every table and figure of the paper, with their outputs in `paper/tables/` and `paper/figures/` |

## Setup

Python 3.12 or later.

```bash
python3 -m pip install -r requirements.txt
```

Decompress the raw simulation output before running the table and figure scripts:

```bash
gunzip -k results_full/sim_fits.csv.gz results_tau/sim_tau.csv.gz
```

## Reproducing the results

Set `OPENBLAS_NUM_THREADS=1` for all runs. The simulations are resumable and use fixed seeds.

1. Numerical checks of the quadrature and the complex-step gradients:
   ```bash
   python3 code/validate.py
   ```
2. Simulation at $\tau=0.5$ (writes `results_full/sim_fits.csv`):
   ```bash
   python3 code/simulate_full.py --reps300 300 --reps100 200 --workers 8
   ```
3. Simulation at $\tau=0.1$ and $0.9$ (writes `results_tau/sim_tau.csv`; requires a Modal account):
   ```bash
   modal run modal_run/sim_tau.py
   ```
4. Bootstrap and cross-validation for the applications (writes `results_real/`):
   ```bash
   modal run modal_run/realdata_modal.py
   ```
5. Rectangular-mixture competitor (writes `results_rect/sim_rect.csv`):
   ```bash
   python3 code/sim_rect.py
   ```
6. Tables and figures (run from `paper/`):
   ```bash
   cd paper
   python3 standard_tables.py      # Tables 1-5, A1-A6, A8 and the win counts quoted in the text
   python3 scoreboard.py           # paired bootstrap comparisons quoted in Section 4.2
   python3 make_figures.py --paper # Figures 1-5 and Table A7
   python3 envelopes.py            # simulated residual envelopes (piped-water data)
   python3 apps.py                 # application estimates, descriptive statistics, envelopes
   python3 apps_perdataset.py      # health insurance coefficient plot
   python3 app_revision_figs.py    # application diagnostics and bootstrap difference intervals
   python3 cv_table.py             # Table 8 (cross-validation)
   ```

Steps 3 and 4 were run on Modal. The completed output is included, so steps 5 and 6 can be run directly after decompressing the files above.

## Estimators

`dpd.fit(X, y, family, gamma, scale, b0)` fits the MDPDE for `family` in `{'UW', 'KW', 'Beta'}`; `scale='logit'` gives the LMDPDE and `scale='original'` the original-scale MDPDE. `dpd.sandwich` returns sandwich standard errors. `models.py` contains the maximum likelihood fits (`UnitWeibullQR`, `KumaraswamyQR`, `BetaReg`) and linear quantile regression on the logit scale (`LogitQR`). The data-driven tuning constant is selected by `simulate_full.select_gamma` (iterated Warwick–Jones procedure).

## Contact

Sumit Kumar, sumitk.ph.ma23@nitp.ac.in
