# Real data: bootstrap CIs and cross-validation

## Bootstrap percentile 95% CIs (case resampling; * = excludes 0)

| dataset | family | tau | gamma | term | MLE_95CI | LMDPDE_95CI | width_ratio | B | failed |
|---|---|---|---|---|---|---|---|---|---|
| Firm | KW | 0.1 | 0.1 | SIZELOG | [-0.945, 0.023] | [-0.985, -0.453] * | 0.55 | 300 | 0 |
| Firm | KW | 0.1 | 0.1 | INDCOST | [0.208, 1.986] * | [0.181, 0.719] * | 0.3 | 300 | 0 |
| Firm | KW | 0.5 | 0.1 | SIZELOG | [-0.986, -0.208] * | [-1.042, -0.580] * | 0.59 | 300 | 1 |
| Firm | KW | 0.5 | 0.1 | INDCOST | [0.279, 2.167] * | [0.209, 0.843] * | 0.34 | 300 | 1 |
| Firm | KW | 0.9 | 0.1 | SIZELOG | [-1.143, -0.294] * | [-1.247, -0.616] * | 0.74 | 300 | 0 |
| Firm | KW | 0.9 | 0.1 | INDCOST | [0.322, 2.416] * | [0.176, 2.136] * | 0.94 | 300 | 0 |
| HIC | KW | 0.1 | 0.4 | URB | [0.253, 0.613] * | [0.221, 0.738] * | 1.43 | 300 | 0 |
| HIC | KW | 0.1 | 0.4 | log(GDP) | [0.095, 0.486] * | [0.048, 0.540] * | 1.26 | 300 | 0 |
| HIC | KW | 0.5 | 0.4 | URB | [0.283, 0.754] * | [0.244, 0.880] * | 1.35 | 300 | 0 |
| HIC | KW | 0.5 | 0.4 | log(GDP) | [0.075, 0.535] * | [0.038, 0.611] * | 1.25 | 300 | 0 |
| HIC | KW | 0.9 | 0.4 | URB | [0.294, 0.963] * | [0.233, 0.836] * | 0.9 | 300 | 0 |
| HIC | KW | 0.9 | 0.4 | log(GDP) | [0.099, 0.680] * | [0.088, 0.751] * | 1.14 | 300 | 0 |
| water | UW | 0.1 | 0.5 | mhdi | [0.790, 1.107] * | [0.630, 0.887] * | 0.81 | 300 | 0 |
| water | UW | 0.1 | 0.5 | log(incpc) | [-0.047, 0.238] | [0.253, 0.511] * | 0.9 | 300 | 0 |
| water | UW | 0.1 | 0.5 | region (NE) | [-0.423, -0.136] * | [-0.192, 0.011] | 0.71 | 300 | 0 |
| water | UW | 0.1 | 0.5 | log(pop) | [0.057, 0.163] * | [0.059, 0.152] * | 0.88 | 300 | 0 |
| water | UW | 0.5 | 0.5 | mhdi | [0.695, 0.932] * | [0.560, 0.782] * | 0.94 | 300 | 0 |
| water | UW | 0.5 | 0.5 | log(incpc) | [0.012, 0.247] * | [0.255, 0.470] * | 0.91 | 300 | 0 |
| water | UW | 0.5 | 0.5 | region (NE) | [-0.350, -0.122] * | [-0.179, 0.000] | 0.79 | 300 | 0 |
| water | UW | 0.5 | 0.5 | log(pop) | [0.069, 0.170] * | [0.069, 0.145] * | 0.75 | 300 | 0 |
| water | UW | 0.9 | 0.5 | mhdi | [0.626, 0.881] * | [0.499, 0.710] * | 0.83 | 300 | 0 |
| water | UW | 0.9 | 0.5 | log(incpc) | [0.013, 0.257] * | [0.262, 0.481] * | 0.9 | 300 | 0 |
| water | UW | 0.9 | 0.5 | region (NE) | [-0.355, -0.116] * | [-0.181, -0.017] * | 0.69 | 300 | 0 |
| water | UW | 0.9 | 0.5 | log(pop) | [0.072, 0.165] * | [0.067, 0.147] * | 0.86 | 300 | 0 |

## Held-out pinball loss x1000 (2 x 5-fold CV; rank in brackets)

| dataset | tau | Beta-LMDPDE_0.3 | KW-LMDPDE_auto | KW-MLE | LogitQR | UW-LMDPDE_auto | UW-MLE |
|---|---|---|---|---|---|---|---|
| Firm | 0.1 | 10.408 (5) | 9.229 (1) | 19.89 (6) | 9.455 (3) | 9.925 (4) | 9.233 (2) |
| Firm | 0.5 | 33.848 (3) | 34.784 (5) | 58.934 (6) | 33.843 (2) | 34.033 (4) | 32.571 (1) |
| Firm | 0.9 | 32.406 (5) | 30.511 (2) | 37.569 (6) | 30.845 (4) | 29.853 (1) | 30.74 (3) |
| HIC | 0.1 | 17.213 (4) | 16.048 (2) | 17.773 (5) | 16.039 (1) | 18.431 (6) | 16.454 (3) |
| HIC | 0.5 | 43.545 (2) | 44.586 (5) | 45.155 (6) | 42.131 (1) | 44.456 (4) | 43.697 (3) |
| HIC | 0.9 | 28.921 (1) | 30.992 (4) | 36.313 (6) | 31.071 (5) | 30.054 (2) | 30.931 (3) |
| water | 0.1 | 23.72 (4) | 27.191 (5) | 28.136 (6) | 22.097 (1) | 22.183 (2) | 23.021 (3) |
| water | 0.5 | 39.292 (1) | 40.521 (5) | 40.942 (6) | 39.987 (4) | 39.866 (3) | 39.831 (2) |
| water | 0.9 | 13.418 (2) | 13.54 (4) | 13.539 (3) | 13.367 (1) | 13.646 (5) | 13.986 (6) |

## Paired differences in held-out pinball loss x1000 (negative favours LMDPDE)

| dataset | tau | comparison | mean | ci95 |
|---|---|---|---|---|
| Firm | 0.1 | UW-LMDPDE_auto - UW-MLE | 0.693 | [-0.604, 1.989] |
| Firm | 0.1 | UW-LMDPDE_auto - LogitQR | 0.47 | [-0.580, 1.521] |
| Firm | 0.1 | KW-LMDPDE_auto - KW-MLE | -10.661 | [-26.636, 5.314] |
| Firm | 0.1 | KW-LMDPDE_auto - LogitQR | -0.227 | [-1.097, 0.644] |
| Firm | 0.5 | UW-LMDPDE_auto - UW-MLE | 1.462 | [-0.683, 3.608] |
| Firm | 0.5 | UW-LMDPDE_auto - LogitQR | 0.19 | [-1.989, 2.370] |
| Firm | 0.5 | KW-LMDPDE_auto - KW-MLE | -24.15 | [-39.320, -8.980] |
| Firm | 0.5 | KW-LMDPDE_auto - LogitQR | 0.941 | [-1.712, 3.593] |
| Firm | 0.9 | UW-LMDPDE_auto - UW-MLE | -0.887 | [-3.876, 2.102] |
| Firm | 0.9 | UW-LMDPDE_auto - LogitQR | -0.993 | [-3.838, 1.852] |
| Firm | 0.9 | KW-LMDPDE_auto - KW-MLE | -7.057 | [-12.376, -1.738] |
| Firm | 0.9 | KW-LMDPDE_auto - LogitQR | -0.334 | [-5.460, 4.792] |
| HIC | 0.1 | UW-LMDPDE_auto - UW-MLE | 1.976 | [-0.520, 4.473] |
| HIC | 0.1 | UW-LMDPDE_auto - LogitQR | 2.392 | [-1.856, 6.639] |
| HIC | 0.1 | KW-LMDPDE_auto - KW-MLE | -1.724 | [-3.171, -0.278] |
| HIC | 0.1 | KW-LMDPDE_auto - LogitQR | 0.01 | [-0.795, 0.814] |
| HIC | 0.5 | UW-LMDPDE_auto - UW-MLE | 0.759 | [-1.474, 2.991] |
| HIC | 0.5 | UW-LMDPDE_auto - LogitQR | 2.326 | [-1.200, 5.851] |
| HIC | 0.5 | KW-LMDPDE_auto - KW-MLE | -0.569 | [-3.591, 2.453] |
| HIC | 0.5 | KW-LMDPDE_auto - LogitQR | 2.455 | [-0.240, 5.150] |
| HIC | 0.9 | UW-LMDPDE_auto - UW-MLE | -0.877 | [-2.765, 1.010] |
| HIC | 0.9 | UW-LMDPDE_auto - LogitQR | -1.017 | [-5.304, 3.269] |
| HIC | 0.9 | KW-LMDPDE_auto - KW-MLE | -5.322 | [-10.511, -0.132] |
| HIC | 0.9 | KW-LMDPDE_auto - LogitQR | -0.08 | [-2.912, 2.752] |
| water | 0.1 | UW-LMDPDE_auto - UW-MLE | -0.838 | [-1.197, -0.478] |
| water | 0.1 | UW-LMDPDE_auto - LogitQR | 0.086 | [-0.089, 0.262] |
| water | 0.1 | KW-LMDPDE_auto - KW-MLE | -0.945 | [-1.290, -0.601] |
| water | 0.1 | KW-LMDPDE_auto - LogitQR | 5.094 | [4.156, 6.032] |
| water | 0.5 | UW-LMDPDE_auto - UW-MLE | 0.034 | [-0.149, 0.218] |
| water | 0.5 | UW-LMDPDE_auto - LogitQR | -0.122 | [-0.312, 0.069] |
| water | 0.5 | KW-LMDPDE_auto - KW-MLE | -0.421 | [-0.583, -0.259] |
| water | 0.5 | KW-LMDPDE_auto - LogitQR | 0.534 | [-0.062, 1.129] |
| water | 0.9 | UW-LMDPDE_auto - UW-MLE | -0.339 | [-0.410, -0.268] |
| water | 0.9 | UW-LMDPDE_auto - LogitQR | 0.279 | [0.029, 0.529] |
| water | 0.9 | KW-LMDPDE_auto - KW-MLE | 0.001 | [-0.030, 0.033] |
| water | 0.9 | KW-LMDPDE_auto - LogitQR | 0.173 | [0.097, 0.248] |

## Mean gamma selected inside CV folds

| dataset | tau | gamma_uw | gamma_kw |
|---|---|---|---|
| Firm | 0.1 | 0.3 | 0.14 |
| Firm | 0.5 | 0.32 | 0.13 |
| Firm | 0.9 | 0.31 | 0.27 |
| HIC | 0.1 | 0.27 | 0.33 |
| HIC | 0.5 | 0.39 | 0.28 |
| HIC | 0.9 | 0.32 | 0.14 |
| water | 0.1 | 0.48 | 0.37 |
| water | 0.5 | 0.49 | 0.25 |
| water | 0.9 | 0.49 | 0.11 |

