"""Copied candidate-1 model implementations; QR retains default standard errors."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, gammainc, expit, logit
from scipy.stats import beta as beta_dist


def negative_loglik(lp):
    """Keep invalid trial steps from contaminating L-BFGS line searches."""
    value = -np.mean(lp)
    return float(value) if np.isfinite(value) else 1e100


class KumaraswamyQR:
    def __init__(self, tau): self.tau = tau
    def _lp(self, v, X, y):
        p = X.shape[1]; a = np.exp(v[p]); q = expit(np.clip(X@v[:p], -30, 30))
        def log1mexp(t):
            # Both branches remain well conditioned in their own half-domain.
            return np.where(t < -np.log(2.),
                            np.log1p(-np.exp(np.minimum(t, -np.log(2.)))),
                            np.log(-np.expm1(np.maximum(t, -np.log(2.)))))
        def logneglog1mexp(t):
            return np.where(t < -36., t, np.log(-log1mexp(np.maximum(t, -36.))))
        logb = np.log(-np.log1p(-self.tau))-logneglog1mexp(a*np.log(q))
        ty = a*np.log(y)
        # Avoid the inf*0 cancellation in b*log(1-y**a).
        return (np.log(a)+logb+(a-1)*np.log(y)
                -np.exp(logb+logneglog1mexp(ty))-log1mexp(ty))
    def fit(self, X, y, b0):
        best = None
        for la in (-1., 0., 1.):
            with np.errstate(all='ignore'):
                r = minimize(lambda v: negative_loglik(self._lp(v, X, y)), np.r_[b0, la], method='L-BFGS-B',
                             bounds=[(-30, 30)]*X.shape[1]+[(-6, 6)], options={'maxiter': 600, 'maxls': 100, 'ftol': 1e-10})
            if np.isfinite(r.fun) and r.fun < 1e100 and (best is None or (r.success and not best.success) or (r.success == best.success and r.fun < best.fun)): best = r
        if best is None:
            raise RuntimeError("No finite optimizer result")
        self.success = bool(best.success)
        self.message = str(best.message)
        self.v = best.x; self.loglik = -best.fun*len(y); self.k = len(best.x); return self
    def quantile(self, X): return expit(np.clip(X@self.v[:X.shape[1]], -30, 30))
    def logpdf(self, X, y): return self._lp(self.v, X, y)


class UnitWeibullQR:
    def __init__(self, tau): self.tau = tau
    def _lp(self, v, X, y):
        p = X.shape[1]; k = np.exp(v[p]); q = expit(np.clip(X@v[:p], -30, 30))
        ly = -np.log(y)
        logal = np.log(-np.log(self.tau))-k*np.log(-np.log(q))
        return logal+np.log(k)-np.log(y)+(k-1)*np.log(ly)-np.exp(logal+k*np.log(ly))
    def fit(self, X, y, b0):
        best = None
        for lk in (-1., 0., 1.):
            with np.errstate(all='ignore'):
                r = minimize(lambda v: negative_loglik(self._lp(v, X, y)), np.r_[b0, lk], method='L-BFGS-B',
                             bounds=[(-30, 30)]*X.shape[1]+[(-6, 6)], options={'maxiter': 600, 'maxls': 100, 'ftol': 1e-10})
            if np.isfinite(r.fun) and r.fun < 1e100 and (best is None or (r.success and not best.success) or (r.success == best.success and r.fun < best.fun)): best = r
        if best is None:
            raise RuntimeError("No finite optimizer result")
        self.success = bool(best.success)
        self.message = str(best.message)
        self.v = best.x; self.loglik = -best.fun*len(y); self.k = len(best.x); return self
    def quantile(self, X): return expit(np.clip(X@self.v[:X.shape[1]], -30, 30))
    def logpdf(self, X, y): return self._lp(self.v, X, y)


class BetaReg:
    """Mean-parameterised beta regression; tau-quantiles from the fitted law."""
    def __init__(self, tau): self.tau = tau
    def _ab(self, v, X):
        p = X.shape[1]; m = expit(np.clip(X@v[:p], -30, 30)); phi = np.exp(v[p]); return m*phi, (1-m)*phi
    def fit(self, X, y, b0):
        ols = np.linalg.lstsq(X, logit(y), rcond=None)[0]
        f = lambda v: negative_loglik(beta_dist.logpdf(y, *self._ab(v, X)))
        best = None
        for lp in (0., 2., 4.):
            with np.errstate(all='ignore'):
                r = minimize(f, np.r_[ols, lp], method='L-BFGS-B', bounds=[(-30, 30)]*X.shape[1]+[(-6, 12)], options={'maxiter': 600, 'maxls': 100, 'ftol': 1e-10})
            if np.isfinite(r.fun) and r.fun < 1e100 and (best is None or (r.success and not best.success) or (r.success == best.success and r.fun < best.fun)): best = r
        if best is None:
            raise RuntimeError("No finite optimizer result")
        self.success = bool(best.success)
        self.message = str(best.message)
        self.v = best.x; self.loglik = -best.fun*len(y); self.k = len(best.x); return self
    def quantile(self, X): return beta_dist.ppf(self.tau, *self._ab(self.v, X))
    def logpdf(self, X, y): return beta_dist.logpdf(y, *self._ab(self.v, X))


class LogitQR:
    """Distribution-free linear quantile regression on logit(y)."""
    def __init__(self, tau): self.tau = tau
    def fit(self, X, y, b0):
        from statsmodels.regression.quantile_regression import QuantReg
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = QuantReg(logit(y), X).fit(q=self.tau, max_iter=5000)
            self.v = result.params
            self.se = result.bse
        self.success = result.iterations < 5000 and not caught
        self.message = '; '.join(str(w.message) for w in caught)
        self.loglik = np.nan; self.k = len(self.v); return self
    def quantile(self, X): return expit(X@self.v)
    def logpdf(self, X, y): return np.full(len(y), np.nan)

def start(X, y, tau):
    """Quantile-regression start on the logit scale, shared by every parametric model."""
    try: return LogitQR(tau).fit(X, y, None).v
    except Exception: return np.linalg.lstsq(X, logit(y), rcond=None)[0]


def pinball(y, q, tau):
    u = y-q; return u*(tau-(u < 0))
