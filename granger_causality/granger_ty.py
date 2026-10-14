"""Bivariate Toda-Yamamoto Granger causality test for one song.

For a chart series (rank) and an activity series (x):
  1. Integration order of each series: the first difference order d (0, 1, 2)
     at which the ADF test rejects a unit root at .05.  dmax = max of the two.
  2. Both series are standardized; lag p is chosen by BIC among 1..max_lag on a
     common response sample (cap reduced only to keep residual df positive).
  3. A VAR(p + dmax) in levels with intercept is fitted, and the first p lags of
     the cause are tested with a modified Wald chi-square(p) test, both ways.
  4. Residual autocorrelation is checked with the adjusted Portmanteau test on
     both the VAR(p) and VAR(p + dmax); diagnostics pass if neither rejects.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

ALPHA = 0.05
MAX_INTEGRATION_ORDER = 2
TOL = 1e-12


def check_fit(fit):
    """Reject numerically degenerate VAR fits (constant residuals, singular design)."""
    effective = np.asarray(fit.endog, dtype=float)[fit.k_ar:]
    variance = effective.var(axis=0)
    if not np.isfinite(variance).all() or (variance <= TOL * np.asarray(fit.endog).var(axis=0)).any():
        raise ValueError("constant_or_negligible_effective_variation")
    singular = np.linalg.svd(fit.endog_lagged, compute_uv=False)
    if fit.df_resid <= 0 or len(singular) != fit.endog_lagged.shape[1] or singular[-1] <= TOL * singular[0]:
        raise ValueError("singular_or_ill_conditioned_design")
    covariance = np.asarray(fit.sigma_u, dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("nonfinite_residual_covariance")
    scale = np.sqrt(variance)
    normalized = covariance / np.outer(scale, scale)
    eig = np.linalg.eigvalsh((normalized + normalized.T) / 2)
    if eig[0] <= TOL or eig[0] <= TOL * eig[-1]:
        raise ValueError("degenerate_residual_covariance")


def integration_order(series):
    """Smallest d <= 2 at which ADF (constant, AIC lags) rejects; None if never."""
    series = np.asarray(series, dtype=float)
    for d in range(MAX_INTEGRATION_ORDER + 1):
        if len(series) < 4 or not np.isfinite(series).all() or np.std(series) == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = adfuller(series, regression="c", autolag="AIC")[1]
            if not np.isfinite(p):
                return None
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            return None
        if p < ALPHA:
            return d
        series = np.diff(series)
    return None


def select_lag(data, dmax, max_lag):
    """BIC-minimizing lag in 1..cap, all fitted on the same response dates."""
    cap = min(max_lag, (len(data) - 2) // 3 - dmax)
    if cap < 1:
        raise ValueError("no_lag_with_positive_residual_df")
    candidates = []
    for p in range(1, cap + 1):
        try:
            fit = VAR(data.iloc[cap - p:]).fit(p, trend="c")
            check_fit(fit)
            if np.isfinite(fit.bic):
                candidates.append((float(fit.bic), p))
        except (ValueError, np.linalg.LinAlgError):
            pass
    if not candidates:
        raise ValueError("no_estimable_bic_candidate")
    return min(candidates)[1]


def modified_wald(fit, cause, effect, p):
    """Wald test that the first p lags of `cause` are zero in the `effect` equation."""
    check_fit(fit)
    index = [(f"L{lag}.{cause}", effect) for lag in range(1, p + 1)]
    b = fit.params.stack().loc[index].to_numpy()
    v = fit.cov_params().loc[index, index].to_numpy()
    np.linalg.cholesky(v)
    d = np.sqrt(np.diag(v))
    eig = np.linalg.eigvalsh(v / np.outer(d, d))
    if eig[0] <= TOL * eig[-1]:
        raise ValueError("ill_conditioned_restriction_covariance")
    stat = float(b @ np.linalg.solve(v, b))
    return float(chi2.sf(stat, p))


def residuals_white(fit, horizon):
    """True / False for adjusted Portmanteau (not) rejecting; None if unavailable."""
    try:
        check_fit(fit)
        test = fit.test_whiteness(nlags=horizon, adjusted=True)
        p = float(test.pvalue)
        if not np.isfinite(p) or not 0 <= p <= 1 or test.test_statistic < 0:
            return None
        return p >= ALPHA
    except (ValueError, np.linalg.LinAlgError):
        return None


def toda_yamamoto(rank, x, max_lag=7):
    """p-values for x -> rank and rank -> x, plus whether the model is usable."""
    out = {"estimable": False, "diagnostics_pass": False, "exclusion_reason": "",
           "dmax": None, "lag": None, "p_x_to_rank": None, "p_rank_to_x": None}
    data = pd.DataFrame({"rank": rank, "x": x}, dtype=float)
    if not np.isfinite(data.to_numpy()).all() or (data.std(ddof=0) == 0).any():
        out["exclusion_reason"] = "constant_or_invalid_series"
        return out
    orders = [integration_order(data[c]) for c in data]
    if None in orders:
        out["exclusion_reason"] = "integration_order_unresolved"
        return out
    dmax = out["dmax"] = max(orders)
    data = (data - data.mean()) / data.std(ddof=0)
    try:
        p = out["lag"] = select_lag(data, dmax, max_lag)
        fit = VAR(data).fit(p + dmax, trend="c")
        check_fit(fit)
        white = []
        for f in (VAR(data).fit(p, trend="c"), fit):
            white.append(residuals_white(f, max(f.k_ar + 1, min(12, len(data) // 3))))
        out["diagnostics_pass"] = white == [True, True]
        # Both directions must be testable; otherwise neither p-value is kept.
        p_x_to_rank = modified_wald(fit, "x", "rank", p)
        p_rank_to_x = modified_wald(fit, "rank", "x", p)
        out.update(p_x_to_rank=p_x_to_rank, p_rank_to_x=p_rank_to_x, estimable=True)
    except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
        out["exclusion_reason"] = str(error)
        out["diagnostics_pass"] = False
    return out
