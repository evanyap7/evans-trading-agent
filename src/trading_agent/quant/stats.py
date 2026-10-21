"""Performance statistics that correct for luck.

A raw Sharpe ratio from a backtest is biased upward by every variant you tried and by fat tails. The
probabilistic and deflated Sharpe ratios (Bailey & Lopez de Prado, 2012/2014) put a probability on
"the true Sharpe is above the benchmark" after accounting for sample length, skew, kurtosis and the
number of trials behind the winner.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe(returns, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def max_drawdown(returns) -> float:
    """Worst peak-to-trough fall of the compounded curve, as a negative fraction."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1).min()) if len(curve) else 0.0


def cagr(returns, periods: int = 252) -> float:
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    if not len(r):
        return 0.0
    total = float(np.prod(1 + r))
    return total ** (periods / len(r)) - 1 if total > 0 else -1.0


def _moments(r: np.ndarray) -> tuple[float, float]:
    m, s = r.mean(), r.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (r - m) / s
    return float((z ** 3).mean()), float((z ** 4).mean())


def probabilistic_sharpe(returns, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr). `benchmark_sr` is per period, not annualised."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = _moments(r)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return 0.5
    return _N.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Per-period Sharpe you would expect from the best of `n_trials` skill-free strategies."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    sd = math.sqrt(sr_variance)
    return sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                 + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns, n_trials: int, sr_variance: float | None = None) -> float:
    """PSR against the best-of-`n_trials` luck hurdle. Below ~0.95 the edge is not distinguishable from selection."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return 0.5
    var = sr_variance if sr_variance is not None else 1.0 / len(r)
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var))


def block_bootstrap_sharpe(returns, *, n_boot: int = 2000, block: int = 10, seed: int = 0,
                           periods: int = 252) -> tuple[float, float]:
    """95% interval for the annualised Sharpe, resampling blocks to keep autocorrelation."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    n = len(r)
    if n < 2 * block:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    sample = r[idx]
    sd = sample.std(axis=1, ddof=1)
    srs = np.where(sd > 0, sample.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(periods), 0.0)
    lo, hi = np.percentile(srs, [2.5, 97.5])
    return float(lo), float(hi)
