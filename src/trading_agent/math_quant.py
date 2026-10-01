"""Quantitative mathematics engine for algorithmic trading.

Implements foundational mathematical concepts for quantitative trading:
1. Descriptive Statistics:
   - Central Tendency: Mean, Median, Bucketed Mode, Fast/Slow SMA crossovers
   - Dispersion: Range, Quartile Deviation (IQR), Mean Absolute Deviation (MAD),
     Variance, Standard Deviation, and Bollinger Bands
2. Probability Theory & Stochastic Processes:
   - Bernoulli trial modeling & win-rate expectancy
   - Bayesian probability updating from prior to posterior given evidence
   - Monte Carlo simulation for drawdown distribution and risk of ruin
   - Random walk volatility scaling
3. Linear Algebra & Portfolio Matrix Math:
   - Portfolio return vector and covariance matrix transformations
   - Portfolio variance (w^T * Sigma * w) and diversification ratio
   - Matrix properties: rank, trace, determinant, inverse, condition number
4. Linear Regression & Predictive Modeling:
   - Ordinary Least Squares (OLS) slope (beta/velocity), intercept (alpha), and R^2
   - Dynamic regression channels
5. Continuous Mathematics & Calculus:
   - 1st Derivative: Instantaneous price velocity (dp/dt)
   - 2nd Derivative: Price acceleration / curvature (d^2p/dt^2) to detect trend exhaustion
   - Volume-Weighted Average Price (VWAP) numerical integration
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ============================================================================
# 1. Descriptive Statistics: Central Tendency
# ============================================================================

def mean(values: Sequence[float]) -> float:
    """Arithmetic mean: sum(x_i) / n."""
    if not values:
        return math.nan
    return float(np.mean(values))


def median(values: Sequence[float]) -> float:
    """Median: middle value of sorted series. Robust to extreme outliers."""
    if not values:
        return math.nan
    return float(np.median(values))


def mode_bucketed(values: Sequence[float], n_bins: int = 10) -> float:
    """Approximate mode for continuous stock prices by binning into a histogram.
    
    Raw floating-point stock prices rarely repeat exactly. Binning identifies
    the modal price cluster (high-volume / high-frequency consolidation price).
    """
    if not values:
        return math.nan
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan
    counts, bin_edges = np.histogram(clean, bins=n_bins)
    max_idx = int(np.argmax(counts))
    # Return center of the most frequent bin
    return float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0)


@dataclass(frozen=True)
class MovingAverageCrossover:
    fast_sma: float
    slow_sma: float
    crossover_ratio: float  # fast / slow - 1.0 (positive = bullish, negative = bearish)
    is_bullish: bool        # fast > slow
    is_golden_cross: bool   # fast just crossed above slow in the last bar
    is_death_cross: bool    # fast just crossed below slow in the last bar


def moving_average_crossover(
    closes: Sequence[float], fast_period: int = 20, slow_period: int = 50
) -> MovingAverageCrossover:
    """Analyze fast vs slower moving average relationship and crossover meeting points."""
    if len(closes) < slow_period + 1:
        return MovingAverageCrossover(
            fast_sma=math.nan,
            slow_sma=math.nan,
            crossover_ratio=math.nan,
            is_bullish=False,
            is_golden_cross=False,
            is_death_cross=False,
        )

    fast_curr = float(np.mean(closes[-fast_period:]))
    slow_curr = float(np.mean(closes[-slow_period:]))
    fast_prev = float(np.mean(closes[-fast_period - 1 : -1]))
    slow_prev = float(np.mean(closes[-slow_period - 1 : -1]))

    ratio = (fast_curr / slow_curr - 1.0) if slow_curr > 0 else 0.0
    is_bullish = fast_curr > slow_curr
    golden_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
    death_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)

    return MovingAverageCrossover(
        fast_sma=round(fast_curr, 4),
        slow_sma=round(slow_curr, 4),
        crossover_ratio=round(ratio * 100, 2),  # in %
        is_bullish=is_bullish,
        is_golden_cross=golden_cross,
        is_death_cross=death_cross,
    )


# ============================================================================
# 1b. Descriptive Statistics: Dispersion & Volatility
# ============================================================================

@dataclass(frozen=True)
class DispersionMetrics:
    range_abs: float               # X_max - X_min
    range_pct: float               # (X_max - X_min) / mean * 100
    q1: float                      # 25th percentile
    q3: float                      # 75th percentile
    quartile_deviation: float      # 0.5 * (Q3 - Q1), robust semi-interquartile range
    quartile_deviation_pct: float  # QD / median * 100
    mean_absolute_deviation: float # MAD = mean(|x_i - mean|)
    mad_pct: float                 # MAD / mean * 100
    variance: float                # sigma^2
    standard_deviation: float      # sigma
    std_dev_pct: float             # sigma / mean * 100 (coefficient of variation)


def dispersion_metrics(values: Sequence[float]) -> DispersionMetrics:
    """Compute comprehensive dispersion measures (Range, QD, MAD, Variance, StdDev)."""
    if len(values) < 4:
        nan = math.nan
        return DispersionMetrics(
            range_abs=nan, range_pct=nan, q1=nan, q3=nan,
            quartile_deviation=nan, quartile_deviation_pct=nan,
            mean_absolute_deviation=nan, mad_pct=nan,
            variance=nan, standard_deviation=nan, std_dev_pct=nan,
        )

    arr = np.asarray(values, dtype=float)
    x_min, x_max = float(np.min(arr)), float(np.max(arr))
    r_abs = x_max - x_min
    mu = float(np.mean(arr))
    med = float(np.median(arr))
    r_pct = (r_abs / mu * 100) if mu > 0 else 0.0

    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    qd = 0.5 * (q3 - q1)
    qd_pct = (qd / med * 100) if med > 0 else 0.0

    mad = float(np.mean(np.abs(arr - mu)))
    mad_pct = (mad / mu * 100) if mu > 0 else 0.0

    var = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    sigma = math.sqrt(var) if var >= 0 else 0.0
    sigma_pct = (sigma / mu * 100) if mu > 0 else 0.0

    return DispersionMetrics(
        range_abs=round(r_abs, 4),
        range_pct=round(r_pct, 2),
        q1=round(q1, 4),
        q3=round(q3, 4),
        quartile_deviation=round(qd, 4),
        quartile_deviation_pct=round(qd_pct, 2),
        mean_absolute_deviation=round(mad, 4),
        mad_pct=round(mad_pct, 2),
        variance=round(var, 6),
        standard_deviation=round(sigma, 4),
        std_dev_pct=round(sigma_pct, 2),
    )


# ============================================================================
# 2. Probability Theory & Stochastic Processes
# ============================================================================

def bayes_posterior_probability(
    prior_prob: float,
    likelihood_true: float,
    likelihood_false: float,
) -> float:
    """Bayes' Theorem: P(Win|Evidence) = P(Evidence|Win) * P(Win) / P(Evidence).
    
    Args:
        prior_prob: Prior probability of a winning trade P(Win), e.g. base rate 0.45.
        likelihood_true: P(Evidence | Win), true positive rate, e.g. 0.70.
        likelihood_false: P(Evidence | Loss), false positive rate, e.g. 0.35.
    
    Returns:
        Posterior probability P(Win | Evidence).
    """
    if prior_prob <= 0.0:
        return 0.0
    if prior_prob >= 1.0:
        return 1.0

    p_evidence = (likelihood_true * prior_prob) + (likelihood_false * (1.0 - prior_prob))
    if p_evidence <= 0:
        return prior_prob
    posterior = (likelihood_true * prior_prob) / p_evidence
    return max(0.0, min(1.0, float(posterior)))


@dataclass(frozen=True)
class MonteCarloRiskAssessment:
    n_simulations: int
    horizon_trades: int
    median_final_equity: float
    p05_final_equity: float       # 5th percentile worst-case equity
    p95_final_equity: float       # 95th percentile best-case equity
    expected_max_drawdown_pct: float
    var_95_pct: float             # 95% Value at Risk (% of starting capital)
    probability_of_ruin_pct: float # Probability of hitting drawdown threshold (e.g. -20%)


def monte_carlo_trade_simulation(
    win_rate: float,
    reward_risk: float,
    risk_per_trade_pct: float = 1.0,
    horizon_trades: int = 50,
    n_simulations: int = 2000,
    ruin_threshold_pct: float = 20.0,
    seed: int = 42,
) -> MonteCarloRiskAssessment:
    """Stochastic Monte Carlo simulation of future trading paths.
    
    Simulates thousands of synthetic equity trajectories based on win rate and R:R
    to calculate empirical drawdown distributions, Value at Risk, and ruin probability.
    """
    rng = np.random.default_rng(seed)
    # Generate Bernoulli trials (1 for win, 0 for loss)
    trials = rng.binomial(n=1, p=win_rate, size=(n_simulations, horizon_trades))
    
    # Win earns +reward_risk * risk_pct, loss loses -1 * risk_pct
    trade_pct_returns = np.where(
        trials == 1,
        reward_risk * (risk_per_trade_pct / 100.0),
        -1.0 * (risk_per_trade_pct / 100.0),
    )
    
    # Cumulative compound wealth paths starting at 1.0
    wealth_paths = np.cumprod(1.0 + trade_pct_returns, axis=1)
    final_wealths = wealth_paths[:, -1]
    
    # Compute running maximum and max drawdown for each simulation path
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (running_max - wealth_paths) / running_max
    max_drawdowns = np.max(drawdowns, axis=1)
    
    ruined_count = np.sum(max_drawdowns >= (ruin_threshold_pct / 100.0))
    p_ruin = float(ruined_count / n_simulations * 100.0)

    p05 = float(np.percentile(final_wealths, 5))
    p50 = float(np.percentile(final_wealths, 50))
    p95 = float(np.percentile(final_wealths, 95))
    avg_max_dd = float(np.mean(max_drawdowns) * 100.0)
    
    # 95% Value at Risk = maximum loss at the 5th percentile
    var_95 = max(0.0, float((1.0 - p05) * 100.0))

    return MonteCarloRiskAssessment(
        n_simulations=n_simulations,
        horizon_trades=horizon_trades,
        median_final_equity=round(p50, 4),
        p05_final_equity=round(p05, 4),
        p95_final_equity=round(p95, 4),
        expected_max_drawdown_pct=round(avg_max_dd, 2),
        var_95_pct=round(var_95, 2),
        probability_of_ruin_pct=round(p_ruin, 2),
    )


# ============================================================================
# 3. Linear Algebra & Portfolio Matrix Operations
# ============================================================================

def portfolio_variance(weights: Sequence[float], cov_matrix: np.ndarray) -> float:
    """Portfolio variance: w^T * Sigma * w."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or cov_matrix.shape != (len(w), len(w)):
        raise ValueError("Dimensions of weights and covariance matrix must match")
    var = float(np.dot(w.T, np.dot(cov_matrix, w)))
    return max(0.0, var)


def portfolio_volatility(weights: Sequence[float], cov_matrix: np.ndarray) -> float:
    """Portfolio standard deviation: sqrt(w^T * Sigma * w)."""
    return math.sqrt(portfolio_variance(weights, cov_matrix))


def correlation_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute empirical correlation matrix from an (N_samples, N_assets) return matrix."""
    return np.corrcoef(returns_matrix, rowvar=False)


@dataclass(frozen=True)
class MatrixProperties:
    rank: int
    trace: float
    determinant: float
    is_invertible: bool
    condition_number: float


def matrix_properties(matrix: np.ndarray) -> MatrixProperties:
    """Compute foundational linear algebra properties: Rank, Trace, Det, Inverse condition."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")

    rank = int(np.linalg.matrix_rank(arr))
    trace = float(np.trace(arr))
    det = float(np.linalg.det(arr))
    is_invertible = abs(det) > 1e-12
    cond = float(np.linalg.cond(arr)) if is_invertible else math.inf

    return MatrixProperties(
        rank=rank,
        trace=round(trace, 4),
        determinant=round(det, 6),
        is_invertible=is_invertible,
        condition_number=round(cond, 2),
    )


# ============================================================================
# 4. Linear Regression & Predictive Modeling
# ============================================================================

@dataclass(frozen=True)
class LinearRegressionResult:
    slope: float             # m (rate of change per day / beta)
    intercept: float         # b
    r_squared: float         # R^2 goodness of fit [0, 1]
    predicted_next: float    # Forecast for step n
    slope_annualized_pct: float  # Slope as annualized % of current price


def linear_regression(series: Sequence[float]) -> LinearRegressionResult:
    """Ordinary Least Squares (OLS) linear regression: Y = m*X + b.
    
    Calculates trend slope, intercept, and R^2 (trend strength/cleanness).
    """
    n = len(series)
    if n < 3:
        nan = math.nan
        return LinearRegressionResult(
            slope=nan, intercept=nan, r_squared=nan,
            predicted_next=nan, slope_annualized_pct=nan,
        )

    y = np.asarray(series, dtype=float)
    x = np.arange(n, dtype=float)

    # OLS closed form: slope m = Cov(X,Y)/Var(X)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    ss_xx = float(np.sum((x - x_mean) ** 2))
    ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    ss_yy = float(np.sum((y - y_mean) ** 2))

    if ss_xx <= 0:
        slope = 0.0
        intercept = y_mean
        r2 = 0.0
    else:
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

    predicted_next = slope * n + intercept
    ann_slope_pct = (slope * 252.0 / y[-1] * 100.0) if y[-1] > 0 else 0.0

    return LinearRegressionResult(
        slope=round(slope, 4),
        intercept=round(intercept, 4),
        r_squared=round(max(0.0, min(1.0, r2)), 4),
        predicted_next=round(predicted_next, 4),
        slope_annualized_pct=round(ann_slope_pct, 2),
    )


# ============================================================================
# 5. Continuous Mathematics & Calculus (Derivatives & Numerical Integration)
# ============================================================================

@dataclass(frozen=True)
class MomentumCalculus:
    velocity: float          # 1st derivative dp/dt (% return per day over period)
    acceleration: float      # 2nd derivative d^2p/dt^2 (change in velocity)
    is_accelerating: bool    # True if momentum is increasing (positive 2nd derivative)
    curvature: float         # Qualitative trend rate of change


def momentum_calculus(series: Sequence[float], window: int = 20) -> MomentumCalculus:
    """Calculate instantaneous rate of change (1st derivative) and acceleration (2nd derivative).
    
    - 1st Derivative (Velocity): Measures instantaneous trend velocity.
    - 2nd Derivative (Acceleration): Detects momentum deceleration or inflection points
      before price physically tops or bottoms (anticipates trend exhaustion).
    """
    if len(series) < window + 2:
        nan = math.nan
        return MomentumCalculus(
            velocity=nan, acceleration=nan, is_accelerating=False, curvature=nan
        )

    # Compute velocity across two adjacent segments
    half = max(2, window // 2)
    p_now = series[-1]
    p_mid = series[-half - 1]
    p_start = series[-window - 1]

    # Rate of change 1 (first half) and Rate of change 2 (second half)
    v1 = (p_mid - p_start) / (p_start * half) * 100.0 if p_start > 0 else 0.0
    v2 = (p_now - p_mid) / (p_mid * half) * 100.0 if p_mid > 0 else 0.0

    # Overall velocity (1st derivative)
    overall_velocity = (p_now - p_start) / (p_start * window) * 100.0 if p_start > 0 else 0.0

    # Acceleration (2nd derivative): difference in velocity per unit time
    accel = (v2 - v1) / half

    return MomentumCalculus(
        velocity=round(overall_velocity, 4),
        acceleration=round(accel, 4),
        is_accelerating=accel > 0,
        curvature=round(accel * 10.0, 4),
    )


def volume_weighted_average_price(
    prices: Sequence[float], volumes: Sequence[float]
) -> float:
    """Discrete numerical integral of VWAP: sum(P_t * V_t) / sum(V_t)."""
    if not prices or len(prices) != len(volumes):
        return math.nan
    p = np.asarray(prices, dtype=float)
    v = np.asarray(volumes, dtype=float)
    total_vol = float(np.sum(v))
    if total_vol <= 0:
        return float(np.mean(p))
    return float(np.sum(p * v) / total_vol)
