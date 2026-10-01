"""Unit tests for the quantitative mathematics engine (trading_agent.math_quant)."""

import math
import numpy as np
import pytest

from trading_agent.math_quant import (
    bayes_posterior_probability,
    correlation_matrix,
    dispersion_metrics,
    linear_regression,
    matrix_properties,
    mean,
    median,
    mode_bucketed,
    momentum_calculus,
    monte_carlo_trade_simulation,
    moving_average_crossover,
    portfolio_variance,
    portfolio_volatility,
    volume_weighted_average_price,
)


def test_central_tendency():
    vals = [12.0, 13.0, 6.0, 7.0, 19.0, 21.0]
    # mean is (12+13+6+7+19+21)/6 = 78/6 = 13.0
    assert mean(vals) == pytest.approx(13.0)
    # sorted: [6, 7, 12, 13, 19, 21] -> median is (12+13)/2 = 12.5
    assert median(vals) == pytest.approx(12.5)

    # Bucketed mode
    repeated = [100.0, 100.2, 100.1, 100.3, 105.0, 110.0, 100.05]
    m = mode_bucketed(repeated, n_bins=5)
    assert 99.0 <= m <= 102.0


def test_moving_average_crossover():
    # Construct 60 ascending values
    closes = [100.0 + i * 0.5 for i in range(60)]
    cross = moving_average_crossover(closes, fast_period=20, slow_period=50)
    assert cross.is_bullish
    assert cross.fast_sma > cross.slow_sma
    assert cross.crossover_ratio > 0

    # Short series
    short_cross = moving_average_crossover([100.0, 101.0], fast_period=20, slow_period=50)
    assert math.isnan(short_cross.fast_sma)
    assert not short_cross.is_bullish


def test_dispersion_metrics():
    # Sample data: 8 points
    vals = [3.0, 6.0, 6.0, 7.0, 8.0, 11.0, 15.0, 16.0]
    # Mean is 72/8 = 9.0
    dm = dispersion_metrics(vals)
    assert dm.range_abs == 16.0 - 3.0  # 13.0
    assert dm.quartile_deviation > 0
    assert dm.mean_absolute_deviation > 0
    assert dm.variance > 0
    assert dm.standard_deviation == pytest.approx(math.sqrt(dm.variance), rel=1e-3)


def test_bayes_posterior_probability():
    # Base rate of a successful setup: 40%
    # With strong momentum alignment: true positive rate 70%, false positive rate 30%
    prior = 0.40
    post = bayes_posterior_probability(prior_prob=prior, likelihood_true=0.70, likelihood_false=0.30)
    # Bayes theorem: (0.7 * 0.4) / (0.7 * 0.4 + 0.3 * 0.6) = 0.28 / (0.28 + 0.18) = 0.28 / 0.46 = 0.6087
    assert post == pytest.approx(0.6087, abs=1e-3)
    assert post > prior


def test_monte_carlo_trade_simulation():
    # 55% win rate, 2.0 reward/risk, 1% risk per trade over 50 trades
    sim = monte_carlo_trade_simulation(
        win_rate=0.55,
        reward_risk=2.0,
        risk_per_trade_pct=1.0,
        horizon_trades=50,
        n_simulations=1000,
        ruin_threshold_pct=25.0,
    )
    assert sim.n_simulations == 1000
    assert sim.median_final_equity > 1.0  # positive expectancy
    assert sim.p95_final_equity > sim.p05_final_equity
    assert sim.expected_max_drawdown_pct >= 0.0


def test_linear_algebra_and_portfolio_matrix():
    weights = [0.6, 0.4]
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    var = portfolio_variance(weights, cov)
    vol = portfolio_volatility(weights, cov)
    # Manual: 0.6^2 * 0.04 + 0.4^2 * 0.09 + 2 * 0.6 * 0.4 * 0.01
    # = 0.36*0.04 + 0.16*0.09 + 0.48*0.01 = 0.0144 + 0.0144 + 0.0048 = 0.0336
    assert var == pytest.approx(0.0336, abs=1e-5)
    assert vol == pytest.approx(math.sqrt(0.0336), abs=1e-5)

    # Matrix properties
    props = matrix_properties(cov)
    assert props.rank == 2
    assert props.is_invertible
    assert props.determinant > 0


def test_linear_regression():
    # Perfect trend y = 2x + 10
    x = np.arange(20, dtype=float)
    y = 2.0 * x + 10.0
    res = linear_regression(y)
    assert res.slope == pytest.approx(2.0, abs=1e-3)
    assert res.intercept == pytest.approx(10.0, abs=1e-3)
    assert res.r_squared == pytest.approx(1.0, abs=1e-3)
    assert res.predicted_next == pytest.approx(2.0 * 20 + 10.0, abs=1e-3)


def test_momentum_calculus():
    # Accelerating price series: quadratic growth y = t^2
    t = np.arange(30, dtype=float)
    prices = 100.0 + (t ** 2) * 0.1
    calc = momentum_calculus(prices, window=20)
    assert calc.velocity > 0
    assert calc.acceleration > 0
    assert calc.is_accelerating


def test_volume_weighted_average_price():
    prices = [10.0, 11.0, 12.0]
    volumes = [100.0, 200.0, 100.0]
    # VWAP = (10*100 + 11*200 + 12*100) / 400 = (1000 + 2200 + 1200) / 400 = 4400 / 400 = 11.0
    vwap = volume_weighted_average_price(prices, volumes)
    assert vwap == pytest.approx(11.0)
