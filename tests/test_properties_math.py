"""Property tests for the quant math engine."""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from trading_agent.math_quant import (
    bayes_posterior_probability,
    dispersion_metrics,
    linear_regression,
    mean,
    median,
    monte_carlo_trade_simulation,
    portfolio_variance,
    volume_weighted_average_price,
)

# Market data (prices, volumes, returns) lives far below float overflow; the bound only keeps sums finite.
market_number = st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False)
price = st.floats(min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False)
probability = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
series = st.lists(market_number, min_size=1, max_size=200)


@given(series)
def test_mean_and_median_stay_within_the_data_range(xs):
    lo, hi = min(xs), max(xs)
    slack = 1e-6 * max(abs(lo), abs(hi), 1.0)  # float summation error, not a contract slack
    assert lo - slack <= mean(xs) <= hi + slack
    assert lo <= median(xs) <= hi


@given(series.flatmap(lambda xs: st.tuples(st.just(xs), st.permutations(xs))))
def test_median_ignores_order(pair):
    xs, shuffled = pair
    assert median(xs) == median(shuffled)


@given(st.just([]) | st.lists(market_number, max_size=3))
def test_dispersion_needs_four_points(xs):
    d = dispersion_metrics(xs)
    assert all(math.isnan(v) for v in vars(d).values())


@given(st.lists(market_number, min_size=4, max_size=200))
def test_dispersion_invariants(xs):
    d = dispersion_metrics(xs)
    assert d.range_abs >= 0 and d.variance >= 0 and d.standard_deviation >= 0
    assert d.mean_absolute_deviation >= 0
    assert d.q1 <= d.q3
    assert d.q3 - d.q1 <= d.range_abs + 1.5e-4 + 1e-9  # q1, q3 and the range are each rounded to 4 dp: 3 x 5e-5


@given(st.floats(min_value=-1e9, max_value=1e9, allow_nan=False), st.integers(min_value=4, max_value=100))
def test_a_constant_series_has_no_spread(c, n):
    d = dispersion_metrics([c] * n)
    assert d.range_abs == 0 and d.quartile_deviation == 0
    assert d.variance == 0 and d.standard_deviation == 0


@given(probability, probability, probability)
def test_bayes_posterior_is_a_probability(prior, lt, lf):
    assert 0.0 <= bayes_posterior_probability(prior, lt, lf) <= 1.0


@given(probability, probability)
def test_uninformative_evidence_leaves_the_prior_alone(prior, lik):
    # Float division, so equal up to rounding.
    assert math.isclose(bayes_posterior_probability(prior, lik, lik), prior, abs_tol=1e-9)


@given(probability, probability, probability, probability)
def test_stronger_evidence_for_a_win_never_lowers_the_posterior(prior, lf, a, b):
    lo, hi = sorted((a, b))
    assert bayes_posterior_probability(prior, lo, lf) <= bayes_posterior_probability(prior, hi, lf) + 1e-9


@given(probability, st.floats(min_value=0.0, max_value=1.0), probability)
def test_certain_priors_are_fixed_points(_, lt, lf):
    assert bayes_posterior_probability(0.0, lt, lf) == 0.0
    assert bayes_posterior_probability(1.0, lt, lf) == 1.0


@given(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    st.integers(min_value=3, max_value=120),
)
def test_regression_recovers_an_exact_line(intercept, slope, n):
    fit = linear_regression([intercept + slope * i for i in range(n)])
    # slope, intercept and the forecast are rounded to 4 dp, and the data are exact to float error.
    assert math.isclose(fit.slope, slope, abs_tol=1e-3)
    assert math.isclose(fit.intercept, intercept, abs_tol=1e-2)
    assert math.isclose(fit.predicted_next, intercept + slope * n, abs_tol=1e-2)
    assert 0.0 <= fit.r_squared <= 1.0


@given(st.lists(st.tuples(price, st.floats(min_value=0, max_value=1e9, allow_nan=False)), min_size=1, max_size=100))
def test_vwap_is_a_weighted_average_of_the_prices(rows):
    prices = [p for p, _ in rows]
    volumes = [v for _, v in rows]
    v = volume_weighted_average_price(prices, volumes)
    slack = 1e-9 * max(prices)
    assert min(prices) - slack <= v <= max(prices) + slack


@given(st.lists(price, min_size=1, max_size=20), st.lists(price, max_size=20))
def test_vwap_rejects_mismatched_lengths(prices, volumes):
    if len(prices) != len(volumes):
        assert math.isnan(volume_weighted_average_price(prices, volumes))


@given(st.integers(min_value=1, max_value=8).flatmap(
    lambda n: st.tuples(
        st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False), min_size=n, max_size=n),
        st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False), min_size=n * n, max_size=n * n),
    )))
def test_portfolio_variance_is_never_negative_for_a_covariance_matrix(case):
    w, flat = case
    n = len(w)
    a = np.array(flat).reshape(n, n)
    cov = a @ a.T  # any A A^T is a valid covariance matrix
    assert portfolio_variance(w, cov) >= 0.0


@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.1, max_value=10.0),
    st.floats(min_value=0.01, max_value=50.0),
    st.integers(min_value=1, max_value=40),
    st.floats(min_value=1.0, max_value=99.0),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_monte_carlo_is_ordered_bounded_and_deterministic(win, rr, risk, horizon, ruin, seed):
    kw = dict(win_rate=win, reward_risk=rr, risk_per_trade_pct=risk, horizon_trades=horizon,
              n_simulations=200, ruin_threshold_pct=ruin, seed=seed)
    r = monte_carlo_trade_simulation(**kw)
    assert r.p05_final_equity <= r.median_final_equity <= r.p95_final_equity
    assert 0.0 <= r.probability_of_ruin_pct <= 100.0
    assert 0.0 <= r.expected_max_drawdown_pct <= 100.0
    assert r.var_95_pct >= 0.0
    assert monte_carlo_trade_simulation(**kw) == r
