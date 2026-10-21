"""Quant layer: statistics are correct, signals never see the future, noise is not mistaken for edge."""

import numpy as np
import pandas as pd

from trading_agent.quant.research import Config, long_short_returns, research_report, walk_forward
from trading_agent.quant.signals import composite, raw_signals, zscore
from trading_agent.quant.stats import (block_bootstrap_sharpe, deflated_sharpe, expected_max_sharpe, max_drawdown,
                                       probabilistic_sharpe, sharpe)


def panel(n_days=1500, n_sym=40, drift_per_rank=0.0, seed=1):
    """Random-walk prices. A positive `drift_per_rank` plants momentum: the strongest recent past keeps winning."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, size=(n_days, n_sym))
    if drift_per_rank:
        trend = rng.normal(0, 1, size=n_sym)  # persistent per-stock drift, so the past predicts the future
        rets += trend * drift_per_rank
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"S{i}" for i in range(n_sym)])


def test_sharpe_and_drawdown_known_values():
    r = np.array([0.01, -0.01] * 50)
    assert abs(sharpe(r)) < 0.2
    assert max_drawdown([0.1, -0.5, 0.1]) < -0.49
    assert max_drawdown([0.01, 0.01]) == 0.0


def test_psr_rises_with_a_real_edge_and_sample_length():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.002, 0.01, 1000)
    noise = rng.normal(0.0, 0.01, 1000)
    assert probabilistic_sharpe(strong) > 0.99
    assert 0.05 < probabilistic_sharpe(noise) < 0.95


def test_deflation_penalises_many_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, 750)
    one, many = deflated_sharpe(r, 1), deflated_sharpe(r, 500, sr_variance=1 / 750)
    assert many < one
    assert expected_max_sharpe(500, 0.01) > expected_max_sharpe(5, 0.01) > 0


def test_bootstrap_interval_brackets_the_point_estimate():
    r = np.random.default_rng(3).normal(0.001, 0.01, 800)
    lo, hi = block_bootstrap_sharpe(r, n_boot=500)
    assert lo < sharpe(r) < hi


def test_zscore_rows_are_standardised_and_clipped():
    df = pd.DataFrame(np.random.default_rng(4).normal(size=(5, 30)))
    df.iloc[0, 0] = 50
    z = zscore(df)
    assert z.max().max() <= 3.0 and abs(z.iloc[1].mean()) < 1e-9


def test_signals_have_no_lookahead():
    closes = panel(400)
    base = composite(raw_signals(closes))
    tampered = closes.copy()
    tampered.iloc[300:] *= 3  # change only the future relative to row 299
    after = composite(raw_signals(tampered))
    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_book_is_dollar_neutral_and_pays_costs():
    closes = panel(600)
    free = long_short_returns(closes, Config(cost_bps=0, borrow_bps_annual=0))
    costly = long_short_returns(closes, Config(cost_bps=25))
    assert costly["net"].sum() < free["net"].sum()
    assert (costly["turnover"] > 0).any()


def test_a_planted_momentum_effect_is_found_out_of_sample():
    closes = panel(1800, drift_per_rank=0.0015, seed=5)
    wf = walk_forward(closes)
    assert len(wf.folds) >= 3
    assert sharpe(wf.returns) > 0.5
    assert sum("momentum" in f.chosen for f in wf.folds) >= len(wf.folds) // 2


def test_trend_following_earns_on_persistent_trends_and_ignores_the_future():
    from trading_agent.quant.trend import tsmom_returns

    closes = panel(900, n_sym=6, drift_per_rank=0.0015, seed=7)
    assert sharpe(tsmom_returns(closes, lookback=63)) > 0.5
    tampered = closes.copy()
    tampered.iloc[600:] *= 0.2  # a crash that only exists after row 599
    pd.testing.assert_series_equal(tsmom_returns(closes).iloc[:600], tsmom_returns(tampered).iloc[:600])


def test_trend_long_short_profits_from_a_falling_asset_long_only_does_not():
    from trading_agent.quant.trend import tsmom_returns

    idx = pd.bdate_range("2020-01-01", periods=600)
    closes = pd.DataFrame({"A": 100 * np.cumprod(1 - 0.001 * np.ones(600)),
                           "B": 100 * np.cumprod(1 - 0.001 * np.ones(600))}, index=idx)
    closes *= 1 + 0.004 * np.sin(np.arange(600))[:, None]  # a little noise so volatility is non-zero
    ls = tsmom_returns(closes, lookback=63, long_only=False)
    lo = tsmom_returns(closes, lookback=63, long_only=True)
    assert ls.sum() > lo.sum() and abs(lo.iloc[200:]).sum() == 0


def test_pure_noise_is_not_called_significant():
    rep = research_report(panel(1800, seed=6))
    assert rep["deflated_sharpe"] < 0.95
    assert rep["trials"] == 24
