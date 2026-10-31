from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent import portfolio_risk as pr


def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-12-01", "2026-10-30")
    spy = rng.normal(0.0003, 0.01, len(idx))
    aaa = 1.5 * spy + rng.normal(0, 0.01, len(idx))   # market-driven, beta about 1.5
    bbb = rng.normal(0, 0.02, len(idx))                # independent and twice as volatile as SPY
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in {"SPY": spy, "AAA": aaa, "BBB": bbb}.items()}, index=idx)


def test_lw_cov_is_symmetric_positive_and_keeps_volatility():
    r = prices().pct_change().dropna().to_numpy()
    cov = pr.lw_cov(r)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    assert np.sqrt(cov[0, 0]) == pytest.approx(r[:, 0].std() * np.sqrt(252), rel=0.01)  # diagonal untouched


def test_analyze_core_only_is_market_risk():
    out = pr.analyze({"SPYM": 1.0}, prices())
    assert out["beta"] == pytest.approx(1.0, abs=0.02)
    assert out["vol_pct"] == pytest.approx(16.0, abs=2.5)        # 1% daily is about 16% a year
    assert out["top_risk"] == [("SPY", 100.0)]


def test_analyze_satellites_raise_risk_and_need_a_bigger_core():
    base = pr.analyze({"SPYM": 0.8, "AAA": 0.1, "BBB": 0.1}, prices())
    heavy = pr.analyze({"SPYM": 0.2, "AAA": 0.4, "BBB": 0.4}, prices())
    assert heavy["vol_pct"] > base["vol_pct"]                    # beta need not rise: BBB is volatile but market-neutral
    assert heavy["top_risk"][0][0] == "BBB"                      # the volatile name dominates risk, not just weight
    assert sum(c for _, c in heavy["top_risk"]) == pytest.approx(100.0, abs=1.5)  # contributions sum to the total
    assert heavy["core_for_target_vol_pct"] is not None and base["effective_bets"] <= 3.0
    loose = pr.analyze({"SPYM": 0.2, "AAA": 0.4, "BBB": 0.4}, prices(), target_vol=0.30)
    assert loose["core_for_target_vol_pct"] <= heavy["core_for_target_vol_pct"]  # a looser target needs less core
    assert "replay_2022_pct" in heavy and "mom_12_1_pct" in heavy["per_symbol"]["AAA"]
    assert "Volatility" in pr.render_risk(heavy)
