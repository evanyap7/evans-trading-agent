from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent import advisor_score as s


def closes(n: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=n)
    return pd.DataFrame({"SPY": 100.0 * (1.0005 ** np.arange(n)), "UP": 100.0 * (1.003 ** np.arange(n)),
                         "DOWN": 100.0 * (0.999 ** np.arange(n))}, index=idx)


def call(day: str, sym: str, direction: str, p: float = 0.7, h: int = 5, arm: str = "news", c: pd.DataFrame | None = None) -> dict:
    c = c if c is not None else closes()
    ts = pd.Timestamp(day)
    return {"date": day, "arm": arm, "symbol": sym, "direction": direction, "horizon_days": h, "p": p,
            "price": float(c.loc[ts, sym]), "spy_price": float(c.loc[ts, "SPY"])}


def test_settle_direction_and_open_horizon():
    c = closes()
    assert s.settle(call("2026-01-05", "UP", "up", c=c), c) is True       # beats SPY
    assert s.settle(call("2026-01-05", "UP", "down", c=c), c) is False
    assert s.settle(call("2026-01-05", "DOWN", "down", c=c), c) is True   # trails SPY
    assert s.settle(call(str(c.index[-2].date()), "UP", "up", c=c), c) is None  # horizon not elapsed yet


def test_independence_keeps_first_call_per_horizon_window():
    c = closes()
    a, b, later = (call(d, "UP", "up", c=c) for d in ("2026-01-05", "2026-01-07", "2026-01-20"))
    kept = s.independent([a, b, later], c)
    assert [k["date"] for k in kept] == ["2026-01-05", "2026-01-20"]      # Jan 7 overlaps the Jan 5 window
    other_arm = call("2026-01-07", "UP", "up", arm="blind", c=c)
    assert len(s.independent([a, other_arm], c)) == 2                     # arms are scored separately


def test_wilson_interval_matches_known_value():
    lo, hi = s.hit_interval(555, 1000)
    assert lo == pytest.approx(0.524, abs=0.002) and hi == pytest.approx(0.586, abs=0.002)


def test_no_verdict_before_min_n_and_perfect_arm_passes_at_min_n():
    c = closes(1900)
    days = [str(d.date()) for d in c.index[:300 * 6:6]][:300]          # 300 spaced-out windows on one symbol
    news = [call(d, "UP", "up", p=0.9, c=c) for d in days]
    blind = [call(d, "UP", "up", p=0.5, arm="blind", c=c) for d in days]
    few = s.evaluate(news[:20] + blind[:20], c)
    assert not few["ready"] and not few["passed"] and "VERDICT: none yet" in s.render(few)
    full = s.evaluate(news + blind, c)
    assert full["stats"]["news"]["n"] == 300 and full["ready"] and full["passed"]
    assert full["brier_diff"][2] < 0                                    # news clearly beats blind on Brier
    wrong = s.evaluate([call(d, "UP", "down", p=0.9, c=c) for d in days] + blind, c)
    assert wrong["ready"] and not wrong["passed"]
