"""LSE adapter: parsing, cross-check, regime scale. Uses a fake client, so no key or network."""

from __future__ import annotations

from datetime import date, datetime, timezone

from trading_agent.lse_data import crosscheck, daily_bars, macro_series, regime_scale
from trading_agent.schemas import Bar


class FakeLSE:
    def candles(self, symbol, timeframe, start=None, end=None, limit=5000):
        assert timeframe == "1d"
        return [
            {"timestamp": "2026-10-01T00:00:00Z", "open": 1, "high": 2, "low": 1, "close": 100.0, "volume": 5},
            {"timestamp": "2026-10-02T00:00:00Z", "open": 1, "high": 2, "low": 1, "close": 110.0, "volume": 5},
        ]

    def series(self, code, start=None):
        return [{"date": "2026-10-01", "value": 4.2}, {"date": "2026-10-02", "value": None}]


def _bar(day: int, close: float) -> Bar:
    return Bar(ts=datetime(2026, 10, day, tzinfo=timezone.utc), open=1, high=2, low=1, close=close, volume=1)


def test_parsing():
    bars = daily_bars("AAPL", "2026-10-01", "2026-10-02", client=FakeLSE())
    assert [b.close for b in bars] == [100.0, 110.0] and bars[0].ts.tzinfo is not None
    assert macro_series("US10Y", client=FakeLSE()) == [(date(2026, 10, 1), 4.2)]  # None value dropped


def test_crosscheck_flags_only_real_gaps():
    lse = daily_bars("AAPL", "x", "y", client=FakeLSE())
    assert crosscheck(lse, [_bar(1, 100.2), _bar(2, 110.0)]).ok  # 0.2% < 0.5% tol
    cc = crosscheck(lse, [_bar(1, 100.0), _bar(2, 120.0), _bar(3, 1.0)])
    assert cc.mismatches == [(date(2026, 10, 2), 110.0, 120.0)] and cc.only_ref == 1 and not cc.ok
    assert not crosscheck(lse, []).ok  # no overlap is a failure, not a pass


def test_regime_scale():
    calm = [15.0 + (i % 3) * 0.1 for i in range(100)]
    assert regime_scale(calm) == 1.0
    assert regime_scale(calm + [40.0]) == 0.5
    assert regime_scale([50.0] * 10) == 1.0  # too little history
