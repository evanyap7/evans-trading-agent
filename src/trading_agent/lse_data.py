"""London Strategic Edge (LSE Data 3.0) adapter: daily bars, macro series, and a cross-check vs a trusted source.

Research-only until `crosscheck` passes against Databento/yfinance. Data terms: personal research/trading
use is allowed, redistribution is not, so never commit cached LSE data. Terms: https://londonstrategicedge.com/terms
Client: `uv add lse-data` (PyPI), key in LSE_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean, pstdev
from typing import Any

from .schemas import Bar


def _client() -> Any:
    from lse import LSE  # lazy: the package is optional until the user installs it

    if not os.environ.get("LSE_API_KEY"):
        raise ValueError("LSE_API_KEY is not set (add it to .env)")
    return LSE()


def daily_bars(symbol: str, start: str, end: str, client: Any = None) -> list[Bar]:
    """Daily OHLCV for a US stock/ETF. ponytail: one call returns <=5000 bars (~20y); page by timestamp beyond that."""
    rows = (client or _client()).candles(symbol, "1d", start=start, end=end, limit=5000)
    return [
        Bar(
            ts=datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00")),
            open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
            close=float(r["close"]), volume=float(r["volume"]),
        )
        for r in rows
    ]


def macro_series(code: str, start: str | None = None, client: Any = None) -> list[tuple[date, float]]:
    """A macro observation or bond-yield series, e.g. 'US10Y'. Rows with a missing value are dropped."""
    rows = (client or _client()).series(code, start=start)
    return [(date.fromisoformat(str(r["date"])[:10]), float(r["value"])) for r in rows if r.get("value") is not None]


@dataclass(frozen=True)
class CrossCheck:
    common_days: int
    only_lse: int
    only_ref: int
    mismatches: list[tuple[date, float, float]]  # (day, lse_close, ref_close) where |rel diff| > tol

    @property
    def ok(self) -> bool:
        return self.common_days > 0 and not self.mismatches


def crosscheck(lse: list[Bar], ref: list[Bar], tol: float = 0.005) -> CrossCheck:
    """Compare daily closes by calendar date. tol is relative (0.005 = 0.5%).

    ponytail: keys on the UTC date of each bar; if LSE stamps sessions at a different offset, every day will
    show as only_lse/only_ref and the first real run will tell us to shift the key.
    """
    a = {b.ts.date(): b.close for b in lse}
    b_ = {b.ts.date(): b.close for b in ref}
    common = sorted(a.keys() & b_.keys())
    bad = [(d, a[d], b_[d]) for d in common if abs(a[d] - b_[d]) > tol * abs(b_[d])]
    return CrossCheck(len(common), len(a.keys() - b_.keys()), len(b_.keys() - a.keys()), bad)


def regime_scale(stress: list[float], window: int = 252, z: float = 1.0, risk_off_scale: float = 0.5) -> float:
    """Position-size multiplier from a stress series (VIX-like: higher = worse).

    1.0 normally; `risk_off_scale` when the latest value is more than `z` stdevs above its trailing-window mean.
    Not a signal: it only shrinks size. Unwired until validated in the backtest (pre-register the test first).
    """
    w = stress[-window:]
    if len(w) < 30:
        return 1.0  # not enough history to call a regime
    sd = pstdev(w)
    return risk_off_scale if sd > 0 and (w[-1] - fmean(w)) / sd > z else 1.0
