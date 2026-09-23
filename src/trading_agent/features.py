"""Deterministic features computed from daily bars, packaged as citable evidence.

The LLM sees these numbers and must cite their evidence IDs. The verifier
recomputes everything it relies on from the same bars, never from LLM text.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from statistics import fmean, pstdev

from .schemas import Bar, Evidence, Quote


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else math.nan


def atr(bars: list[Bar], n: int = 14) -> float:
    if len(bars) < n + 1:
        return math.nan
    trs = [max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close)) for p, b in zip(bars[-n - 1:-1], bars[-n:])]
    return fmean(trs)


def price_features(bars: list[Bar]) -> dict[str, float]:
    """Features as of the last bar's close. Requires ~200 bars for the long averages."""
    closes = [b.close for b in bars]
    c = closes[-1]

    def ret(n: int) -> float:
        return _pct(c, closes[-n - 1]) if len(closes) > n else math.nan

    def sma(n: int) -> float:
        return fmean(closes[-n:]) if len(closes) >= n else math.nan

    daily = [_pct(b, a) / 100 for a, b in zip(closes[-21:-1], closes[-20:])]
    a14 = atr(bars)
    sma20, sma50, sma200 = sma(20), sma(50), sma(200)
    vols = [b.volume for b in bars[-20:]]
    high_252 = max(b.high for b in bars[-252:])
    return {
        "close": round(c, 4),
        "ret_5d_pct": round(ret(5), 2),
        "ret_20d_pct": round(ret(20), 2),
        "ret_60d_pct": round(ret(60), 2),
        "ret_120d_pct": round(ret(120), 2),
        "dist_sma20_pct": round(_pct(c, sma20), 2),
        "dist_sma50_pct": round(_pct(c, sma50), 2),
        "dist_sma200_pct": round(_pct(c, sma200), 2),
        "sma50_above_sma200": float(sma50 > sma200) if not math.isnan(sma200) else math.nan,
        "atr14": round(a14, 4),
        "atr14_pct": round(a14 / c * 100, 2),
        "realized_vol_20d_ann_pct": round(pstdev(daily) * math.sqrt(252) * 100, 2) if len(daily) >= 10 else math.nan,
        "avg_volume_20d": round(fmean(vols), 0),
        "avg_dollar_volume_20d": round(fmean(vols) * c, 0),
        "dist_52w_high_pct": round(_pct(c, high_252), 2),
        "n_bars": float(len(bars)),
    }


def _eid(prefix: str, symbol: str | None, as_of: datetime, payload: dict) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{prefix}_{symbol or 'mkt'}_{as_of:%Y%m%d}_{h}"


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def build_evidence(bars_by_symbol: dict[str, list[Bar]], quotes: dict[str, Quote],
                   benchmark: str) -> tuple[list[Evidence], dict[str, dict[str, float]]]:
    evidence: list[Evidence] = []
    features: dict[str, dict[str, float]] = {}
    for sym, bars in sorted(bars_by_symbol.items()):
        if len(bars) < 60:
            continue
        f = price_features(bars)
        f["bar_date"] = bars[-1].ts.date().isoformat()  # type: ignore[assignment]
        features[sym] = f
        payload = _clean(f)
        evidence.append(Evidence(evidence_id=_eid("px", sym, bars[-1].ts, payload), kind="price_features",
                                 symbol=sym, as_of=bars[-1].ts, source="webull.daily_bars", payload=payload))
    for sym, q in sorted(quotes.items()):
        payload = {"bid": q.bid, "ask": q.ask, "last": q.last, "spread_pct": round(q.spread_pct, 3)}
        evidence.append(Evidence(evidence_id=_eid("qt", sym, q.fetched_at, payload), kind="quote", symbol=sym,
                                 as_of=q.fetched_at, source="webull.snapshot", payload=payload))
    if benchmark in features:
        above50 = [s for s, f in features.items() if (f.get("dist_sma50_pct") or 0) > 0]
        b = features[benchmark]
        payload = _clean({
            "benchmark": benchmark,
            "benchmark_above_sma200": (b.get("dist_sma200_pct") or 0) > 0,
            "benchmark_ret_20d_pct": b.get("ret_20d_pct"),
            "benchmark_vol_20d_ann_pct": b.get("realized_vol_20d_ann_pct"),
            "universe_pct_above_sma50": round(len(above50) / len(features) * 100, 1),
        })
        as_of = bars_by_symbol[benchmark][-1].ts
        evidence.append(Evidence(evidence_id=_eid("regime", None, as_of, payload), kind="regime", symbol=None,
                                 as_of=as_of, source="derived", payload=payload))
    return evidence, features
