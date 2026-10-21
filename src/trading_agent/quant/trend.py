"""Time-series momentum (trend following) on a diversified ETF set.

Moskowitz, Ooi & Pedersen (2012): an asset's own past 1-12 month return predicts its next month, across
equities, bonds, commodities and currencies. Each asset is sized to a common volatility so no single
asset dominates, and the book is rebalanced monthly to keep costs low. The edge, where it exists, is
crisis alpha: shorting or sitting out falling markets while the long-only alternative drawdowns.

Unlike the cross-sectional ranking in `research.py` this is not market-neutral, so it is compared with
the buy-and-hold of the same assets and judged on risk-adjusted return and drawdown, not raw return.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .research import stats_row, walk_forward_select
from .stats import block_bootstrap_sharpe, deflated_sharpe, probabilistic_sharpe

PRIOR_TRIALS = 24  # the cross-sectional grid tried before this one; the luck hurdle must include it
PRIOR_TRIALS_AFTER_UNIVERSE_TEST = 30  # plus the six trend variants on the original ETF set
LOOKBACKS = (63, 126, 252)
# Fixed before any result was seen: the diversifiers the tradable universe lacks (bonds, commodities, dollar, ex-US).
# Research only; adding them to config/universe.yaml would widen what the live agent may trade.
DIVERSIFIERS = ("IEF", "LQD", "DBC", "UUP", "EFA", "EEM", "VNQ")


def tsmom_returns(closes: pd.DataFrame, *, lookback: int = 252, long_only: bool = False, target_vol: float = 0.10,
                  vol_window: int = 63, rebalance_days: int = 21, max_gross: float = 2.0,
                  cost_bps: float = 5.0, borrow_bps_annual: float = 50.0) -> pd.Series:
    """Daily net return. Each asset gets sign(lookback return) x (target_vol / its own vol) / N assets.

    Weights are decided at a close and earn the next day's return. Gross exposure is capped so a calm
    market cannot lever the book without limit.
    """
    rets = closes.pct_change(fill_method=None)
    vol = rets.rolling(vol_window).std() * math.sqrt(252)
    signal = np.sign(closes / closes.shift(lookback) - 1)
    if long_only:
        signal = signal.clip(lower=0)
    n = closes.shape[1]
    raw = signal * (target_vol / vol.replace(0, np.nan)) / n
    gross = raw.abs().sum(axis=1)
    raw = raw.div(np.maximum(gross / max_gross, 1.0), axis=0)  # scale down only when over the cap
    ready = raw.notna().all(axis=1)
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    last = None
    for i in range(len(w)):
        if i % rebalance_days == 0 and ready.iloc[i]:
            last = raw.iloc[i]
        if last is not None:
            w.iloc[i] = last
    held = w.shift(1).fillna(0.0)
    book = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    borrow = (-held.clip(upper=0)).sum(axis=1) * borrow_bps_annual / 1e4 / 252
    return book - turnover * cost_bps / 1e4 - borrow


def trend_grid() -> list[dict]:
    return [{"lookback": lb, "long_only": lo} for lb, lo in itertools.product(LOOKBACKS, (False, True))]


def _label(p: dict) -> str:
    return f"tsmom|lb={p['lookback']}d|{'long-only' if p['long_only'] else 'long-short'}"


def trend_report(closes: pd.DataFrame, *, prior_trials: int = PRIOR_TRIALS) -> dict:
    grid = trend_grid()
    results = {_label(p): tsmom_returns(closes, **p) for p in grid}
    wf = walk_forward_select(results)
    oos = wf.returns
    hold = closes.pct_change(fill_method=None).mean(axis=1)
    total_trials = len(grid) + prior_trials
    var = float(np.var(list(wf.trial_sharpes.values()))) / 252
    lo, hi = block_bootstrap_sharpe(oos)
    return {
        "universe": {"assets": list(closes.columns), "start": str(closes.index[0].date()), "end": str(closes.index[-1].date())},
        "trials_this_grid": len(grid), "trials_including_prior_work": total_trials,
        "in_sample_all_variants": {k: stats_row(r, hold) for k, r in results.items()},
        "walk_forward_oos": stats_row(oos, hold.reindex(oos.index)) if len(oos) else {},
        "oos_sharpe_95ci": [None if math.isnan(lo) else round(lo, 2), None if math.isnan(hi) else round(hi, 2)],
        "probabilistic_sharpe_vs_0": round(probabilistic_sharpe(oos), 3) if len(oos) else None,
        "deflated_sharpe": round(deflated_sharpe(oos, total_trials, var), 3) if len(oos) else None,
        "buy_and_hold_equal_weight": stats_row(hold.reindex(oos.index).dropna()) if len(oos) else {},
        "folds": [f.__dict__ for f in wf.folds],
        "caveats": [
            "ETF set is today's listings; every asset here existed for the whole sample, which limits but does not remove survivorship",
            "monthly rebalance on closes: no intraday slippage, and a 5 bps/turnover cost is an assumption, not a measurement",
            "the deflated Sharpe includes the cross-sectional grid already tried, so it is deliberately harsh",
        ],
    }
