"""Walk-forward research harness for cross-sectional long/short signals.

Honest by construction:
  * signals use data up to day t; the book earns the return from t to t+1, never the same bar;
  * every rebalance pays commission+slippage on turnover, and the short book pays borrow;
  * parameters are chosen on a training window and scored only on the unseen window that follows;
  * the out-of-sample Sharpe is deflated by the number of configurations that were tried.

What it cannot fix: the universe is today's survivors, so absolute returns are flattered. The
long/short book is dollar-neutral, which cancels most of that bias, but not all of it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import DEFAULT_WEIGHTS, composite, raw_signals
from .stats import block_bootstrap_sharpe, cagr, deflated_sharpe, max_drawdown, probabilistic_sharpe, sharpe

TRAIN_DAYS, TEST_DAYS = 504, 126  # two years to choose, six months to judge


@dataclass(frozen=True)
class Config:
    signal: str = "composite"      # one of momentum / reversal / low_vol / composite
    quantile: float = 0.2          # fraction of the universe held on each side
    rebalance_days: int = 5
    cost_bps: float = 10.0         # per unit of turnover: half-spread plus slippage plus commission
    borrow_bps_annual: float = 50.0

    def label(self) -> str:
        return f"{self.signal}|q={self.quantile}|rb={self.rebalance_days}d"


def build_closes(bars: dict[str, list]) -> pd.DataFrame:
    frame = {s: pd.Series({b.ts.date(): b.close for b in bs}) for s, bs in bars.items() if bs}
    out = pd.DataFrame(frame).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def scores_for(closes: pd.DataFrame, signal: str) -> pd.DataFrame:
    raw = raw_signals(closes)
    return composite(raw, DEFAULT_WEIGHTS) if signal == "composite" else composite(raw, {signal: 1.0})


def target_weights(score: pd.DataFrame, quantile: float, rebalance_days: int) -> pd.DataFrame:
    """Equal-weight top/bottom quantile, +0.5 gross long and -0.5 gross short, held between rebalances."""
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    last = None
    for i, (_, row) in enumerate(score.iterrows()):
        if i % rebalance_days == 0 and row.notna().sum() >= max(4, int(1 / quantile)):
            k = max(1, int(round(row.notna().sum() * quantile)))
            ranked = row.dropna().sort_values()
            new = pd.Series(0.0, index=score.columns)
            new[ranked.index[-k:]] = 0.5 / k
            new[ranked.index[:k]] = -0.5 / k
            last = new
        if last is not None:
            w.iloc[i] = last
    return w


def long_short_returns(closes: pd.DataFrame, cfg: Config, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily net return of the book decided at each close and held to the next. Columns: net, gross, turnover."""
    score = scores if scores is not None else scores_for(closes, cfg.signal)
    w = target_weights(score, cfg.quantile, cfg.rebalance_days)
    held = w.shift(1).fillna(0.0)                       # decided yesterday, earns today
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    borrow = (-held.clip(upper=0)).sum(axis=1) * cfg.borrow_bps_annual / 1e4 / 252
    net = gross - turnover * cfg.cost_bps / 1e4 - borrow
    return pd.DataFrame({"net": net, "gross": gross, "turnover": turnover})


def default_grid() -> list[Config]:
    return [Config(signal=s, quantile=q, rebalance_days=r)
            for s, q, r in itertools.product(["momentum", "reversal", "low_vol", "composite"],
                                             [0.2, 0.3], [5, 10, 21])]


@dataclass
class Fold:
    train: tuple[str, str]
    test: tuple[str, str]
    chosen: str
    train_sharpe: float
    test_sharpe: float


@dataclass
class WalkForward:
    returns: pd.Series
    folds: list[Fold]
    n_trials: int
    trial_sharpes: dict[str, float] = field(default_factory=dict)


def walk_forward_select(results: dict[str, pd.Series], *, train_days: int = TRAIN_DAYS,
                        test_days: int = TEST_DAYS) -> WalkForward:
    """Pick the best-Sharpe candidate on each training window and keep only its return on the next test window."""
    index = next(iter(results.values())).index
    folds, oos = [], []
    start = 0
    while start + train_days + test_days <= len(index):
        tr = slice(start, start + train_days)
        te = slice(start + train_days, start + train_days + test_days)
        best = max(results, key=lambda k: sharpe(results[k].iloc[tr]))
        oos.append(results[best].iloc[te])
        folds.append(Fold(
            train=(str(index[tr.start].date()), str(index[tr.stop - 1].date())),
            test=(str(index[te.start].date()), str(index[te.stop - 1].date())),
            chosen=best, train_sharpe=round(sharpe(results[best].iloc[tr]), 2),
            test_sharpe=round(sharpe(results[best].iloc[te]), 2)))
        start += test_days
    returns = pd.concat(oos) if oos else pd.Series(dtype=float)
    return WalkForward(returns=returns, folds=folds, n_trials=len(results),
                       trial_sharpes={k: sharpe(r) for k, r in results.items()})


def walk_forward(closes: pd.DataFrame, grid: list[Config] | None = None, *, train_days: int = TRAIN_DAYS,
                 test_days: int = TEST_DAYS) -> WalkForward:
    grid = grid or default_grid()
    cache_scores: dict[str, pd.DataFrame] = {}
    results = {}
    for cfg in grid:
        sc = cache_scores.setdefault(cfg.signal, scores_for(closes, cfg.signal))
        results[cfg.label()] = long_short_returns(closes, cfg, sc)["net"]
    return walk_forward_select(results, train_days=train_days, test_days=test_days)


def _beta_alpha(strategy: pd.Series, market: pd.Series) -> tuple[float, float]:
    both = pd.concat([strategy, market], axis=1).dropna()
    if len(both) < 30 or both.iloc[:, 1].var() == 0:
        return math.nan, math.nan
    beta = both.cov().iloc[0, 1] / both.iloc[:, 1].var()
    alpha = (both.iloc[:, 0].mean() - beta * both.iloc[:, 1].mean()) * 252
    return float(beta), float(alpha)


def stats_row(r: pd.Series, market: pd.Series | None = None) -> dict:
    r = r.dropna()
    row = {"days": len(r), "sharpe": round(sharpe(r), 2), "cagr_pct": round(cagr(r) * 100, 2),
           "max_drawdown_pct": round(max_drawdown(r) * 100, 2),
           "ann_vol_pct": round(float(r.std(ddof=1)) * math.sqrt(252) * 100, 2) if len(r) > 1 else None}
    if market is not None:
        beta, alpha = _beta_alpha(r, market)
        row["beta_to_universe"] = None if math.isnan(beta) else round(beta, 3)
        row["alpha_pct"] = None if math.isnan(alpha) else round(alpha * 100, 2)
    return row


def research_report(closes: pd.DataFrame, grid: list[Config] | None = None) -> dict:
    """Everything a reviewer would ask for: in-sample table, walk-forward OOS, luck-adjusted significance."""
    grid = grid or default_grid()
    market = closes.pct_change(fill_method=None).mean(axis=1)  # equal-weight universe, the free alternative
    wf = walk_forward(closes, grid)
    oos = wf.returns
    var = float(np.var(list(wf.trial_sharpes.values()))) / 252 if wf.trial_sharpes else None  # per-period units
    lo, hi = block_bootstrap_sharpe(oos)
    singles = {}
    for sig in ["momentum", "reversal", "low_vol", "composite"]:
        cfg = Config(signal=sig)
        singles[sig] = stats_row(long_short_returns(closes, cfg)["net"], market)
    out = {
        "universe": {"symbols": closes.shape[1], "start": str(closes.index[0].date()), "end": str(closes.index[-1].date())},
        "trials": wf.n_trials,
        "in_sample_default_config": singles,
        "walk_forward_oos": stats_row(oos, market.reindex(oos.index)) if len(oos) else {},
        "oos_sharpe_95ci": [None if math.isnan(lo) else round(lo, 2), None if math.isnan(hi) else round(hi, 2)],
        "probabilistic_sharpe_vs_0": round(probabilistic_sharpe(oos), 3) if len(oos) else None,
        "deflated_sharpe": round(deflated_sharpe(oos, wf.n_trials, var), 3) if len(oos) else None,
        "equal_weight_universe": stats_row(market.reindex(oos.index).dropna()) if len(oos) else {},
        "folds": [f.__dict__ for f in wf.folds],
        "costs": {"cost_bps_per_turnover": Config().cost_bps, "borrow_bps_annual": Config().borrow_bps_annual},
        "caveats": [
            "universe is today's constituents (survivorship bias); the book is dollar-neutral, which cancels most but not all of it",
            "daily closes only: no intraday timing, queue position or market impact",
            "deflated Sharpe counts only the grid searched here, not every idea tried before it was written down",
        ],
    }
    return out
