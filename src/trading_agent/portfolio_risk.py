"""Portfolio risk analytics for the advisor: the standard quant-desk risk view, applied to the live holdings.

Risk control, not alpha. The repo's own studies found no tradable edge in systematic signals, so momentum and
volatility are reported as information about each holding, never as buy/sell triggers.

- covariance: Ledoit-Wolf shrinkage (toward no correlation, on standardised returns) so 12 names over 2 years is not
  over-fitted by the raw sample covariance
- risk contribution: each position's share of portfolio variance (w_i * (S w)_i / w'Sw), which is not its weight
- PCA: how much of the risk is one common factor, and the effective number of independent bets
- volatility target: the smallest core weight that keeps annual volatility within TARGET_VOL_MULT x the index's
- replay: the Jan-Oct 2022 selloff applied to today's weights
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .math_quant import portfolio_volatility

TARGET_VOL_MULT = 1.3      # target = this multiple of the index's own volatility (a fixed 15% is below SPY's ~17%)
LOOKBACK_START = "2021-12-01"
RECENT_START = "2024-10-01"  # covariance window: about two years of daily returns
CORE_PROXY = "SPY"           # the core fund (SPYM) has too little history; SPY tracks the same index


def lw_cov(r: np.ndarray) -> np.ndarray:
    """Annualised Ledoit-Wolf covariance. r is T x N daily returns. Shrinks the correlation matrix toward identity
    with the Ledoit-Wolf intensity, then restores each asset's own volatility."""
    T, N = r.shape
    s = r.std(axis=0, ddof=0)
    z = (r - r.mean(axis=0)) / s
    S = z.T @ z / T
    d2 = float(((S - np.eye(N)) ** 2).sum()) / N
    b2 = min(float(np.mean([((np.outer(zt, zt) - S) ** 2).sum() for zt in z])) / N / T, d2)
    k = b2 / d2 if d2 else 0.0
    C = (1 - k) * S + k * np.eye(N)
    return np.outer(s, s) * C * 252


def analyze(weights: dict[str, float], px: pd.DataFrame, target_vol: float | None = None) -> dict[str, Any]:
    """weights: symbol -> fraction of equity (cash is the unallocated remainder). px: daily closes incl. SPY."""
    w: dict[str, float] = {}
    for sym, x in weights.items():  # merge the core fund into its proxy
        key = CORE_PROXY if sym in ("SPYM", "SPY") else sym
        if key in px.columns:
            w[key] = w.get(key, 0.0) + x
    syms = list(w)
    all_syms = sorted(set(syms) | {CORE_PROXY})
    r = px[all_syms].pct_change().loc[RECENT_START:].dropna()
    cov = lw_cov(r.to_numpy())
    idx = {s: i for i, s in enumerate(all_syms)}
    index_vol = float(np.sqrt(cov[idx[CORE_PROXY], idx[CORE_PROXY]]))
    target_vol = target_vol or TARGET_VOL_MULT * index_vol
    vec = np.zeros(len(all_syms))
    for s, x in w.items():
        vec[idx[s]] = x
    var = float(vec @ cov @ vec)
    vol = portfolio_volatility(vec, cov)
    port = r.to_numpy() @ vec
    spy = r[CORE_PROXY].to_numpy()
    beta = float(np.cov(port, spy)[0, 1] / spy.var(ddof=1))
    contrib = vec * (cov @ vec) / var if var else vec
    top = sorted(((s, float(contrib[idx[s]]) * 100) for s in syms), key=lambda kv: -kv[1])[:4]
    # PCA of the risk-weighted covariance: PC1 share and the effective number of independent bets
    ev = np.sort(np.linalg.eigvalsh(np.diag(vec) @ cov @ np.diag(vec)))[::-1]
    ev = np.clip(ev, 0, None)
    pc1 = float(ev[0] / ev.sum()) * 100 if ev.sum() else 0.0
    bets = float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() else 0.0
    # smallest core weight that meets the volatility target, satellites scaled pro-rata
    sat = vec.copy()
    sat[idx[CORE_PROXY]] = 0.0
    sat_total = float(sat.sum())
    core_for_target = None
    for c in np.arange(0.0, 1.0001, 0.01):
        v = sat / sat_total * (1 - c) if sat_total else sat.copy()
        v[idx[CORE_PROXY]] = c
        if portfolio_volatility(v, cov) <= target_vol:
            core_for_target = float(c) * 100
            break
    # 2022 replay on today's weights; names with no 2022 history take twice SPY's move (flagged)
    p22 = px.loc["2022-01-03":"2022-10-12"]
    spy22 = float(p22[CORE_PROXY].iloc[-1] / p22[CORE_PROXY].iloc[0] - 1)
    replay, proxied = 0.0, []
    for s, x in w.items():
        col = p22[s].dropna() if s in p22 else pd.Series(dtype=float)
        if len(col) < 100:
            proxied.append(s)
            replay += x * 2 * spy22
        else:
            replay += x * float(col.iloc[-1] / col.iloc[0] - 1)
    per = {}
    for s in syms:  # information only, never a trigger
        col = px[s].dropna()
        info: dict[str, float] = {"vol_1y_pct": round(float(col.pct_change().iloc[-252:].std() * np.sqrt(252) * 100), 0)}
        if len(col) > 253:
            info["mom_12_1_pct"] = round(float((col.iloc[-22] / col.iloc[-253] - 1) * 100), 0)
        per[s] = info
    return {"vol_pct": round(vol * 100, 1), "beta": round(beta, 2), "replay_2022_pct": round(replay * 100, 0),
            "replay_proxied": proxied, "top_risk": [(s, round(c, 0)) for s, c in top], "pc1_pct": round(pc1, 0),
            "effective_bets": round(bets, 1), "core_for_target_vol_pct": core_for_target, "target_vol_pct": round(target_vol * 100, 1),
            "index_vol_pct": round(index_vol * 100, 1), "per_symbol": per}


def fetch_prices(symbols: list[str]) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            px = yf.download(sorted(set(symbols) | {CORE_PROXY}), start=LOOKBACK_START, auto_adjust=True,
                             progress=False)["Close"]
        return px.dropna(how="all")
    except Exception:
        return None


def render_risk(risk: dict[str, Any]) -> str:
    core = risk["core_for_target_vol_pct"]
    need = f"{core:.0f}% core" if core is not None else "not reachable"
    lines = ["", "RISK (2y daily returns, shrunk covariance)",
             f"- Volatility {risk['vol_pct']:.0f}%/yr, beta {risk['beta']:.2f}; 2022-style replay {risk['replay_2022_pct']:+.0f}%"
             + (f" ({', '.join(risk['replay_proxied'])} proxied at 2x SPY)" if risk["replay_proxied"] else ""),
             "- Biggest risk contributors: " + ", ".join(f"{s} {c:.0f}%" for s, c in risk["top_risk"]),
             f"- One common factor drives {risk['pc1_pct']:.0f}% of the risk; about {risk['effective_bets']:.1f} independent bets",
             f"- Index volatility is {risk['index_vol_pct']:.0f}%; to stay within {TARGET_VOL_MULT}x of it "
             f"({risk['target_vol_pct']:.0f}%) with these satellites you need {need}"]
    return "\n".join(lines)


def analyze_account(account: Any) -> dict[str, Any] | None:
    """Live entry point: weights from the account, prices from yfinance. None if the data is unavailable."""
    eq = account.equity or 1.0
    weights = {p.symbol: p.market_value / eq for p in account.positions}
    px = fetch_prices(list(weights))
    if px is None or px.empty:
        return None
    try:
        return analyze(weights, px)
    except Exception:
        return None
