"""Cross-sectional alpha signals built from documented anomalies.

Each signal is a DataFrame (dates x symbols) using only data up to and including the row's date, so a
row can be traded at the next bar with no lookahead. The live agent calls the same `composite` on a
one-row frame built from the evidence the verifier already trusts, so research and production share
one definition.

  momentum   Jegadeesh & Titman (1993): winners keep winning over 3-12 months. The most recent month
             is skipped because of short-term reversal.
  reversal   Jegadeesh (1990), Lehmann (1990): last week's losers bounce, a liquidity-provision effect.
  low_vol    Ang, Hodrick, Xing & Zhang (2006), Frazzini & Pedersen (2014): low-risk names earn more
             per unit of risk than high-risk ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOM_LONG, MOM_SKIP, REV_DAYS, VOL_DAYS = 120, 20, 5, 20
DEFAULT_WEIGHTS = {"momentum": 0.5, "reversal": 0.25, "low_vol": 0.25}


def zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per row, winsorised so one outlier cannot dominate."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def raw_signals(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rets = closes.pct_change(fill_method=None)
    return {
        "momentum": closes.shift(MOM_SKIP) / closes.shift(MOM_LONG) - 1,
        "reversal": -(closes / closes.shift(REV_DAYS) - 1),
        "low_vol": -rets.rolling(VOL_DAYS).std(),
    }


def composite(raw: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    total = sum(abs(v) for v in w.values())
    parts = [zscore(raw[k]) * (v / total) for k, v in w.items() if v]
    out = parts[0]
    for p in parts[1:]:
        out = out.add(p, fill_value=0)
    return out


def raw_from_features(features: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """One-row version of `raw_signals`, from the verifier-trusted price features of each symbol.

    momentum = ret_120d - ret_20d matches closes[-21]/closes[-121] - 1 up to compounding, and the
    volatility feature is the same 20-day daily-return deviation annualised, so the ranking is identical.
    """
    syms = [s for s, f in features.items()
            if all(f.get(k) is not None for k in ("ret_120d_pct", "ret_20d_pct", "ret_5d_pct", "realized_vol_20d_ann_pct"))]
    idx = [0]

    def frame(fn):
        return pd.DataFrame([[fn(features[s]) for s in syms]], columns=syms, index=idx, dtype=float)

    return {
        "momentum": frame(lambda f: (f["ret_120d_pct"] - f["ret_20d_pct"]) / 100),
        "reversal": frame(lambda f: -f["ret_5d_pct"] / 100),
        "low_vol": frame(lambda f: -f["realized_vol_20d_ann_pct"] / 100),
    }
