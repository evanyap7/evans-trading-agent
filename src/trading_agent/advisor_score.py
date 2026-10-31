"""Score the advisor prediction log exactly as docs/advisor-preregistration.md fixes it. Read-only.

Outcome: `up` is correct if the stock's return over the horizon beats SPY's, `down` if it trails. Returns run from the
price logged at call time to the close `horizon_days` trading days after the call date. Only the first call per
(arm, symbol, horizon) per horizon window counts toward n. No pass/fail is reported before MIN_N independent calls.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd

from .ledger import Ledger

MIN_N = 300
BOOT = 5000


def load_calls(ledger: Ledger) -> list[dict]:
    rows = [json.loads(r["payload"]) for r in ledger.events("advisor_call", limit=1_000_000)]
    return sorted(rows, key=lambda c: (c["date"], c["arm"], c["symbol"], c["horizon_days"]))


def settle(call: dict, closes: pd.DataFrame) -> bool | None:
    """True/False once the horizon has fully elapsed, None while it is still open or prices are missing."""
    sym, h = call["symbol"], call["horizon_days"]
    if sym not in closes or "SPY" not in closes:
        return None
    after = closes.loc[closes.index > pd.Timestamp(call["date"]), [sym, "SPY"]].dropna()
    if len(after) < h:
        return None
    end = after.iloc[h - 1]
    beat = end[sym] / call["price"] - 1 > end["SPY"] / call["spy_price"] - 1
    return bool(beat) if call["direction"] == "up" else not beat


def independent(calls: list[dict], closes: pd.DataFrame) -> list[dict]:
    """First call per (arm, symbol, horizon); the next one counts only after the previous one's horizon has run out."""
    kept: list[dict] = []
    busy_until: dict = {}
    for c in calls:  # sorted by date
        key = (c["arm"], c["symbol"], c["horizon_days"])
        if key in busy_until and pd.Timestamp(c["date"]) <= busy_until[key]:
            continue
        after = closes.index[closes.index > pd.Timestamp(c["date"])]
        busy_until[key] = after[c["horizon_days"] - 1] if len(after) >= c["horizon_days"] else pd.Timestamp.max
        kept.append(c)
    return kept


def hit_interval(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a hit rate (the plain normal interval is poor near the edges at small n)."""
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def arm_stats(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["correct"] for r in rows)
    lo, hi = hit_interval(k, n) if n else (0.0, 1.0)
    brier = float(np.mean([(r["p"] - r["correct"]) ** 2 for r in rows])) if n else float("nan")
    return {"n": n, "hit": k / n if n else float("nan"), "lo": lo, "hi": hi, "brier": brier}


def paired_brier_diff(news: list[dict], blind: list[dict], seed: int = 0) -> tuple[float, float, float, int]:
    """Mean (news - blind) Brier on the same (date, symbol, horizon) calls, with a 95% bootstrap interval. Negative
    favours news. Returns (mean, lo, hi, pairs)."""
    key = lambda r: (r["date"], r["symbol"], r["horizon_days"])  # noqa: E731
    b = {key(r): (r["p"] - r["correct"]) ** 2 for r in blind}
    d = np.array([(r["p"] - r["correct"]) ** 2 - b[key(r)] for r in news if key(r) in b])
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    means = np.random.default_rng(seed).choice(d, size=(BOOT, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(d)


def evaluate(calls: list[dict], closes: pd.DataFrame) -> dict:
    """Pure: calls plus a closes table (columns = symbols incl. SPY) -> report dict."""
    scored: dict[str, list[dict]] = {"news": [], "blind": []}
    open_calls = 0
    for c in independent(calls, closes):
        ok = settle(c, closes)
        if ok is None:
            open_calls += 1
        else:
            scored[c["arm"]].append({**c, "correct": int(ok)})
    stats = {arm: arm_stats(rows) for arm, rows in scored.items()}
    diff = paired_brier_diff(scored["news"], scored["blind"])
    s = stats["news"]
    ready = s["n"] >= MIN_N
    passed = ready and s["lo"] > 0.5 and s["brier"] < 0.25 and diff[2] < 0  # upper end below zero: news beats blind
    return {"stats": stats, "brier_diff": diff, "open": open_calls, "ready": ready, "passed": bool(passed)}


def render(rep: dict) -> str:
    out = ["ADVISOR PREDICTION LOG (pre-registered; see docs/advisor-preregistration.md)"]
    for arm in ("news", "blind"):
        s = rep["stats"][arm]
        out.append(f"- {arm}: n={s['n']}, hit {s['hit']:.1%} (95% {s['lo']:.1%}-{s['hi']:.1%}), Brier {s['brier']:.3f}"
                   if s["n"] else f"- {arm}: no settled independent calls yet")
    m, lo, hi, pairs = rep["brier_diff"]
    if pairs >= 2:
        out.append(f"- news minus blind Brier on {pairs} paired calls: {m:+.3f} (95% {lo:+.3f} to {hi:+.3f}; negative favours news)")
    out.append(f"- still open (horizon not elapsed): {rep['open']}")
    n = rep["stats"]["news"]["n"]
    out.append(f"VERDICT: {'PASS' if rep['passed'] else 'FAIL'}" if rep["ready"]
               else f"VERDICT: none yet. Needs {MIN_N} independent news-arm calls; have {n}. Hit rates below that are noise.")
    return "\n".join(out)


def fetch_closes(symbols: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = yf.download(sorted(set(symbols) | {"SPY"}), start=start, auto_adjust=False, progress=False)["Close"]
    return px.dropna(how="all")


def run(ledger: Ledger, printer=print) -> dict | None:
    calls = load_calls(ledger)
    if not calls:
        printer("No advisor calls logged yet. Run `trading-agent advise --force` after the design freeze.")
        return None
    closes = fetch_closes([c["symbol"] for c in calls], min(c["date"] for c in calls))
    rep = evaluate(calls, closes)
    printer(render(rep))
    return rep
