#!/usr/bin/env python3
"""Event study: do open-market insider buys predict excess returns, as seen from the public filing date?

Event = first trading day a symbol's trailing-90-day filed buys reach --min-usd (one event per symbol per
--cooldown days). Entry = NEXT session's open (we only see the filing after the fact). Exit = close N sessions
later. Excess = stock return minus SPY over the same window.

    uv run python scripts/insider_study.py --start 2022-01-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading_agent.backtest import load_bars
from trading_agent.config import CONFIG_DIR, load_universe
from trading_agent.insider import buy_clusters, load_insider_buys

CACHE = ROOT / "state" / "backtest_cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--min-usd", type=float, default=100_000)
    ap.add_argument("--hold", type=int, nargs="+", default=[5, 20, 60])
    ap.add_argument("--cooldown", type=int, default=30)
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)

    buys = load_insider_buys(start - timedelta(days=90), end, ROOT / "state" / "insider")
    print(f"{len(buys)} open-market buy transactions filed up to {end}")

    univ = load_universe(CONFIG_DIR / "universe_top100.yaml")
    syms = sorted(s for s, v in univ.symbols.items() if v.type == "EQUITY")
    bars = load_bars(sorted(set(syms) | {"SPY"}), start, end, CACHE)
    spy = bars["SPY"]
    day_idx = {b.ts.date(): i for i, b in enumerate(spy)}

    by_sym: dict[str, list[dict]] = {}
    for x in buys:
        by_sym.setdefault(x["symbol"], []).append(x)
    def find_events(hit) -> list[tuple[str, date]]:
        out = []
        for sym in syms:
            if sym not in bars:
                continue
            last_event = date.min
            for b in bars[sym]:
                d = b.ts.date()
                if d < start or d > end or (d - last_event).days < a.cooldown:
                    continue
                if hit(sym, d):
                    last_event = d
                    out.append((sym, d))
        return out

    insider = find_events(lambda sym, d: buy_clusters(by_sym.get(sym, []), sym, d)["usd"] >= a.min_usd)
    # control: same symbols, same spacing, no insider condition. Shows drift from survivorship and dips.
    control = find_events(lambda sym, d: True)
    print(f"{len(insider)} insider events, {len(control)} control events, {len(syms)} large-cap stocks "
          f"(>= ${a.min_usd:,.0f} filed in 90d)")

    def study(events, label):
        per_hold: dict[int, list[tuple[int, float]]] = {h: [] for h in a.hold}
        for sym, d in events:
            sb = bars[sym]
            i = [b.ts.date() for b in sb].index(d) + 1  # next session's open
            if i >= len(sb) or sb[i].ts.date() not in day_idx:
                continue
            j = day_idx[sb[i].ts.date()]
            for h in a.hold:
                if i + h >= len(sb) or j + h >= len(spy):
                    continue
                per_hold[h].append((d.year, (sb[i + h].close / sb[i].open - spy[j + h].close / spy[j].open) * 100))
        print(f"\n[{label}] {'hold':>4} {'n':>5} {'mean excess %':>14} {'hit rate':>9} {'t-stat':>7}")
        for h, rows in per_hold.items():
            xs = [x for _, x in rows]
            if len(xs) < 3:
                print(f"{h:>12} {len(xs):>5}  too few events")
                continue
            t = fmean(xs) / (stdev(xs) / math.sqrt(len(xs)))
            print(f"{'':>10}{h:>4} {len(xs):>5} {fmean(xs):>14.2f} {sum(x > 0 for x in xs) / len(xs):>9.0%} {t:>7.2f}")
        h = a.hold[-1]
        print(f"  by year at {h}d: " + "  ".join(
            f"{y}: {fmean([x for yy, x in per_hold[h] if yy == y]):+.1f}% (n={sum(yy == y for yy, _ in per_hold[h])})"
            for y in sorted({yy for yy, _ in per_hold[h]})))

    study(insider, "insider")
    study(control, "control")
    print("\nInsider edge = insider minus control. Events overlap in time, so t-stats are optimistic.")


if __name__ == "__main__":
    main()
