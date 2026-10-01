"""Paper account vs SPY buy-and-hold, and vs doing nothing, since the paper test started.

    uv run python scripts/paper_vs_spy.py

If state/paper/seed.json exists (written by seed_paper.py), the paper account started with holdings the
agent did not choose. Their moves would otherwise be credited to the agent, so the "hold only" line shows
what the starting holdings and cash would be worth with the agent switched off.
"""

import json
import sqlite3
from pathlib import Path

import yfinance as yf

paper_dir = Path(__file__).resolve().parents[1] / "state" / "paper"
db = sqlite3.connect(paper_dir / "ledger.db")
(d0, e0), = db.execute("SELECT trading_date, equity FROM equity_snapshots ORDER BY ts LIMIT 1")
(d1, e1, s1), = db.execute("SELECT trading_date, equity, sweep_pnl FROM equity_snapshots ORDER BY ts DESC LIMIT 1")
closed, = db.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()
spy = yf.Ticker("SPY").history(start=d0, auto_adjust=True)["Close"]
spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
ret = (e1 / e0 - 1) * 100
print(f"{d0} -> {d1}")
print(f"paper account: ${e0:,.2f} -> ${e1:,.2f}  ({ret:+.2f}%)")
print(f"  of which cash sweep: ${s1:+,.2f}, everything else: ${e1 - e0 - s1:+,.2f}")
print(f"closed trades: {closed} (need 30+ before deciding)")
print(f"SPY buy-and-hold:      {spy_ret:+.2f}%")

hold_ret = None
seed_path = paper_dir / "seed.json"
if seed_path.exists():
    seed = json.loads(seed_path.read_text())
    start = seed["cash"] + sum(p["quantity"] * p["price"] for p in seed["positions"])
    now = seed["cash"] + sum(
        p["quantity"] * yf.Ticker(p["symbol"]).history(period="5d")["Close"].iloc[-1] for p in seed["positions"])
    hold_ret = (now / start - 1) * 100
    held = ", ".join(p["symbol"] for p in seed["positions"])
    print(f"hold only ({held} + cash, agent off): {hold_ret:+.2f}%")

print()
print("agent is AHEAD of SPY (before API costs)" if ret > spy_ret else "agent is BEHIND SPY")
if hold_ret is not None:
    print("agent is AHEAD of doing nothing" if ret > hold_ret else "agent is BEHIND doing nothing")
