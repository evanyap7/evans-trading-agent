"""Paper account vs SPY buy-and-hold since the paper test started.

    uv run python scripts/paper_vs_spy.py
"""

import sqlite3
from pathlib import Path

import yfinance as yf

db = sqlite3.connect(Path(__file__).resolve().parents[1] / "state" / "paper" / "ledger.db")
(d0, e0), = db.execute("SELECT trading_date, equity FROM equity_snapshots ORDER BY ts LIMIT 1")
(d1, e1, s1), = db.execute("SELECT trading_date, equity, sweep_pnl FROM equity_snapshots ORDER BY ts DESC LIMIT 1")
closed, = db.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()
spy = yf.Ticker("SPY").history(start=d0, auto_adjust=True)["Close"]
spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
ret = (e1 / e0 - 1) * 100
print(f"{d0} -> {d1}")
print(f"paper account: ${e0:,.2f} -> ${e1:,.2f}  ({ret:+.2f}%)")
print(f"  of which cash sweep: ${s1:+,.2f}, agent trades: ${e1 - e0 - s1:+,.2f}")
print(f"closed trades: {closed} (need 30+ before deciding)")
print(f"SPY buy-and-hold:      {spy_ret:+.2f}%")
print("agent is AHEAD of SPY (before API costs)" if ret > spy_ret else "agent is BEHIND SPY")
