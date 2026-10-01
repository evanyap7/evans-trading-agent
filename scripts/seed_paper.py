"""Reset the paper account to mirror the part of the real Webull account the agent can trade.

    1. Stop the paper job:   launchctl unload ~/Library/LaunchAgents/com.trading-agent.paper.plist
    2. Run this:             uv run python scripts/seed_paper.py
    3. Start it again:       launchctl load ~/Library/LaunchAgents/com.trading-agent.paper.plist

The old paper state (ledger and broker book) is moved aside, never deleted: a ledger that remembers sweep
buys the new book does not hold would fail reconciliation and engage the kill switch, so both are reset
together. Edit the numbers below to match your account before running.

The mutual fund in the real account (LU1858852160) is left out on purpose: the agent cannot trade it, and
its price would never update here, which would only add noise to the comparison. Paper equity is therefore
smaller than the real account's.
"""

import json
import shutil
import sys
from datetime import datetime

from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings

CASH_USD = 705.76  # updated with $800 SGD deposit (~$615 USD)
# symbol, quantity, average cost, last price (USD), from the Webull app on 1 Oct 2026
POSITIONS = [
    ("SMCI", 1, 41.50, 41.58),
    ("PLTR", 1, 191.00, 188.82),
    ("XOM", 1, 161.14, 161.40),
]


def main() -> None:
    paper_dir = load_settings().state_dir / "paper"
    if paper_dir.exists():
        archive = paper_dir.with_name(f"paper-old-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.move(str(paper_dir), str(archive))
        print(f"old paper state moved to {archive}")
    paper_dir.mkdir(parents=True)

    positions = [{"symbol": s, "quantity": q, "avg_cost": c, "last_price": p} for s, q, c, p in POSITIONS]
    (paper_dir / "paper_broker.json").write_text(json.dumps(
        {"cash": CASH_USD, "positions": positions, "orders": [], "day_orders": {}}, indent=1))
    # Read by paper_vs_spy.py: the "do nothing" benchmark is these holdings left alone.
    (paper_dir / "seed.json").write_text(json.dumps(
        {"started": datetime.now().date().isoformat(), "cash": CASH_USD,
         "positions": [{"symbol": s, "quantity": q, "price": p} for s, q, _, p in POSITIONS]}, indent=1))

    acct = PaperBroker(None, paper_dir / "paper_broker.json").get_account()  # loads the file: proves it is valid
    print(f"paper account seeded: cash ${acct.cash:,.2f}, equity ${acct.equity:,.2f}")
    for p in acct.positions:
        print(f"  {p.symbol} x{p.quantity:g} @ {p.avg_cost}")
    if acct.equity <= 0:
        sys.exit("seeded equity is not positive; check the numbers")


if __name__ == "__main__":
    main()
