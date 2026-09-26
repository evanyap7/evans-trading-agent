"""Cash sweep: idle cash parked in one broad ETF, sold again to fund entries.

The sweep owns only the shares its own orders bought (purpose SWEEP in the ledger), so a manual holding in
the same ETF stays a manual position. Those shares are not a trade: they are removed from the positions
the agent, sizing and risk engine see, and their market value counts as spendable cash. The drawdown and
daily-loss breakers subtract the sweep's P&L, so an ordinary market dip does not halt the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import Ledger
from .schemas import AccountState

PURPOSE = "SWEEP"


@dataclass(frozen=True)
class SweepState:
    ledger_qty: float  # bought minus sold, from the sweep's own fills
    qty: int           # of those, what the broker still shows us holding
    net_cost: float    # cash put in minus cash taken out
    last: float

    @property
    def value(self) -> float:
        return self.qty * self.last

    @property
    def pnl(self) -> float:
        """Realized plus unrealized gain of the sweep since it started (0 when it has never traded)."""
        return self.value - self.net_cost


EMPTY = SweepState(ledger_qty=0.0, qty=0, net_cost=0.0, last=0.0)


def sweep_state(ledger: Ledger, account: AccountState, symbol: str) -> SweepState:
    qty = net = 0.0
    for r in ledger.iter_rows(
            "SELECT side, filled_quantity, filled_price, limit_price FROM orders WHERE purpose=? AND filled_quantity>0",
            (PURPOSE,)):
        sign = 1 if r["side"] == "BUY" else -1
        qty += sign * r["filled_quantity"]
        net += sign * r["filled_quantity"] * (r["filled_price"] or r["limit_price"] or 0.0)
    held = account.position(symbol)
    held_qty = held.quantity if held else 0.0
    return SweepState(ledger_qty=qty, qty=int(max(0.0, min(qty, held_qty))), net_cost=net,
                      last=held.last_price if held else 0.0)


def trading_view(account: AccountState, state: SweepState, symbol: str) -> AccountState:
    """The account as the agent, sizing and risk engine see it: sweep shares turned back into cash."""
    if state.qty <= 0:
        return account
    positions = []
    for p in account.positions:
        if p.symbol != symbol:
            positions.append(p)
        elif p.quantity - state.qty > 1e-9:
            positions.append(p.model_copy(update={"quantity": p.quantity - state.qty}))
    return account.model_copy(update={"positions": positions, "cash": account.cash + state.value,
                                      "buying_power": account.buying_power + state.value})
