"""Paper broker: real market data, simulated fills, state saved to disk between ticks.

Wraps any data source (normally WebullBroker) for quotes, bars and earnings, and fills orders with
the SimulatedBroker rules. Each `tick` is a new process, so cash, positions and orders are saved
to a JSON file after every change. DAY orders left from an earlier trading day are cancelled on load.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..market_calendar import to_trading_date
from ..schemas import BrokerOrder, BrokerOrderStatus, Position, Quote, utcnow
from .base import OrderRequest
from .simulated import SimulatedBroker

STARTING_CASH = 1500.0  # matches the backtest default; delete the state file to restart


class PaperBroker(SimulatedBroker):
    name = "paper"

    def __init__(self, data, path: Path, cash: float = STARTING_CASH):
        super().__init__(cash=cash, account_id="PAPER")
        self.data, self.path = data, Path(path)
        self.day_orders: dict[str, str] = {}  # client_order_id -> trading date it was placed
        if self.path.exists():
            s = json.loads(self.path.read_text())
            self.cash = s["cash"]
            self.positions = {p["symbol"]: Position.model_validate(p) for p in s["positions"]}
            self.orders = {o["client_order_id"]: BrokerOrder.model_validate(o) for o in s["orders"]}
            self.day_orders = s.get("day_orders", {})
        today = to_trading_date(utcnow()).isoformat()
        for coid, placed in list(self.day_orders.items()):
            if placed < today:
                self.cancel(coid)
                del self.day_orders[coid]
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "cash": self.cash,
            "positions": [p.model_dump(mode="json") for p in self.positions.values()],
            "orders": [o.model_dump(mode="json") for o in self.orders.values()],
            "day_orders": self.day_orders,
        }, indent=1))

    # -- market data comes from the real source; every fresh quote can trigger fills ----------

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        quotes = self.data.get_quotes(symbols)
        for sym, q in quotes.items():
            self.quotes[sym] = q
            if pos := self.positions.get(sym):
                self.positions[sym] = pos.model_copy(update={"last_price": q.last})
            self._match(sym)
        self._save()
        return quotes

    def get_daily_bars(self, symbols: list[str], count: int):
        return self.data.get_daily_bars(symbols, count)

    def get_earnings_dates(self, symbols: list[str], today: date) -> dict[str, date]:
        return self.data.get_earnings_dates(symbols, today)

    # -- order calls persist ------------------------------------------------------------

    def place(self, order: OrderRequest) -> str | None:
        if order.symbol not in self.quotes:
            self.get_quotes([order.symbol])
        try:
            return super().place(order)
        finally:
            o = self.orders.get(order.client_order_id)
            if order.time_in_force == "DAY" and o and o.status == BrokerOrderStatus.WORKING:
                self.day_orders[order.client_order_id] = to_trading_date(utcnow()).isoformat()
            self._save()

    def cancel(self, client_order_id: str) -> None:
        super().cancel(client_order_id)
        self._save()
