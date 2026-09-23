"""In-memory broker for tests and offline dry runs.

Fills marketable LIMIT orders at the limit price, triggers STOP_LOSS orders
when the last price trades through the stop, and can inject failures.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..schemas import AccountState, Bar, BrokerOrder, BrokerOrderStatus, Position, Quote, utcnow
from .base import BrokerError, OrderRequest, PreviewResult


class SimulatedBroker:
    name = "simulated"

    def __init__(self, cash: float = 100_000.0, account_id: str = "SIM"):
        self.account_id = account_id
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.quotes: dict[str, Quote] = {}
        self.bars: dict[str, list[Bar]] = {}
        self.fail_next_place: str | None = None  # "reject" | "timeout_received" | "timeout_lost"
        self.place_calls = 0

    # -- test helpers -------------------------------------------------------------

    def set_quote(self, symbol: str, bid: float, ask: float, last: float | None = None, volume: float = 1e6,
                  fetched_at: datetime | None = None) -> None:
        self.quotes[symbol] = Quote(symbol=symbol, bid=bid, ask=ask, last=last or (bid + ask) / 2, volume=volume,
                                    fetched_at=fetched_at or utcnow())
        pos = self.positions.get(symbol)
        if pos:
            self.positions[symbol] = pos.model_copy(update={"last_price": self.quotes[symbol].last})
        self._match(symbol)

    def set_trend_bars(self, symbol: str, start: float, daily_pct: float, n: int = 260, volume: float = 5e6,
                       end: datetime | None = None) -> None:
        end = end or utcnow()
        bars, px = [], start
        for i in range(n):
            ts = end - timedelta(days=n - 1 - i)
            nxt = px * (1 + daily_pct / 100)
            bars.append(Bar(ts=ts, open=px, high=max(px, nxt) * 1.01, low=min(px, nxt) * 0.99, close=nxt, volume=volume))
            px = nxt
        self.bars[symbol] = bars

    # -- Broker protocol ----------------------------------------------------------

    def get_account(self) -> AccountState:
        mv = sum(p.market_value for p in self.positions.values())
        open_orders = [o for o in self.orders.values()
                       if o.status in (BrokerOrderStatus.WORKING, BrokerOrderStatus.PARTIALLY_FILLED)]
        return AccountState(account_id=self.account_id, equity=self.cash + mv, cash=self.cash,
                            buying_power=self.cash, positions=list(self.positions.values()),
                            open_orders=open_orders, as_of=utcnow())

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    def get_daily_bars(self, symbols: list[str], count: int) -> dict[str, list[Bar]]:
        return {s: self.bars[s][-count:] for s in symbols if s in self.bars}

    def preview(self, order: OrderRequest) -> PreviewResult:
        px = order.limit_price or order.stop_price or self.quotes[order.symbol].ask
        return PreviewResult(ok=True, estimated_cost=px * order.quantity, estimated_fees=0.0)

    def place(self, order: OrderRequest) -> str | None:
        self.place_calls += 1
        if order.client_order_id in self.orders:
            raise BrokerError(f"duplicate client_order_id {order.client_order_id}")
        mode, self.fail_next_place = self.fail_next_place, None
        if mode == "reject":
            raise BrokerError("simulated rejection")
        if mode == "timeout_lost":
            raise BrokerError("simulated timeout (order never arrived)", ambiguous=True)
        self.orders[order.client_order_id] = BrokerOrder(
            client_order_id=order.client_order_id, broker_order_id=f"B{len(self.orders) + 1}", symbol=order.symbol,
            side=order.side, order_type=order.order_type, quantity=order.quantity, limit_price=order.limit_price,
            stop_price=order.stop_price, status=BrokerOrderStatus.WORKING, raw_status="SUBMITTED",
        )
        self._match(order.symbol)
        if mode == "timeout_received":
            raise BrokerError("simulated timeout (order did arrive)", ambiguous=True)
        return self.orders[order.client_order_id].broker_order_id

    def cancel(self, client_order_id: str) -> None:
        o = self.orders.get(client_order_id)
        if o and o.status == BrokerOrderStatus.WORKING:
            self.orders[client_order_id] = o.model_copy(update={"status": BrokerOrderStatus.CANCELLED})

    def get_order(self, client_order_id: str) -> BrokerOrder:
        o = self.orders.get(client_order_id)
        if o is None:
            return BrokerOrder(client_order_id=client_order_id, symbol="", side="BUY", order_type="",
                               quantity=0, status=BrokerOrderStatus.NOT_FOUND)
        return o

    # -- matching -----------------------------------------------------------------

    def _match(self, symbol: str) -> None:
        q = self.quotes.get(symbol)
        if not q:
            return
        for coid, o in list(self.orders.items()):
            if o.symbol != symbol or o.status != BrokerOrderStatus.WORKING:
                continue
            fill_px = None
            if o.order_type == "LIMIT" and o.side == "BUY" and q.ask <= (o.limit_price or 0):
                fill_px = o.limit_price
            elif o.order_type == "LIMIT" and o.side == "SELL" and q.bid >= (o.limit_price or 1e18):
                fill_px = o.limit_price
            elif o.order_type == "STOP_LOSS" and o.side == "SELL" and q.last <= (o.stop_price or 0):
                fill_px = q.bid
            elif o.order_type == "MARKET":
                fill_px = q.ask if o.side == "BUY" else q.bid
            if fill_px is not None:
                self._fill(coid, fill_px)

    def _fill(self, coid: str, px: float) -> None:
        o = self.orders[coid]
        sign = 1 if o.side == "BUY" else -1
        self.cash -= sign * px * o.quantity
        pos = self.positions.get(o.symbol)
        qty = (pos.quantity if pos else 0) + sign * o.quantity
        if qty == 0:
            self.positions.pop(o.symbol, None)
        else:
            avg = px if not pos or sign < 0 else (pos.avg_cost * pos.quantity + px * o.quantity) / qty
            self.positions[o.symbol] = Position(symbol=o.symbol, quantity=qty, avg_cost=avg,
                                                last_price=self.quotes[o.symbol].last)
        self.orders[coid] = o.model_copy(update={"status": BrokerOrderStatus.FILLED, "filled_quantity": o.quantity,
                                                 "filled_price": px})
