"""Order execution with idempotency, preview checks and ambiguity handling.

State is persisted *before* each broker call. A timeout never triggers a blind
retry: the order goes to UNKNOWN_RECONCILE and is resolved by querying Webull.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta

from .broker.base import Broker, BrokerError, OrderRequest
from .config import RiskLimits
from .ledger import Ledger
from .market_calendar import add_trading_days, to_trading_date
from .schemas import BrokerOrderStatus, OrderState, Proposal, TERMINAL_STATES, utcnow

UNKNOWN_GRACE = timedelta(minutes=10)

BROKER_TO_STATE = {
    BrokerOrderStatus.WORKING: OrderState.ACKNOWLEDGED,
    BrokerOrderStatus.PARTIALLY_FILLED: OrderState.PARTIALLY_FILLED,
    BrokerOrderStatus.FILLED: OrderState.FILLED,
    BrokerOrderStatus.CANCELLED: OrderState.CANCELLED,
    BrokerOrderStatus.REJECTED: OrderState.REJECTED,
}


def client_order_id(decision_id: str, purpose: str) -> str:
    """Deterministic, so a re-run of the same decision can never create a second order."""
    return hashlib.sha256(f"{decision_id}:{purpose}".encode()).hexdigest()[:32]


class ExecutionEngine:
    def __init__(self, broker: Broker, ledger: Ledger, limits: RiskLimits, send_orders: bool):
        self.broker = broker
        self.ledger = ledger
        self.limits = limits
        self.send_orders = send_orders  # False => shadow mode

    # -- submission -----------------------------------------------------------------

    def submit(self, req: OrderRequest, decision_id: str, purpose: str) -> OrderState:
        existing = self.ledger.get_order(req.client_order_id)
        if existing is not None and existing["state"] not in (OrderState.PROPOSED.value, OrderState.RISK_APPROVED.value):
            return OrderState(existing["state"])
        self.ledger.upsert_order(
            client_order_id=req.client_order_id, decision_id=decision_id, purpose=purpose, symbol=req.symbol,
            side=req.side, order_type=req.order_type, quantity=req.quantity, limit_price=req.limit_price,
            stop_price=req.stop_price, time_in_force=req.time_in_force, state=OrderState.RISK_APPROVED,
        )
        if not self.send_orders:
            self.ledger.update_order(req.client_order_id, OrderState.SHADOW)
            return OrderState.SHADOW

        preview = self.broker.preview(req)
        self.ledger.append("order_preview", {"ok": preview.ok, "cost": preview.estimated_cost,
                                             "fees": preview.estimated_fees, "error": preview.error, "raw": preview.raw},
                           decision_id=decision_id, client_order_id=req.client_order_id)
        if not preview.ok:
            self.ledger.update_order(req.client_order_id, OrderState.REJECTED, reason=f"preview failed: {preview.error}")
            return OrderState.REJECTED
        mismatch = self._preview_mismatch(req, preview.estimated_cost)
        if mismatch:
            self.ledger.update_order(req.client_order_id, OrderState.REJECTED, reason=mismatch)
            return OrderState.REJECTED

        self.ledger.update_order(req.client_order_id, OrderState.SUBMITTING)
        try:
            broker_id = self.broker.place(req)
        except BrokerError as e:
            if not e.ambiguous:
                self.ledger.update_order(req.client_order_id, OrderState.REJECTED, reason=str(e))
                return OrderState.REJECTED
            self.ledger.update_order(req.client_order_id, OrderState.UNKNOWN_RECONCILE, reason=str(e))
            return self.sync(req.client_order_id)
        self.ledger.update_order(req.client_order_id, OrderState.ACKNOWLEDGED, broker_order_id=broker_id)
        return self.sync(req.client_order_id)

    def _preview_mismatch(self, req: OrderRequest, cost: float | None) -> str:
        if cost is None or req.limit_price is None:
            return ""
        expected = req.limit_price * req.quantity
        tolerance = 0.03 * expected + self.limits.execution.fee_per_order_usd + 1
        return "" if abs(cost - expected) <= tolerance else f"preview cost {cost} differs from intended {expected:.2f}"

    def cancel(self, coid: str) -> None:
        row = self.ledger.get_order(coid)
        if row is None or OrderState(row["state"]) in TERMINAL_STATES:
            return
        self.ledger.update_order(coid, OrderState.CANCEL_PENDING)
        if self.send_orders:
            try:
                self.broker.cancel(coid)
            except BrokerError as e:
                self.ledger.append("cancel_error", {"error": str(e)}, client_order_id=coid)
            self.sync(coid)
        else:
            self.ledger.update_order(coid, OrderState.CANCELLED)

    # -- state sync -----------------------------------------------------------------

    def sync(self, coid: str) -> OrderState:
        row = self.ledger.get_order(coid)
        if row is None:
            raise KeyError(coid)
        current = OrderState(row["state"])
        if current in TERMINAL_STATES:
            return current
        try:
            bo = self.broker.get_order(coid)
        except Exception as e:
            err_str = (str(e) + " " + getattr(e, "error_code", "")).upper()
            if "TOO_MANY_REQUESTS" in err_str or "TOOMANYREQUESTS" in err_str or "429" in err_str:
                return current
            raise
        if bo.status == BrokerOrderStatus.NOT_FOUND:
            age = utcnow() - datetime.fromisoformat(row["updated_at"])
            if current in (OrderState.UNKNOWN_RECONCILE, OrderState.SUBMITTING) and age > UNKNOWN_GRACE:
                self.ledger.update_order(coid, OrderState.REJECTED, reason="broker has no record after grace period")
                return OrderState.REJECTED
            if current == OrderState.SUBMITTING:
                self.ledger.update_order(coid, OrderState.UNKNOWN_RECONCILE, reason="not found at broker yet")
                return OrderState.UNKNOWN_RECONCILE
            return current
        new = BROKER_TO_STATE[bo.status]
        if current == OrderState.CANCEL_PENDING and new == OrderState.ACKNOWLEDGED:
            new = OrderState.CANCEL_PENDING
        if new != current or bo.filled_quantity != row["filled_quantity"]:
            self.ledger.update_order(coid, new, broker_order_id=bo.broker_order_id,
                                     filled_quantity=bo.filled_quantity, filled_price=bo.filled_price)
        if new in TERMINAL_STATES and bo.filled_quantity > 0:
            self._on_fill(row, bo.filled_quantity, bo.filled_price or row["limit_price"] or 0.0)
        return new

    def _on_fill(self, row, qty: float, px: float) -> None:
        decision_id = row["decision_id"]
        if row["purpose"] == "ENTRY":
            pa = self.ledger.db.execute("SELECT proposal FROM pending_actions WHERE decision_id=?", (decision_id,)).fetchone()
            if pa is None:
                return
            prop = Proposal.model_validate_json(pa["proposal"]).trade
            today = to_trading_date(utcnow())
            self.ledger.open_trade(decision_id, row["symbol"], qty, px, prop.exit.stop_loss, prop.exit.take_profit,
                                   add_trading_days(today, prop.exit.time_stop_days))
            if self.limits.execution.broker_side_stops:
                self.ensure_protective_stop(decision_id, row["symbol"], int(qty), prop.exit.stop_loss)
        elif row["purpose"] in ("STOP", "EXIT"):
            trade = next((t for t in self.ledger.open_trades() if t["decision_id"] == decision_id), None)
            if trade is None:
                return
            if qty + 1e-9 < trade["quantity"]:
                # Partial exit: keep managing (and protecting) the shares still held.
                self.ledger.reduce_trade(decision_id, trade["quantity"] - qty)
            else:
                self.ledger.close_trade(decision_id, "stop_filled" if row["purpose"] == "STOP" else "exit_filled")

    # -- protective stops -----------------------------------------------------------

    MAX_STOP_ORDERS_PER_TRADE = 10

    def stop_orders(self, decision_id: str) -> list:
        return list(self.ledger.iter_rows(
            "SELECT * FROM orders WHERE decision_id=? AND purpose='STOP' ORDER BY created_at, rowid", (decision_id,)))

    def active_stop(self, decision_id: str):
        """The most recent protective stop for a trade, whatever its state."""
        rows = self.stop_orders(decision_id)
        return rows[-1] if rows else None

    def place_protective_stop(self, decision_id: str, symbol: str, qty: int, stop: float) -> OrderState:
        n = len(self.stop_orders(decision_id))
        purpose_key = "STOP" if n == 0 else f"STOP:{n + 1}"  # first stop keeps its historical id
        req = OrderRequest(client_order_id=client_order_id(decision_id, purpose_key), symbol=symbol, side="SELL",
                           order_type="STOP_LOSS", quantity=qty, time_in_force="GTC", stop_price=round(stop, 2))
        state = self.submit(req, decision_id, "STOP")
        if state not in (OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.SHADOW):
            self.ledger.append("unprotected_position", {"symbol": symbol, "qty": qty, "stop": stop,
                                                        "stop_state": state.value}, decision_id=decision_id)
            try:
                from .alerts import alert_unprotected
                alert_unprotected(symbol, qty, stop, state.value)
            except Exception:
                pass
        return state

    def ensure_protective_stop(self, decision_id: str, symbol: str, qty: int, stop: float) -> OrderState | None:
        """Place a broker-side stop if the trade has none working. Bounded so a broker that keeps
        rejecting cannot turn this into an order storm."""
        if qty < 1:
            return None
        row = self.active_stop(decision_id)
        if row is not None:
            state = OrderState(row["state"])
            if state not in (OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED):
                return state  # working, pending, unknown (being reconciled), filled or shadow
        terminal = [st.value for st in TERMINAL_STATES]
        live_exit = self.ledger.db.execute(
            f"SELECT 1 FROM orders WHERE decision_id=? AND purpose='EXIT' AND state NOT IN ({','.join('?' * len(terminal))}) LIMIT 1",
            (decision_id, *terminal)).fetchone()
        if live_exit is not None:
            return None  # a sell is already working for these shares; a second one could oversell
        if len(self.stop_orders(decision_id)) >= self.MAX_STOP_ORDERS_PER_TRADE:
            return None
        return self.place_protective_stop(decision_id, symbol, qty, stop)

    # -- exits ----------------------------------------------------------------------

    def _exit_slot(self, decision_id: str, today: date):
        """Return (client_order_id, existing_row) for today's current exit attempt.

        existing_row is a live or filled order to report on, or None when a fresh attempt may be sent.
        Returns (None, None) when today's attempts are used up."""
        for n in range(1, self.limits.execution.max_exit_attempts_per_day + 1):
            key = f"EXIT:{today.isoformat()}" + ("" if n == 1 else f":{n}")
            coid = client_order_id(decision_id, key)
            row = self.ledger.get_order(coid)
            if row is None:
                return coid, None
            state = OrderState(row["state"])
            if state not in (OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED):
                return coid, row
        return None, None

    def exit_trade(self, decision_id: str, symbol: str, qty: int, ref_price: float, reason: str, today: date) -> OrderState:
        """Cancel the protective stop, then sell with a marketable limit.

        A few attempts per trading day are allowed; the stop is only touched when an attempt will
        actually be made, and `ensure_protective_stop` re-arms it if the sell does not go through."""
        if qty < 1 or ref_price <= 0:
            self.ledger.append("exit_skipped", {"reason": reason, "qty": qty, "ref_price": ref_price,
                                                "why": "no sellable quantity or no usable price"}, decision_id=decision_id)
            return OrderState.REJECTED
        coid, existing = self._exit_slot(decision_id, today)
        if existing is not None:
            return OrderState(existing["state"])
        if coid is None:
            self.ledger.append("exit_attempts_exhausted", {"reason": reason, "date": today.isoformat()},
                               decision_id=decision_id)
            return OrderState.REJECTED
        stop_row = self.active_stop(decision_id)
        if stop_row is not None and OrderState(stop_row["state"]) not in TERMINAL_STATES:
            self.cancel(stop_row["client_order_id"])
            state = OrderState(self.ledger.get_order(stop_row["client_order_id"])["state"])
            if state == OrderState.FILLED:
                return state  # the stop already closed the position
            if state not in TERMINAL_STATES:
                self.ledger.append("exit_deferred", {"reason": "stop cancel not confirmed"}, decision_id=decision_id)
                return state
        elif stop_row is not None and OrderState(stop_row["state"]) == OrderState.FILLED:
            return OrderState.FILLED
        limit = round(ref_price * (1 - self.limits.execution.price_collar_pct / 100), 2)
        req = OrderRequest(client_order_id=coid, symbol=symbol, side="SELL", order_type="LIMIT", quantity=qty,
                           time_in_force="DAY", limit_price=limit)
        self.ledger.append("exit_requested", {"reason": reason, "limit": limit}, decision_id=decision_id)
        return self.submit(req, decision_id, "EXIT")
