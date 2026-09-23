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
        bo = self.broker.get_order(coid)
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
                self.place_protective_stop(decision_id, row["symbol"], int(qty), prop.exit.stop_loss)
        elif row["purpose"] in ("STOP", "EXIT"):
            open_ids = {t["decision_id"] for t in self.ledger.open_trades()}
            if decision_id in open_ids:
                self.ledger.close_trade(decision_id, "stop_filled" if row["purpose"] == "STOP" else "exit_filled")

    def place_protective_stop(self, decision_id: str, symbol: str, qty: int, stop: float) -> OrderState:
        req = OrderRequest(client_order_id=client_order_id(decision_id, "STOP"), symbol=symbol, side="SELL",
                           order_type="STOP_LOSS", quantity=qty, time_in_force="GTC", stop_price=round(stop, 2))
        return self.submit(req, decision_id, "STOP")

    def exit_trade(self, decision_id: str, symbol: str, qty: int, bid: float, reason: str, today: date) -> OrderState:
        """Cancel the protective stop, then sell with a marketable limit. One attempt per trading day."""
        stop_coid = client_order_id(decision_id, "STOP")
        stop_row = self.ledger.get_order(stop_coid)
        if stop_row is not None:
            self.cancel(stop_coid)
            state = OrderState(self.ledger.get_order(stop_coid)["state"])
            if state == OrderState.FILLED:
                return state  # the stop already closed the position
            if state not in TERMINAL_STATES:
                self.ledger.append("exit_deferred", {"reason": "stop cancel not confirmed"}, decision_id=decision_id)
                return state
        limit = round(bid * (1 - self.limits.execution.price_collar_pct / 100), 2)
        req = OrderRequest(client_order_id=client_order_id(decision_id, f"EXIT:{today.isoformat()}"), symbol=symbol,
                           side="SELL", order_type="LIMIT", quantity=qty, time_in_force="DAY", limit_price=limit)
        self.ledger.append("exit_requested", {"reason": reason, "limit": limit}, decision_id=decision_id)
        return self.submit(req, decision_id, "EXIT")
