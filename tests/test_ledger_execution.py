import sqlite3
from datetime import timedelta

import pytest

from trading_agent.broker.base import OrderRequest
from trading_agent.broker.simulated import SimulatedBroker
from trading_agent.execution import ExecutionEngine, client_order_id
from trading_agent.ledger import Ledger
from trading_agent.schemas import OrderState, utcnow


def test_events_are_append_only(ledger):
    ledger.append("x", {"a": 1})
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute("UPDATE events SET payload='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute("DELETE FROM events")


def test_hash_chain_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.db")
    for i in range(3):
        led.append("x", {"i": i})
    assert led.verify_chain()
    led.db.execute("DROP TRIGGER events_no_update")
    led.db.execute("UPDATE events SET payload='{\"i\":99}' WHERE seq=2")
    assert not led.verify_chain()


def _req(decision="d1", qty=5, limit=101.0):
    return OrderRequest(client_order_id=client_order_id(decision, "ENTRY"), symbol="XLK", side="BUY",
                        order_type="LIMIT", quantity=qty, time_in_force="DAY", limit_price=limit)


@pytest.fixture
def sim():
    b = SimulatedBroker()
    b.set_quote("XLK", bid=100.99, ask=101.01, last=101.0)
    return b


def test_shadow_mode_sends_nothing(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=False)
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.SHADOW
    assert sim.place_calls == 0


def test_duplicate_scheduler_run_places_once(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    eng.submit(_req(), "d1", "ENTRY")
    eng.submit(_req(), "d1", "ENTRY")
    assert sim.place_calls == 1


def test_timeout_after_broker_received_is_reconciled_not_retried(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "timeout_received"
    state = eng.submit(_req(limit=100.0), "d1", "ENTRY")  # rests (not marketable)
    assert state == OrderState.ACKNOWLEDGED
    assert sim.place_calls == 1


def test_timeout_where_order_was_lost_stays_unknown_then_resolves(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "timeout_lost"
    coid = _req().client_order_id
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.UNKNOWN_RECONCILE
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.UNKNOWN_RECONCILE  # no blind retry
    assert sim.place_calls == 1
    old = (utcnow() - timedelta(minutes=30)).isoformat()
    ledger.db.execute("UPDATE orders SET updated_at=? WHERE client_order_id=?", (old, coid))
    assert eng.sync(coid) == OrderState.REJECTED


def test_explicit_rejection_is_terminal(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "reject"
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.REJECTED


def test_preview_mismatch_blocks_submission(sim, ledger, limits):
    from trading_agent.broker.base import PreviewResult

    sim.preview = lambda o: PreviewResult(ok=True, estimated_cost=99_999.0, estimated_fees=0)
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.REJECTED
    assert sim.place_calls == 0


def test_partial_fill_then_restart_is_recovered(sim, ledger, limits, tmp_path):
    """A new engine (after restart) picks up broker state from the ledger + broker."""
    from trading_agent.schemas import BrokerOrderStatus

    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    coid = _req(limit=100.0).client_order_id
    eng.submit(_req(limit=100.0), "d1", "ENTRY")
    o = sim.orders[coid]
    sim.orders[coid] = o.model_copy(update={"status": BrokerOrderStatus.PARTIALLY_FILLED, "filled_quantity": 2,
                                            "filled_price": 100.0})
    restarted = ExecutionEngine(sim, ledger, limits, send_orders=True)
    assert restarted.sync(coid) == OrderState.PARTIALLY_FILLED
    assert ledger.get_order(coid)["filled_quantity"] == 2
