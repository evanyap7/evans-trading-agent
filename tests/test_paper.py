import json

from trading_agent.broker.base import OrderRequest
from trading_agent.broker.paper import PaperBroker
from trading_agent.broker.simulated import SimulatedBroker
from trading_agent.schemas import BrokerOrderStatus


def _data():
    d = SimulatedBroker()
    d.set_quote("XLK", bid=99.99, ask=100.01, last=100.0)
    return d


def _order(coid, side="BUY", **kw):
    base = dict(client_order_id=coid, symbol="XLK", side=side, order_type="LIMIT", quantity=5,
                time_in_force="DAY", limit_price=100.05)
    return OrderRequest(**{**base, **kw})


def test_paper_book_survives_restarts_and_fills_on_live_quotes(tmp_path):
    path, data = tmp_path / "paper.json", _data()
    b = PaperBroker(data, path, cash=1000)
    b.place(_order("entry"))  # marketable: fills at the limit against the real quote
    assert b.get_order("entry").status == BrokerOrderStatus.FILLED
    b.place(_order("stop", side="SELL", order_type="STOP_LOSS", limit_price=None, stop_price=95.0,
                   time_in_force="GTC"))

    b2 = PaperBroker(data, path)  # next tick is a new process
    assert b2.cash == 1000 - 5 * 100.05 and b2.positions["XLK"].quantity == 5
    assert b2.get_order("stop").status == BrokerOrderStatus.WORKING

    data.set_quote("XLK", bid=94.0, ask=94.02, last=94.01)
    b2.get_quotes(["XLK"])  # a fresh real quote triggers the resting stop
    assert b2.get_order("stop").status == BrokerOrderStatus.FILLED and "XLK" not in b2.positions
    assert PaperBroker(data, path).get_order("stop").status == BrokerOrderStatus.FILLED


def test_yesterdays_day_order_is_cancelled_on_load(tmp_path):
    path, data = tmp_path / "paper.json", _data()
    b = PaperBroker(data, path, cash=1000)
    b.place(_order("resting", limit_price=90.0))  # not marketable: rests
    state = json.loads(path.read_text())
    state["day_orders"]["resting"] = "2000-01-03"
    path.write_text(json.dumps(state))
    assert PaperBroker(data, path).get_order("resting").status == BrokerOrderStatus.CANCELLED
