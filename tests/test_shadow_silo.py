from unittest.mock import patch

from conftest import IN_SESSION, ScriptedAgent, make_orchestrator, make_settings, seeded_broker

from trading_agent import alerts
from trading_agent.cli import _ledger_path
from trading_agent.config import TradingMode
from trading_agent.ledger import Ledger
from trading_agent.schemas import AgentOutput


def idle(ctx):
    return AgentOutput(market_view="x", proposals=[], no_trade_reason="idle")


def test_shadow_is_quiet_but_critical_alerts_get_through():
    with patch.object(alerts, "get_telegram_config", return_value=("t", "c")), \
            patch.object(alerts.requests, "post") as post:
        post.return_value.status_code = 200
        alerts.QUIET = True
        alerts.alert_trade_exited("SMCI", 1, 44.0, "time_stop (SHADOW)")
        assert post.call_count == 0
        alerts.alert_killswitch("test")
        alerts.alert_cycle_error("monitor", "boom")
        assert post.call_count == 2


def test_orchestrator_mutes_only_outside_broker_mode(tmp_path):
    make_orchestrator(tmp_path, seeded_broker(IN_SESSION), ScriptedAgent(idle), IN_SESSION, mode=TradingMode.SHADOW)
    assert alerts.QUIET is True
    make_orchestrator(tmp_path, seeded_broker(IN_SESSION), ScriptedAgent(idle), IN_SESSION, mode=TradingMode.BROKER)
    assert alerts.QUIET is False


def test_shadow_ledger_is_a_separate_copy_and_live_stays_untouched(tmp_path):
    live = Ledger(tmp_path / "ledger.db")
    live.append("marker", {"x": 1})
    live.db.close()
    shadow_settings = make_settings(tmp_path, TradingMode.SHADOW, "prod")
    path = _ledger_path(shadow_settings)
    assert path == tmp_path / "shadow" / "ledger.db" and path != tmp_path / "ledger.db"
    sh = Ledger(path)
    assert len(sh.events("marker")) == 1  # seeded from the live ledger
    sh.append("shadow_only", {"y": 2})
    assert Ledger(tmp_path / "ledger.db").events("shadow_only") == []
    assert _ledger_path(make_settings(tmp_path, TradingMode.BROKER, "prod")) == tmp_path / "ledger.db"
