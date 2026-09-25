from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from trading_agent.alerts import (
    alert_killswitch,
    alert_order_submitted,
    alert_research_summary,
    alert_stock_bought,
    alert_stock_sold,
    alert_stop_raised,
    alert_trade_exited,
    alert_trade_queued,
    build_daily_briefing_text,
    get_telegram_config,
    send_daily_morning_briefing,
    send_telegram,
)


def test_get_telegram_config(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token_123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_456")
    token, chat_id = get_telegram_config()
    assert token == "fake_token_123"
    assert chat_id == "fake_chat_456"


@patch("trading_agent.alerts.get_telegram_config", return_value=("fake_token", "fake_chat"))
@patch("requests.post")
def test_send_telegram_success(mock_post, mock_cfg):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    ok = send_telegram("Hello Evan!")
    assert ok is True
    assert mock_post.called
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"]["chat_id"] == "fake_chat"
    assert "Hello Evan!" in kwargs["json"]["text"]


@patch("trading_agent.alerts.get_telegram_config", return_value=("fake_token", "fake_chat"))
@patch("requests.post")
def test_send_telegram_failure_does_not_crash(mock_post, mock_cfg):
    mock_post.side_effect = Exception("network down")
    ok = send_telegram("Test message")
    assert ok is False


@patch("trading_agent.alerts.send_telegram")
def test_alert_helpers(mock_send):
    alert_research_summary("2026-09-23", 2, 10, ["Note 1", "Note 2"])
    assert mock_send.called
    assert "Research Complete" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_trade_queued("NVDA", 10, 120.0, 115.0, 135.0, "AI acceleration momentum")
    assert "NVDA" in mock_send.call_args[0][0]
    assert "AI acceleration momentum" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_order_submitted("NVDA", "BUY", 10, 120.0, is_shadow=True)
    assert "[SHADOW MODE]" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_order_submitted("NVDA", "BUY", 10, 120.0, is_shadow=False)
    assert "[LIVE WEBULL]" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_stock_bought("NVDA", 5, 120.0, stop_loss=115.0, take_profit=135.0, thesis="Strong earnings beat")
    msg = mock_send.call_args[0][0]
    assert "Stock Bought: NVDA" in msg
    assert "120.00" in msg
    assert "115.00" in msg
    assert "135.00" in msg
    assert "Strong earnings beat" in msg

    mock_send.reset_mock()
    alert_stock_sold("NVDA", 5, 135.0, entry_price=120.0, reason="take_profit", pnl=75.0, pnl_pct=12.5)
    msg = mock_send.call_args[0][0]
    assert "Stock Sold: NVDA" in msg
    assert "135.00" in msg
    assert "+$75.00" in msg
    assert "+12.50%" in msg
    assert "Take-Profit Target Hit" in msg

    mock_send.reset_mock()
    alert_stock_sold("PLTR", 2, 185.0, entry_price=190.0, reason="stop_loss", pnl=-10.0, pnl_pct=-2.63)
    msg = mock_send.call_args[0][0]
    assert "Stock Sold: PLTR" in msg
    assert "185.00" in msg
    assert "-$10.00" in msg
    assert "Protective Stop Triggered" in msg

    mock_send.reset_mock()
    alert_stop_raised("PLTR", old_stop=191.0, new_stop=196.0, current_price=200.0)
    msg = mock_send.call_args[0][0]
    assert "Trailing Stop Raised: PLTR" in msg
    assert "196.00" in msg
    assert "191.00" in msg
    assert "200.00" in msg

    mock_send.reset_mock()
    alert_trade_exited("NVDA", 10, 135.0, "take_profit")
    assert "Exit Order" in mock_send.call_args[0][0]
    assert "take_profit" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_killswitch("1% daily loss limit", engaged=True)
    assert "EMERGENCY KILL SWITCH ENGAGED" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_killswitch("", engaged=False)
    assert "Kill Switch Released" in mock_send.call_args[0][0]

    # Short alerts
    from trading_agent.alerts import alert_short_covered, alert_stock_shorted

    mock_send.reset_mock()
    alert_stock_shorted("TSLA", 5, 200.0, stop_loss=210.0, take_profit=180.0, thesis="Structural breakdown")
    msg = mock_send.call_args[0][0]
    assert "Stock Shorted: TSLA" in msg
    assert "200.00" in msg
    assert "210.00" in msg
    assert "180.00" in msg
    assert "Structural breakdown" in msg

    mock_send.reset_mock()
    alert_short_covered("TSLA", 5, 180.0, entry_price=200.0, reason="take_profit", pnl=100.0, pnl_pct=10.0)
    msg = mock_send.call_args[0][0]
    assert "Short Covered: TSLA" in msg
    assert "180.00" in msg
    assert "+$100.00" in msg
    assert "+10.00%" in msg
    assert "Take-Profit Target Hit" in msg

    mock_send.reset_mock()
    alert_stop_raised("TSLA", old_stop=210.0, new_stop=200.0, current_price=190.0, is_short=True)
    msg = mock_send.call_args[0][0]
    assert "Trailing Stop Lowered (Short): TSLA" in msg
    assert "200.00" in msg
    assert "lower level" in msg



def test_daily_morning_briefing():
    from trading_agent.schemas import AccountState, Position

    acct = AccountState(
        account_id="12345",
        equity=956.11,
        cash=0.26,
        buying_power=0.26,
        positions=[
            Position(symbol="GOOG", quantity=1.0, avg_cost=150.0, last_price=170.0),
            Position(symbol="NFLX", quantity=2.0, avg_cost=60.0, last_price=72.44),
        ],
        open_orders=[],
        as_of=datetime.now(),
    )
    text = build_daily_briefing_text(acct, date_str="23 Sep 2026")
    assert "Morning Briefing" in text
    assert "$956.11 USD" in text
    assert "GOOG" in text
    assert "NFLX" in text
    assert "P&L Summary" in text
    assert "Open Unrealized P&L" in text
    assert "Senior Trade Analyst" in text

    with patch("trading_agent.alerts.send_telegram", return_value=True) as mock_tg:
        ok = send_daily_morning_briefing(acct, date_str="23 Sep 2026")
        assert ok is True
        assert mock_tg.called


