from __future__ import annotations

from unittest.mock import MagicMock, patch

from trading_agent.alerts import (
    alert_killswitch,
    alert_order_submitted,
    alert_research_summary,
    alert_trade_exited,
    alert_trade_queued,
    get_telegram_config,
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
    alert_trade_exited("NVDA", 10, 135.0, "take_profit")
    assert "Position Exited" in mock_send.call_args[0][0]
    assert "take_profit" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_killswitch("1% daily loss limit", engaged=True)
    assert "EMERGENCY KILL SWITCH ENGAGED" in mock_send.call_args[0][0]

    mock_send.reset_mock()
    alert_killswitch("", engaged=False)
    assert "Kill Switch Released" in mock_send.call_args[0][0]
