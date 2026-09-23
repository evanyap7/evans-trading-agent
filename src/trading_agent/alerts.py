"""Telegram alert dispatcher for Evan's Trading Agent.

Sends real-time updates directly to Evan's Telegram personal assistant:
- Approved trade proposals & theses
- Order submissions and Webull fills
- Broker-side GTC stop-loss confirmations
- Automated take-profit and stop-loss exits
- Circuit breakers & emergency stops
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv

ASSISTANT_ENV = Path("/Users/evanyap7/telegram-personal-assistant/.env.local")


def get_telegram_config() -> tuple[str, str]:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        if ASSISTANT_ENV.exists():
            try:
                vals = dotenv_values(ASSISTANT_ENV)
                token = token or str(vals.get("TELEGRAM_BOT_TOKEN", "")).strip()
                chat_id = chat_id or str(vals.get("TELEGRAM_ALLOWED_USER_ID", "")).strip()
            except Exception:
                pass
    return token, chat_id


def send_telegram(text: str) -> bool:
    """Dispatches a markdown-formatted message to Evan's Telegram assistant."""
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=8,
        )
        return resp.status_code == 200
    except Exception:
        return False


def alert_research_summary(date_str: str, n_proposals: int, n_evidence: int, notes: list[str]) -> None:
    lines = [
        f"📊 *Evan's Trading Agent: Research Complete*",
        f"📅 *Date*: `{date_str}`",
        f"🌐 *Evidence Gathered*: `{n_evidence}` items (Bloomberg, WSJ, Economist)",
        f"💡 *Actionable Proposals*: `{n_proposals}`",
    ]
    if notes:
        lines.append("\n*Notes*:")
        for n in notes[:5]:
            lines.append(f"• {n}")
    send_telegram("\n".join(lines))


def alert_trade_queued(symbol: str, shares: int, limit_price: float, stop_loss: float, take_profit: float, thesis: str) -> None:
    msg = (
        f"💡 *Trade Approved & Queued*\n\n"
        f"• *Ticker*: `{symbol}`\n"
        f"• *Quantity*: `{shares}` shares\n"
        f"• *Limit Entry*: `${limit_price:.2f}`\n"
        f"• *Stop Loss*: `${stop_loss:.2f}`\n"
        f"• *Take Profit*: `${take_profit:.2f}`\n\n"
        f"📝 *Thesis*: {thesis}"
    )
    send_telegram(msg)


def alert_order_submitted(symbol: str, side: str, qty: int, price: float | None, is_shadow: bool = False) -> None:
    tag = "🧪 [SHADOW MODE]" if is_shadow else "🚀 [LIVE WEBULL]"
    p_str = f"${price:.2f}" if price is not None else "Market"
    msg = (
        f"{tag} *Order Submitted*\n\n"
        f"• *Action*: `{side}` {qty} shares of `{symbol}`\n"
        f"• *Target Price*: `{p_str}`\n"
        f"• *Broker-Side Stop*: GTC stop armed on fill"
    )
    send_telegram(msg)


def alert_trade_exited(symbol: str, qty: int, price: float, reason: str) -> None:
    emoji = "🎯" if "profit" in reason.lower() else "🛡️"
    msg = (
        f"{emoji} *Position Exited*\n\n"
        f"• *Ticker*: `{symbol}`\n"
        f"• *Shares Sold*: `{qty}`\n"
        f"• *Execution Price*: `${price:.2f}`\n"
        f"• *Exit Reason*: `{reason}`"
    )
    send_telegram(msg)


def alert_killswitch(reason: str, engaged: bool = True) -> None:
    if engaged:
        msg = f"🚨 *EMERGENCY KILL SWITCH ENGAGED*\n\n*Reason*: {reason}\n*Status*: All working orders cancelled. Trading paused."
    else:
        msg = f"✅ *Kill Switch Released*\n\nAutonomous trading resumed."
    send_telegram(msg)
