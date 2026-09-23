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


def build_daily_briefing_text(account: Any, ledger: Any = None, date_str: str = "") -> str:
    """Generates the executive-grade 9:00 AM daily portfolio & activity briefing."""
    from datetime import datetime

    d_str = date_str or datetime.now().strftime("%d %b %Y")
    lines = [
        "🌅 *Evan's Trading Agent: 9:00 AM Morning Briefing*",
        f"📅 *Date*: `{d_str} (09:00 SGT)`",
        "🏛️ *Account*: `Webull SG Cash`",
        "",
        "💰 *Portfolio Snapshot*:",
        f"• *Net Equity*: `${account.equity:,.2f}`",
        f"• *USD Cash*: `${account.cash:,.2f}`",
        f"• *Buying Power*: `${account.buying_power:,.2f}`",
    ]

    # Calculate 1-day equity movement if ledger is available
    if ledger is not None:
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            sod_equity = ledger.start_of_day_equity(datetime.now().date())
            if sod_equity and sod_equity > 0:
                day_change = account.equity - sod_equity
                day_pct = (account.equity / sod_equity - 1) * 100
                emoji = "🟢" if day_change >= 0 else "🔴"
                lines.append(f"• *Daily Performance*: {emoji} `${day_change:+,.2f}` (`{day_pct:+,.2f}%`)")
        except Exception:
            pass

    lines.append("\n📊 *Current Holdings*:")
    if not account.positions:
        lines.append("• _No active equity positions held._")
    else:
        open_trades = {}
        if ledger is not None:
            try:
                open_trades = {t["symbol"]: t for t in ledger.open_trades()}
            except Exception:
                pass

        for p in account.positions:
            qty_str = f"{p.quantity:g}"
            pos_line = f"• *{p.symbol}*: `{qty_str}` shares @ `${p.last_price:,.2f}` (Val: `${p.market_value:,.2f}`)"
            if p.avg_cost and p.avg_cost > 0:
                pnl = (p.last_price - p.avg_cost) * p.quantity
                pnl_pct = (p.last_price / p.avg_cost - 1) * 100
                em = "🟢" if pnl >= 0 else "🔴"
                pos_line += f"\n  └ {em} Unrealized P&L: `${pnl:+,.2f}` (`{pnl_pct:+,.2f}%`)"
            if p.symbol in open_trades:
                t = open_trades[p.symbol]
                pos_line += f"\n  └ 🛡️ Stop: `${t['stop_loss']:.2f}` | 🎯 Target: `${t['take_profit']:.2f}`"
            lines.append(pos_line)

    lines.append("\n📋 *Previous Day Activity & Executions*:")
    activity_found = False
    if ledger is not None:
        try:
            # Check closed trades
            closed = list(ledger.iter_rows("SELECT * FROM trades WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 3"))
            for c in closed:
                activity_found = True
                lines.append(f"• 🎯 *Closed {c['symbol']}*: {c['quantity']:g} shares | Reason: `{c['exit_reason']}`")
            # Check recent order fills/submissions
            recent_orders = list(ledger.iter_rows("SELECT * FROM orders ORDER BY created_at DESC LIMIT 3"))
            for o in recent_orders:
                activity_found = True
                lines.append(f"• ⚡ *Order {o['symbol']}*: `{o['purpose']} {o['side']}` {o['quantity']:g} @ `${o.get('limit_price') or 0:.2f}` -> `{o['state']}`")
        except Exception:
            pass

    if not activity_found:
        lines.append("• _All positions held within risk parameters. No exits or new entries triggered._")

    lines.extend([
        "\n🦅 *Senior Trade Analyst Stance*:",
        "• *Mandate*: Always maximize profits and cut losses immediately.",
        "• *Capital Rotation*: Active. Ready to liquidate stalled holdings if higher-velocity breakouts emerge.",
        "• *Bias*: Bullish Asymmetry. Tight broker-side stops armed on all fills.",
    ])

    return "\n".join(lines)


def send_daily_morning_briefing(account: Any, ledger: Any = None, date_str: str = "") -> bool:
    """Dispatches the 9:00 AM daily briefing to Evan's Telegram assistant."""
    text = build_daily_briefing_text(account, ledger, date_str)
    return send_telegram(text)

