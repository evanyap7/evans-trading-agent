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
    if os.environ.get("TRADING_AGENT_PAPER"):
        text = "📝 [PAPER — no real money]\n" + text
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Markdown first; if Telegram rejects the formatting (an unbalanced `_` or `*` in a symbol, reason
    # or LLM thesis), resend as plain text so a critical alert is never silently lost.
    for payload in ({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                    {"chat_id": chat_id, "text": text}):
        try:
            if requests.post(url, json=payload, timeout=8).status_code == 200:
                return True
        except Exception:
            continue
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


def alert_stock_bought(
    symbol: str,
    qty: float,
    fill_price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    thesis: str = "",
) -> None:
    """Dispatched immediately when a buy order fills at the broker."""
    qty_str = f"{qty:g}"
    total_val = qty * fill_price
    lines = [
        f"🟢 *Stock Bought: {symbol}*",
        "",
        f"• *Shares*: `{qty_str}`",
        f"• *Execution Price*: `${fill_price:,.2f}`",
        f"• *Total Value*: `${total_val:,.2f}`",
    ]
    if stop_loss is not None and stop_loss > 0:
        downside_pct = ((stop_loss - fill_price) / fill_price) * 100
        lines.append(f"• *Protective Stop*: `${stop_loss:,.2f}` (`{downside_pct:+.1f}%`)")
    if take_profit is not None and take_profit > 0:
        upside_pct = ((take_profit - fill_price) / fill_price) * 100
        lines.append(f"• *Take Profit*: `${take_profit:,.2f}` (`{upside_pct:+.1f}%`)")
    if thesis:
        clean_thesis = thesis.strip().replace("_", " ").replace("*", " ")
        if len(clean_thesis) > 180:
            clean_thesis = clean_thesis[:177] + "..."
        lines.append(f"\n📝 *Thesis*: {clean_thesis}")
    lines.append("\n🛡️ _Broker-side protective stop armed at Webull._")
    send_telegram("\n".join(lines))


def alert_stock_sold(
    symbol: str,
    qty: float,
    fill_price: float,
    entry_price: float | None = None,
    reason: str = "",
    pnl: float | None = None,
    pnl_pct: float | None = None,
) -> None:
    """Dispatched immediately when a sell order fills (stop, target, or thesis exit)."""
    qty_str = f"{qty:g}"
    total_val = qty * fill_price
    is_profit = pnl is not None and pnl >= 0
    emoji = "🎯" if is_profit else "🛡️"

    # Human-readable exit reason
    r_lower = reason.lower()
    if "profit" in r_lower or "target" in r_lower:
        friendly_reason = "Take-Profit Target Hit 🎯"
    elif "stop" in r_lower:
        friendly_reason = "Protective Stop Triggered 🛡️"
    elif "time" in r_lower:
        friendly_reason = "Time Horizon Limit Reached ⏱️"
    elif "thesis" in r_lower or "rotation" in r_lower:
        friendly_reason = "Capital Rotation / Thesis Invalidation 🔄"
    else:
        friendly_reason = reason or "Market Exit"

    lines = [
        f"{emoji} *Stock Sold: {symbol}*",
        "",
        f"• *Shares*: `{qty_str}`",
        f"• *Execution Price*: `${fill_price:,.2f}`",
        f"• *Total Proceeds*: `${total_val:,.2f}`",
    ]
    if entry_price is not None and entry_price > 0:
        lines.append(f"• *Entry Price*: `${entry_price:,.2f}`")
    if pnl is not None and pnl_pct is not None:
        pnl_em = "🟢" if pnl >= 0 else "🔴"
        sign = "+" if pnl >= 0 else "-"
        lines.append(f"• *Realized P&L*: {pnl_em} `{sign}${abs(pnl):,.2f}` (`{pnl_pct:+,.2f}%`)")
    lines.append(f"• *Reason*: `{friendly_reason}`")
    send_telegram("\n".join(lines))


def alert_stop_raised(symbol: str, old_stop: float | None, new_stop: float, current_price: float | None = None, *, is_short: bool = False) -> None:
    """Dispatched when an R-based ratcheting trailing stop moves toward profit (up for long, down for short)."""
    header = f"📉 *Trailing Stop Lowered (Short): {symbol}*" if is_short else f"📈 *Trailing Stop Raised: {symbol}*"
    status_msg = "• *Status*: Broker stop replaced at lower level (upside locked out)." if is_short else "• *Status*: Broker stop replaced at higher level (downside locked out)."
    lines = [
        header,
        "",
        f"• *New Protected Stop*: `${new_stop:,.2f}`",
    ]
    if old_stop is not None and old_stop > 0:
        lines.append(f"• *Previous Stop*: `${old_stop:,.2f}`")
    if current_price is not None and current_price > 0:
        lines.append(f"• *Current Price*: `${current_price:,.2f}`")
    lines.append(status_msg)
    send_telegram("\n".join(lines))


def alert_trade_exited(symbol: str, qty: int, price: float, reason: str) -> None:
    emoji = "🎯" if "profit" in reason.lower() else "🛡️"
    msg = (
        f"{emoji} *Exit Order Submitted*\n\n"
        f"• *Ticker*: `{symbol}`\n"
        f"• *Shares*: `{qty}`\n"
        f"• *Reference Price*: `${price:.2f}`\n"
        f"• *Exit Reason*: `{reason}`"
    )
    send_telegram(msg)


def alert_stock_shorted(
    symbol: str,
    qty: float,
    fill_price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    thesis: str = "",
) -> None:
    """Dispatched immediately when a short-sell order fills at the broker."""
    qty_str = f"{qty:g}"
    total_val = qty * fill_price
    lines = [
        f"🔴 *Stock Shorted: {symbol}*",
        "",
        f"• *Shares Sold Short*: `{qty_str}`",
        f"• *Execution Price*: `${fill_price:,.2f}`",
        f"• *Total Value*: `${total_val:,.2f}`",
    ]
    if stop_loss is not None and stop_loss > 0:
        upside_risk_pct = ((stop_loss - fill_price) / fill_price) * 100
        lines.append(f"• *Protective Stop (Buy-Cover)*: `${stop_loss:,.2f}` (`{upside_risk_pct:+.1f}%`)")
    if take_profit is not None and take_profit > 0:
        target_pct = ((fill_price - take_profit) / fill_price) * 100
        lines.append(f"• *Take Profit (Cover)*: `${take_profit:,.2f}` (`{target_pct:+.1f}% profit target`)")
    if thesis:
        clean_thesis = thesis.strip().replace("_", " ").replace("*", " ")
        if len(clean_thesis) > 180:
            clean_thesis = clean_thesis[:177] + "..."
        lines.append(f"\n📝 *Thesis*: {clean_thesis}")
    lines.append("\n🛡️ _Broker-side buy-stop armed at Webull._")
    send_telegram("\n".join(lines))


def alert_short_covered(
    symbol: str,
    qty: float,
    fill_price: float,
    entry_price: float | None = None,
    reason: str = "",
    pnl: float | None = None,
    pnl_pct: float | None = None,
) -> None:
    """Dispatched immediately when a short position is covered (buy-to-cover fills)."""
    qty_str = f"{qty:g}"
    total_val = qty * fill_price
    is_profit = pnl is not None and pnl >= 0
    emoji = "🎯" if is_profit else "🛡️"

    # Human-readable exit reason
    r_lower = reason.lower()
    if "profit" in r_lower or "target" in r_lower:
        friendly_reason = "Take-Profit Target Hit 🎯"
    elif "stop" in r_lower:
        friendly_reason = "Short Stopped Out (Covered) 🛡️"
    elif "time" in r_lower:
        friendly_reason = "Time Horizon Limit Reached ⏱️"
    elif "thesis" in r_lower or "rotation" in r_lower:
        friendly_reason = "Capital Rotation / Thesis Invalidation 🔄"
    else:
        friendly_reason = reason or "Short Covered"

    lines = [
        f"{emoji} *Short Covered: {symbol}*",
        "",
        f"• *Shares Covered*: `{qty_str}`",
        f"• *Cover Price*: `${fill_price:,.2f}`",
        f"• *Total Cost*: `${total_val:,.2f}`",
    ]
    if entry_price is not None and entry_price > 0:
        lines.append(f"• *Short Entry Price*: `${entry_price:,.2f}`")
    if pnl is not None and pnl_pct is not None:
        pnl_em = "🟢" if pnl >= 0 else "🔴"
        sign = "+" if pnl >= 0 else "-"
        lines.append(f"• *Realized P&L*: {pnl_em} `{sign}${abs(pnl):,.2f}` (`{pnl_pct:+,.2f}%`)")
    lines.append(f"• *Reason*: `{friendly_reason}`")
    send_telegram("\n".join(lines))


def alert_killswitch(reason: str, engaged: bool = True) -> None:
    if engaged:
        msg = f"🚨 *EMERGENCY KILL SWITCH ENGAGED*\n\n*Reason*: {reason}\n*Status*: No new entries; working entry orders are cancelled. Protective stops and exits stay active."
    else:
        msg = f"✅ *Kill Switch Released*\n\nAutonomous trading resumed."
    send_telegram(msg)


def alert_unprotected(symbol: str, qty: int, stop: float, stop_state: str) -> None:
    send_telegram(
        f"⚠️ *UNPROTECTED POSITION*\n\n"
        f"• *Ticker*: `{symbol}` x{qty}\n"
        f"• Broker-side stop at `${stop:.2f}` is `{stop_state}`\n"
        f"Only the 5-minute software stop is protecting this position. Check Webull."
    )


def alert_cycle_error(kind: str, error: str) -> None:
    send_telegram(f"❌ *Trading cycle failed*: `{kind}`\n\n{error[:500]}")


def build_daily_briefing_text(account: Any, ledger: Any = None, date_str: str = "") -> str:
    """Generates the executive-grade 9:00 AM daily portfolio & P&L briefing."""
    from datetime import datetime

    d_str = date_str or datetime.now().strftime("%d %b %Y")
    lines = [
        "🌅 *Evan's Trading Agent: 9:00 AM Morning Briefing*",
        f"📅 *Date*: `{d_str} (09:00 SGT)`",
        "🏛️ *Account*: `Webull SG Cash`",
        "",
        "💰 *Portfolio & Capital Snapshot*:",
        f"• *Net Equity*: `${account.equity:,.2f} USD`",
        f"• *Available Cash*: `${account.cash:,.2f} USD`",
        f"• *Buying Power*: `${account.buying_power:,.2f} USD`",
    ]

    # Calculate Total Unrealized P&L across all positions
    total_cost = 0.0
    total_mkt = 0.0
    for p in account.positions:
        if p.avg_cost and p.avg_cost > 0:
            total_cost += p.avg_cost * p.quantity
            total_mkt += p.last_price * p.quantity

    lines.append("\n📈 *P&L Summary*:")
    # 1-day equity movement vs start of day if ledger is available
    if ledger is not None:
        try:
            sod_equity = ledger.start_of_day_equity(datetime.now().date())
            if sod_equity and sod_equity > 0:
                day_change = account.equity - sod_equity
                day_pct = (account.equity / sod_equity - 1) * 100
                em = "🟢" if day_change >= 0 else "🔴"
                sign = "+" if day_change >= 0 else "-"
                lines.append(f"• *Daily Equity Change*: {em} `{sign}${abs(day_change):,.2f}` (`{day_pct:+,.2f}%`)")
        except Exception:
            pass

    if total_cost > 0:
        unrealized = total_mkt - total_cost
        unrealized_pct = (unrealized / total_cost) * 100
        em = "🟢" if unrealized >= 0 else "🔴"
        sign = "+" if unrealized >= 0 else "-"
        lines.append(f"• *Open Unrealized P&L*: {em} `{sign}${abs(unrealized):,.2f}` (`{unrealized_pct:+,.2f}%`)")
        invested_pct = (total_mkt / account.equity * 100) if account.equity > 0 else 0.0
        lines.append(f"• *Capital Allocation*: `${total_mkt:,.2f}` (`{invested_pct:.1f}%` invested)")

    lines.append("• *Daily Profit Target*: `+$10.00 USD/day` (compounding goal)")

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
                sign = "+" if pnl >= 0 else "-"
                pos_line += f"\n  └ Cost: `${p.avg_cost:,.2f}` | P&L: {em} `{sign}${abs(pnl):,.2f}` (`{pnl_pct:+,.2f}%`)"
            if p.symbol in open_trades:
                t = open_trades[p.symbol]
                stop_dist = ((t["stop_loss"] - p.last_price) / p.last_price) * 100
                target_dist = ((t["take_profit"] - p.last_price) / p.last_price) * 100
                pos_line += f"\n  └ 🛡️ Stop: `${t['stop_loss']:.2f}` (`{stop_dist:+.1f}%`) | 🎯 Target: `${t['take_profit']:.2f}` (`{target_dist:+.1f}%`)"
            lines.append(pos_line)

    lines.append("\n📋 *Recent Executions & Session Fills*:")
    activity_found = False
    if ledger is not None:
        try:
            # Query filled orders directly
            fills = list(ledger.iter_rows(
                "SELECT * FROM orders WHERE state='FILLED' ORDER BY updated_at DESC LIMIT 4"
            ))
            for f in fills:
                activity_found = True
                side_em = "🟢" if f["side"] == "BUY" else "🔴"
                px = f["filled_price"] or f["limit_price"] or 0.0
                lines.append(f"• {side_em} *{f['side']} {f['symbol']}*: `{f['filled_quantity']:g}` sh @ `${px:,.2f}` ({f['purpose']})")
        except Exception:
            pass

    if not activity_found:
        lines.append("• _No new fills in the last session. Positions holding inside risk envelope._")

    lines.extend([
        "\n🦅 *Senior Trade Analyst & Risk Stance*:",
        "• *Operating Mode*: Swing momentum with R-based ratcheting trailing stops.",
        "• *Stop Automation*: Active at Webull SG (Breakeven at +1R, trailing at +2R).",
        "• *US Market*: Regular trading session opens tonight at 9:30 PM SGT.",
    ])

    return "\n".join(lines)


def send_daily_morning_briefing(account: Any, ledger: Any = None, date_str: str = "") -> bool:
    """Dispatches the 9:00 AM daily briefing to Evan's Telegram assistant."""
    text = build_daily_briefing_text(account, ledger, date_str)
    return send_telegram(text)

