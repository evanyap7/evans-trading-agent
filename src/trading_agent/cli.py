"""Command-line entry point.

    trading-agent accounts                 list Webull accounts (find WEBULL_ACCOUNT_ID)
    trading-agent probe --symbol SPY       dump raw Webull responses to check field mappings
    trading-agent research [--agent llm]   post-close research cycle
    trading-agent execute                  in-session execution of queued proposals
    trading-agent monitor                  reconcile, circuit breakers, exits
    trading-agent tick                     scheduler entry: runs whichever of the above is due (US/Eastern)
    trading-agent status                   kill switch, open trades, orders, recent events
    trading-agent kill --reason "..."      operator stop: block new orders, cancel working entries
    trading-agent unkill
    trading-agent verify-ledger            check the audit hash chain
    trading-agent calibration              the agent's stated confidence vs actual results on closed trades
    trading-agent backtest --start 2024-01-01 [--end ...] [--agent llm] [--cash 1500] [--sweep SPYM]
                                           replay history through the live pipeline

Add `--broker sim` to any cycle to run against synthetic data with no credentials.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import socket
import sys
from pathlib import Path

from .agents import BaselineMomentumAgent, ClaudeResearchAgent, TieredResearchAgent
from .config import TradingMode, load_events, load_risk_limits, load_settings, load_universe
from .killswitch import KillSwitch
from .ledger import Ledger
from .orchestrator import Orchestrator


def _sim_broker(universe):
    from .broker.simulated import SimulatedBroker

    b = SimulatedBroker(cash=10_000)
    for i, sym in enumerate(sorted(universe.symbols)):
        start = 50 + 10 * i
        b.set_trend_bars(sym, start=start, daily_pct=0.15 - 0.02 * (i % 10))
        last = b.bars[sym][-1].close
        b.set_quote(sym, bid=round(last * 0.9998, 2), ask=round(last * 1.0002, 2), last=last)
    return b


NETWORK_TIMEOUT_SECONDS = 60  # backstop for any library call that forgets its own timeout


@contextlib.contextmanager
def _single_instance(state_dir: Path):
    """Exclusive lock so a manual run can never overlap the scheduled tick and double-submit."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "agent.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another trading-agent cycle is running; try again shortly")
        yield


def _run_cycle(args) -> None:
    orch = _build(args)
    print(f"{orch.now().isoformat(timespec='seconds')} mode={orch.settings.trading_mode.value} "
          f"env={orch.settings.webull_environment} broker={args.broker} agent={orch.agent.name}")
    if args.cmd == "morning-report":
        reports = [orch.morning_briefing()]
    elif args.cmd == "tick":
        reports = orch.tick()
    elif args.cmd == "trade-now":
        reports = [orch.research()]
        if orch.ledger.pending():
            reports.append(orch.execute())
        reports.append(orch.monitor())
    else:
        reports = [getattr(orch, args.cmd)()]
    for rep in reports:
        print(f" [{rep.kind}]")
        for n in rep.notes:
            print(f"  - {n}")


def _build(args) -> Orchestrator:
    settings = load_settings()
    limits, universe, events = load_risk_limits(), load_universe(), load_events()
    if args.broker == "sim":
        broker = _sim_broker(universe)
        settings = settings.model_copy(update={"state_dir": settings.state_dir / "sim"})
    else:
        from .broker.webull import WebullBroker

        if not settings.webull_account_id:
            sys.exit("WEBULL_ACCOUNT_ID is not set (run `trading-agent accounts`)")
        broker = WebullBroker(settings.webull_account_id, settings.webull_region, settings.webull_environment)
        if args.broker == "paper":
            from .broker.paper import PaperBroker

            # Real quotes, simulated fills: "sending" orders only ever reaches the paper book.
            os.environ["TRADING_AGENT_PAPER"] = "1"
            settings = settings.model_copy(update={"state_dir": settings.state_dir / "paper",
                                                   "trading_mode": TradingMode.BROKER, "webull_environment": "paper"})
            broker = PaperBroker(broker, settings.state_dir / "paper_broker.json")
    if settings.trading_mode == TradingMode.BROKER and settings.is_production and not limits.live_trading.enabled:
        print("NOTE: prod + broker mode but live_trading.enabled is false: every entry will be risk-rejected.")
    agent = (
        TieredResearchAgent(model_reasoning=settings.llm_model, model_fast=settings.llm_model_fast)
        if args.agent == "llm"
        else BaselineMomentumAgent()
    )
    ledger = Ledger(settings.state_dir / "ledger.db")
    continuous = getattr(args, "continuous", None)
    return Orchestrator(settings=settings, limits=limits, universe=universe, events=events, broker=broker,
                        ledger=ledger, agent=agent, continuous_trading=continuous)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trading-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("research", "execute", "monitor", "tick", "morning-report", "trade-now"):
        c = sub.add_parser(name)
        c.add_argument("--broker", choices=["webull", "paper", "sim"], default="webull",
                       help="paper: live Webull prices, simulated fills, separate ledger in state/paper")
        c.add_argument("--agent", choices=["llm", "baseline"], default="baseline")
        if name == "tick":
            c.add_argument("--continuous", action="store_true", default=None,
                           help="Enable continuous intraday scans every SCAN_INTERVAL_MINUTES")
    sub.add_parser("accounts")
    pr = sub.add_parser("probe")
    pr.add_argument("--symbol", default="SPY")
    sub.add_parser("status")
    k = sub.add_parser("kill")
    k.add_argument("--reason", required=True)
    sub.add_parser("unkill")
    sub.add_parser("verify-ledger")
    sub.add_parser("calibration")
    bt = sub.add_parser("backtest", help="replay historical daily bars through the live decision pipeline")
    bt.add_argument("--start", required=True, type=_date, help="first trading day (YYYY-MM-DD)")
    bt.add_argument("--end", type=_date, default=None, help="last trading day (default: yesterday)")
    bt.add_argument("--agent", choices=["llm", "baseline"], default="baseline")
    bt.add_argument("--cash", type=float, default=1500.0, help="starting cash in USD")
    bt.add_argument("--no-trailing", action="store_true", help="disable the trailing stop (A/B comparison)")
    bt.add_argument("--no-halt", action="store_true",
                    help="keep trading after the drawdown kill switch would have tripped (it is still reported)")
    bt.add_argument("--ignore-earnings", action="store_true",
                    help="allow stocks with no known earnings date (live blocks them)")
    sweep = bt.add_mutually_exclusive_group()
    sweep.add_argument("--sweep", metavar="SYMBOL", default=None,
                       help="park idle cash in this ETF (e.g. SPYM, SGOV), overriding cash_sweep in the config")
    sweep.add_argument("--no-sweep", action="store_true", help="leave idle cash uninvested, whatever the config says")
    bt.add_argument("--max-llm-calls", type=int, default=40,
                    help="refuse to make more than this many uncached LLM calls (each costs money)")
    bt.add_argument("--out", type=Path, default=None, help="directory for summary.json, trades.csv, equity.csv")
    args = p.parse_args(argv)

    settings = load_settings()
    if args.cmd in ("accounts", "probe"):
        from .broker.webull import WebullBroker

        b = WebullBroker(settings.webull_account_id, settings.webull_region, settings.webull_environment)
        out = b.list_accounts() if args.cmd == "accounts" else b.probe(args.symbol)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd in ("research", "execute", "monitor", "tick", "morning-report", "trade-now"):
        socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        lock_dir = settings.state_dir / args.broker if args.broker in ("sim", "paper") else settings.state_dir
        with _single_instance(lock_dir):  # paper has its own lock so it never blocks a live tick
            _run_cycle(args)
    elif args.cmd == "kill":
        KillSwitch(settings.state_dir).engage(args.reason, by="operator")
        print("kill switch ENGAGED. The next execute/monitor pass cancels working entry orders.")
    elif args.cmd == "unkill":
        KillSwitch(settings.state_dir).release()
        print("kill switch released")
    elif args.cmd == "verify-ledger":
        ok = Ledger(settings.state_dir / "ledger.db").verify_chain()
        print("ledger hash chain OK" if ok else "LEDGER TAMPERING DETECTED")
        sys.exit(0 if ok else 1)
    elif args.cmd == "status":
        _status(settings)
    elif args.cmd == "calibration":
        _calibration(settings)
    elif args.cmd == "backtest":
        socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        _backtest(args, settings)


def _date(s: str):
    from datetime import date

    return date.fromisoformat(s)


def _backtest(args, settings) -> None:
    from datetime import date, datetime, timedelta

    from .backtest import AgentOutputCache, Backtester, load_bars, load_earnings_history

    limits, universe = load_risk_limits(), load_universe()
    end = args.end or date.today() - timedelta(days=1)
    if args.no_trailing:
        limits = limits.model_copy(update={"execution": limits.execution.model_copy(update={"trailing_stops": False})})
    if args.ignore_earnings:
        limits = limits.model_copy(update={"events": limits.events.model_copy(
            update={"require_earnings_data_for_stocks": False})})
    if args.sweep or args.no_sweep:
        cs = limits.cash_sweep.model_copy(update={"enabled": not args.no_sweep,
                                                  **({"symbol": args.sweep.upper()} if args.sweep else {})})
        limits = limits.model_copy(update={"cash_sweep": cs})
    cache_dir = settings.state_dir / "backtest_cache"
    symbols = sorted(set(universe.symbols) | ({limits.cash_sweep.symbol} if limits.cash_sweep.enabled else set()))
    print(f"loading daily bars for {len(symbols)} symbols ...")
    bars = load_bars(symbols, args.start, end, cache_dir)
    stocks = [s for s, sec in universe.symbols.items() if sec.type == "EQUITY"]
    earnings = {} if args.ignore_earnings else load_earnings_history(stocks, cache_dir)
    if args.agent == "llm":
        agent = TieredResearchAgent(model_reasoning=settings.llm_model, model_fast=settings.llm_model_fast)
        print("NOTE: the LLM was trained on text covering these dates, so its results are optimistic (lookahead).")
    else:
        agent = BaselineMomentumAgent()
    cache = (AgentOutputCache(cache_dir / "agent_outputs" / agent.model.replace("/", "_"), args.max_llm_calls)
             if args.agent == "llm" else None)
    bt = Backtester(bars=bars, agent=agent, limits=limits, universe=universe, start=args.start, end=end,
                    starting_cash=args.cash, earnings_history=earnings, halt_on_kill_switch=not args.no_halt,
                    output_cache=cache)
    result = bt.run()
    if not args.ignore_earnings:
        for s in sorted(x for x in stocks if not earnings.get(x)):
            result.warnings.append(f"{s}: no earnings history, so never traded (use --ignore-earnings to allow)")
    missing = sorted(set(symbols) - set(bars))
    if missing:
        result.warnings.append(f"no price data for {', '.join(missing)}")
    result.warnings.append("universe is today's list (survivorship bias); no historical news is replayed")
    print(result.report())
    out = args.out or settings.state_dir / "backtests" / datetime.now().strftime("%Y%m%d-%H%M%S")
    result.write(out)
    print(f"wrote {out}/summary.json, trades.csv, equity.csv")


def _calibration(settings) -> None:
    from .backtest import calibration
    from .schemas import Proposal

    led = Ledger(settings.state_dir / "ledger.db")
    pairs = []
    for t in led.iter_rows("SELECT * FROM trades WHERE status='CLOSED'"):
        pa = led.db.execute("SELECT proposal FROM pending_actions WHERE decision_id=?", (t["decision_id"],)).fetchone()
        fills = led.db.execute("SELECT SUM(filled_quantity) q, SUM(filled_quantity * filled_price) v FROM orders "
                               "WHERE decision_id=? AND purpose IN ('EXIT','STOP') AND state='FILLED'",
                               (t["decision_id"],)).fetchone()
        risk = abs(t["entry_price"] - (t["initial_stop"] or t["stop_loss"]))
        if pa is None or not fills["q"] or risk <= 0:
            continue
        sign = -1 if t["side"] == "SELL_SHORT" else 1
        pairs.append((Proposal.model_validate_json(pa["proposal"]).trade.confidence,
                      sign * (fills["v"] / fills["q"] - t["entry_price"]) / risk))
    print(f"{len(pairs)} closed system trades with a known confidence and exit fill")
    for bucket, row in calibration(pairs).items():
        print(f"  confidence {bucket}: {row}")


def _status(settings) -> None:
    ks = KillSwitch(settings.state_dir)
    led = Ledger(settings.state_dir / "ledger.db")
    limits = load_risk_limits()
    target = getattr(limits.account, "daily_profit_target_usd", 10.0)
    print(f"mode={settings.trading_mode.value} env={settings.webull_environment} continuous={settings.continuous_trading}")
    print(f"daily profit target: +${target:.2f} USD")
    print(f"kill switch: {'ENGAGED - ' + ks.reason() if ks.engaged() else 'off'}")
    print("open trades:")
    for t in led.open_trades():
        print(f"  {t['symbol']} x{t['quantity']} entry {t['entry_price']} stop {t['stop_loss']} "
              f"target {t['take_profit']} time-stop {t['time_stop_date']}")
    print("live orders:")
    for o in led.live_orders():
        print(f"  {o['purpose']} {o['side']} {o['symbol']} x{o['quantity']} {o['order_type']} {o['state']}")
    print("pending actions:")
    for r in led.pending():
        print(f"  {r['decision_id'][:24]} created {r['created_at'][:19]}")
    print("recent events:")
    for e in led.events(limit=15):
        print(f"  {e['ts'][:19]} {e['kind']:<16} {e['payload'][:110]}")


if __name__ == "__main__":
    main()
