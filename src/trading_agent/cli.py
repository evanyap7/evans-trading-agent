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

Add `--broker sim` to any cycle to run against synthetic data with no credentials.
"""

from __future__ import annotations

import argparse
import json
import sys

from .agents import BaselineMomentumAgent, ClaudeResearchAgent
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
    if settings.trading_mode == TradingMode.BROKER and settings.is_production and not limits.live_trading.enabled:
        print("NOTE: prod + broker mode but live_trading.enabled is false: every entry will be risk-rejected.")
    agent = ClaudeResearchAgent(settings.llm_model) if args.agent == "llm" else BaselineMomentumAgent()
    ledger = Ledger(settings.state_dir / "ledger.db")
    return Orchestrator(settings=settings, limits=limits, universe=universe, events=events, broker=broker,
                        ledger=ledger, agent=agent)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trading-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("research", "execute", "monitor", "tick"):
        c = sub.add_parser(name)
        c.add_argument("--broker", choices=["webull", "sim"], default="webull")
        c.add_argument("--agent", choices=["llm", "baseline"], default="baseline")
    sub.add_parser("accounts")
    pr = sub.add_parser("probe")
    pr.add_argument("--symbol", default="SPY")
    sub.add_parser("status")
    k = sub.add_parser("kill")
    k.add_argument("--reason", required=True)
    sub.add_parser("unkill")
    sub.add_parser("verify-ledger")
    args = p.parse_args(argv)

    settings = load_settings()
    if args.cmd in ("accounts", "probe"):
        from .broker.webull import WebullBroker

        b = WebullBroker(settings.webull_account_id, settings.webull_region, settings.webull_environment)
        out = b.list_accounts() if args.cmd == "accounts" else b.probe(args.symbol)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd in ("research", "execute", "monitor", "tick"):
        orch = _build(args)
        print(f"{orch.now().isoformat(timespec='seconds')} mode={orch.settings.trading_mode.value} "
              f"env={orch.settings.webull_environment} broker={args.broker} agent={orch.agent.name}")
        reports = orch.tick() if args.cmd == "tick" else [getattr(orch, args.cmd)()]
        for rep in reports:
            print(f" [{rep.kind}]")
            for n in rep.notes:
                print(f"  - {n}")
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


def _status(settings) -> None:
    ks = KillSwitch(settings.state_dir)
    led = Ledger(settings.state_dir / "ledger.db")
    print(f"mode={settings.trading_mode.value} env={settings.webull_environment}")
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
