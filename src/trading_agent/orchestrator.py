"""The trading cycles.

research  (after the US close)   evidence -> agent -> verify -> size -> risk -> pending actions
execute   (early regular session) fresh quotes -> re-verify -> re-size -> full risk -> preview -> submit
monitor   (every few minutes)     reconcile -> circuit breakers -> stops / targets / time stops
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable

from .agents import AgentContext, ResearchAgent
from .broker.base import Broker, OrderRequest
from .config import Events, RiskLimits, Settings, TradingMode, Universe
from .execution import ExecutionEngine, client_order_id
from .features import build_evidence
from .killswitch import KillSwitch
from .ledger import Ledger
from .market_calendar import ET, is_regular_session, is_trading_day, to_trading_date
from .portfolio import size_order
from .risk import OpenRisk, RiskContext, evaluate
from .schemas import AccountState, Evidence, OrderState, Proposal, TradeProposal, utcnow
from .verifier import verify

BAR_HISTORY = 260
EXECUTE_AFTER = time(9, 45)    # let the opening auction settle before entering
RESEARCH_AFTER = time(16, 20)  # daily bars are final shortly after the close


@dataclass
class CycleReport:
    kind: str
    notes: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.notes.append(msg)


class Orchestrator:
    def __init__(self, *, settings: Settings, limits: RiskLimits, universe: Universe, events: Events, broker: Broker,
                 ledger: Ledger, agent: ResearchAgent, now: Callable[[], datetime] = utcnow, enable_news: bool = True):
        self.settings = settings
        self.limits = limits
        self.universe = universe
        self.events = events
        self.broker = broker
        self.ledger = ledger
        self.agent = agent
        self.now = now
        self.enable_news = enable_news
        self.kill = KillSwitch(settings.state_dir)
        self.exec = ExecutionEngine(broker, ledger, limits, send_orders=settings.trading_mode == TradingMode.BROKER)

    @property
    def sends_real_orders_to_prod(self) -> bool:
        return self.settings.trading_mode == TradingMode.BROKER and self.settings.is_production

    # -- shared helpers ---------------------------------------------------------------

    def _account(self) -> AccountState:
        acct = self.broker.get_account()
        self.ledger.snapshot_equity(acct.as_of, to_trading_date(acct.as_of), acct.equity)
        return acct

    def reconcile(self, account: AccountState) -> tuple[bool, list[str]]:
        """Sync every live order with the broker, then check positions match what we believe we hold."""
        issues: list[str] = []
        for row in self.ledger.live_orders():
            state = self.exec.sync(row["client_order_id"]) if self.exec.send_orders else OrderState(row["state"])
            if state == OrderState.UNKNOWN_RECONCILE:
                issues.append(f"order {row['client_order_id']} ({row['symbol']}) state unknown at broker")
        held = {p.symbol: p.quantity for p in account.positions}
        expected: dict[str, float] = {}
        for t in self.ledger.open_trades():
            expected[t["symbol"]] = expected.get(t["symbol"], 0) + t["quantity"]
        for sym, qty in expected.items():
            if held.get(sym, 0) + 1e-9 < qty:
                stop_filled = any(
                    r["purpose"] == "STOP" and r["state"] == OrderState.FILLED.value
                    for r in self.ledger.iter_rows("SELECT * FROM orders WHERE symbol=?", (sym,))
                )
                if not stop_filled:
                    issues.append(f"{sym}: ledger expects {qty} shares, broker shows {held.get(sym, 0)}")
        ok = not issues
        self.ledger.append("reconciliation", {"ok": ok, "issues": issues})
        return ok, issues

    def _open_risks(self, account: AccountState) -> list[OpenRisk]:
        stops = {t["symbol"]: t["stop_loss"] for t in self.ledger.open_trades()}
        risks = [OpenRisk(p.symbol, p.quantity, p.last_price, stops.get(p.symbol)) for p in account.positions]
        pending = {r["decision_id"]: r for r in self.ledger.iter_rows("SELECT * FROM pending_actions")}
        for o in self.ledger.live_orders():
            if o["purpose"] != "ENTRY":
                continue
            stop = None
            if o["decision_id"] in pending:
                stop = Proposal.model_validate_json(pending[o["decision_id"]]["proposal"]).trade.exit.stop_loss
            remaining = o["quantity"] - o["filled_quantity"]
            risks.append(OpenRisk(o["symbol"], remaining, o["limit_price"] or 0, stop))
        return risks

    def _risk_context(self, account: AccountState, quote, decision_id: str, require_session: bool,
                      reconciled: bool) -> RiskContext:
        now = self.now()
        td = to_trading_date(now)
        return RiskContext(
            now=now, trading_date=td, account=account, quote=quote, kill_switch_engaged=self.kill.engaged(),
            sends_real_orders_to_prod=self.sends_real_orders_to_prod, require_session=require_session,
            reconciled=reconciled, entries_today=self.ledger.entries_submitted_on(td),
            start_of_day_equity=self.ledger.start_of_day_equity(td), peak_equity=self.ledger.peak_equity(),
            decision_already_executed=self.ledger.get_order(client_order_id(decision_id, "ENTRY")) is not None,
            open_risks=self._open_risks(account),
        )

    # -- research ---------------------------------------------------------------------

    def research(self) -> CycleReport:
        rep = CycleReport("research")
        cycle_id = uuid.uuid4().hex
        account = self._account()
        reconciled, issues = self.reconcile(account)
        for i in issues:
            rep.add(f"reconcile: {i}")

        symbols = sorted(self.universe.symbols)
        bars = self.broker.get_daily_bars(symbols, BAR_HISTORY)
        quotes = self.broker.get_quotes(symbols)
        evidence, features = build_evidence(bars, quotes, self.universe.regime_benchmark)

        # Ingest real-time news & macro evidence from Bloomberg, WSJ, The Economist, Reuters, NYSE
        if self.enable_news:
            try:
                from .news import fetch_all_market_news
                news_docs = fetch_all_market_news(symbols)
                evidence.extend(news_docs)
            except Exception as e:
                rep.add(f"news ingestion note: {e}")
        trades = {t["symbol"]: t for t in self.ledger.open_trades()}
        for p in account.positions:
            t = trades.get(p.symbol)
            payload = {"quantity": p.quantity, "avg_cost": p.avg_cost, "last_price": p.last_price,
                       "system_stop": t["stop_loss"] if t else None, "system_target": t["take_profit"] if t else None,
                       "time_stop_date": t["time_stop_date"] if t else None}
            evidence.append(Evidence(evidence_id=f"pos_{p.symbol}_{cycle_id[:8]}", kind="position", symbol=p.symbol,
                                     as_of=account.as_of, source="webull.positions", payload=payload))
        for ev in evidence:
            self.ledger.add_evidence(cycle_id, ev)
        rep.add(f"cycle {cycle_id[:8]}: {len(evidence)} evidence items, {len(features)} symbols with features")

        exposure = sum(p.market_value for p in account.positions)
        ctx = AgentContext(
            as_of=to_trading_date(self.now()), evidence=evidence,
            universe={s: {"type": sec.type, "sector": sec.sector} for s, sec in self.universe.symbols.items()},
            account_summary={"equity_usd": round(account.equity, 2), "cash_usd": round(account.cash, 2),
                             "gross_exposure_pct": round(exposure / account.equity * 100, 2) if account.equity else None},
            positions=[{"symbol": p.symbol, "quantity": p.quantity, "avg_cost": p.avg_cost, "last": p.last_price}
                       for p in account.positions],
            known_earnings={s: d.isoformat() for s, d in self.events.earnings.items()},
            features=features,
        )
        try:
            output = self.agent.propose(ctx)
        except Exception as e:  # a failed model call is a NO_TRADE, never a crash that skips monitoring
            self.ledger.append("agent_error", {"agent": self.agent.name, "error": str(e)[:500]})
            rep.add(f"agent error: {e}")
            return rep
        self.ledger.append("agent_output", {"agent": self.agent.name, "model": self.agent.model,
                                            "cycle_id": cycle_id, "output": output.model_dump(mode="json")})
        rep.add(f"agent: {len(output.proposals)} proposals, {len(output.exits)} exits"
                + (f" (no trade: {output.no_trade_reason})" if not output.proposals else ""))

        cycle_ev = self.ledger.evidence_ids(cycle_id)
        for trade in output.proposals:
            self._consider_entry(trade, cycle_id, cycle_ev, features, quotes, account, reconciled, rep)
        for ex in output.exits:
            held = account.position(ex.symbol)
            grounded = all(e in cycle_ev for e in ex.evidence_ids)
            if held and grounded and ex.symbol in trades:
                did = trades[ex.symbol]["decision_id"]
                self.ledger.add_pending_action(f"close:{did}:{cycle_id[:8]}", cycle_id,
                                               ex.model_dump_json())
                rep.add(f"exit queued for {ex.symbol}: {ex.reason[:80]}")
            else:
                rep.add(f"exit for {ex.symbol} ignored (held={bool(held)}, grounded={grounded}, system trade={ex.symbol in trades})")
        return rep

    def _consider_entry(self, trade: TradeProposal, cycle_id: str, cycle_ev: dict, features: dict, quotes: dict,
                        account: AccountState, reconciled: bool, rep: CycleReport) -> None:
        decision_id = uuid.uuid4().hex
        prop = Proposal(decision_id=decision_id, cycle_id=cycle_id, created_at=self.now(), source=self.agent.name,
                        model=self.agent.model, trade=trade)
        self.ledger.append("proposal", prop.model_dump(mode="json"), decision_id=decision_id)
        v = verify(trade, cycle_evidence=cycle_ev, features=features.get(trade.symbol), quote=quotes.get(trade.symbol),
                   limits=self.limits, universe=self.universe, now=self.now(), require_fresh_quote=False)
        self.ledger.append("verification", v.model_dump(), decision_id=decision_id)
        if not v.passed:
            rep.add(f"{trade.symbol}: verifier rejected: {'; '.join(v.failures)}")
            return
        sized = size_order(decision_id, trade, account, features[trade.symbol]["avg_volume_20d"], self.limits)
        if isinstance(sized, str):
            self.ledger.append("sizing_rejected", {"reason": sized}, decision_id=decision_id)
            rep.add(f"{trade.symbol}: sizing rejected: {sized}")
            return
        ctx = self._risk_context(account, quotes.get(trade.symbol), decision_id, require_session=False,
                                 reconciled=reconciled)
        d = evaluate(sized, trade.instrument_type, trade.holding_period_days, ctx, self.limits, self.universe, self.events)
        self.ledger.append("risk_decision", {"stage": "research", **d.model_dump(), "sized": sized.model_dump()},
                           decision_id=decision_id)
        if not d.approved:
            rep.add(f"{trade.symbol}: risk rejected: {', '.join(d.failed_checks)}")
            return
        self.ledger.add_pending_action(decision_id, cycle_id, prop.model_dump_json())
        rep.add(f"{trade.symbol}: queued {sized.quantity} @ {sized.limit_price} (stop {sized.stop_loss}, "
                f"target {sized.take_profit}, net edge {v.metrics.get('net_edge_pct')}%)")

    # -- execute ------------------------------------------------------------------------

    def execute(self) -> CycleReport:
        rep = CycleReport("execute")
        now = self.now()
        if self.kill.engaged():
            self._cancel_working_entries(rep)
            rep.add(f"kill switch engaged ({self.kill.reason()}): nothing sent")
            return rep
        if not is_regular_session(now):
            rep.add("market closed: nothing sent")
            return rep
        account = self._account()
        reconciled, issues = self.reconcile(account)
        rep.notes += [f"reconcile: {i}" for i in issues]
        pending = self.ledger.pending()
        symbols = sorted({self._pending_symbol(r) for r in pending})
        quotes = self.broker.get_quotes(symbols) if symbols else {}
        bars = self.broker.get_daily_bars(symbols, BAR_HISTORY) if symbols else {}
        evidence, features = build_evidence(bars, quotes, self.universe.regime_benchmark)
        today = to_trading_date(now)

        for row in pending:
            if row["decision_id"].startswith("close:"):
                self._execute_close(row, account, quotes, today, rep)
                continue
            prop = Proposal.model_validate_json(row["proposal"])
            t = prop.trade
            if to_trading_date(prop.created_at) < self._previous_trading_day(today):
                self.ledger.set_pending_status(prop.decision_id, "EXPIRED")
                rep.add(f"{t.symbol}: proposal expired")
                continue
            v = verify(t, cycle_evidence=self.ledger.evidence_ids(prop.cycle_id), features=features.get(t.symbol),
                       quote=quotes.get(t.symbol), limits=self.limits, universe=self.universe, now=now,
                       require_fresh_quote=True)
            self.ledger.append("verification", {"stage": "execute", **v.model_dump()}, decision_id=prop.decision_id)
            if not v.passed:
                self.ledger.set_pending_status(prop.decision_id, "DROPPED")
                rep.add(f"{t.symbol}: dropped at execution: {'; '.join(v.failures)}")
                continue
            sized = size_order(prop.decision_id, t, account, features[t.symbol]["avg_volume_20d"], self.limits)
            if isinstance(sized, str):
                self.ledger.set_pending_status(prop.decision_id, "DROPPED")
                rep.add(f"{t.symbol}: dropped at sizing: {sized}")
                continue
            ctx = self._risk_context(account, quotes.get(t.symbol), prop.decision_id, require_session=True,
                                     reconciled=reconciled)
            d = evaluate(sized, t.instrument_type, t.holding_period_days, ctx, self.limits, self.universe, self.events)
            self.ledger.append("risk_decision", {"stage": "execute", **d.model_dump(), "sized": sized.model_dump()},
                               decision_id=prop.decision_id)
            if not d.approved:
                self.ledger.set_pending_status(prop.decision_id, "DROPPED")
                rep.add(f"{t.symbol}: risk rejected at execution: {', '.join(d.failed_checks)}")
                continue
            req = OrderRequest(client_order_id=client_order_id(prop.decision_id, "ENTRY"), symbol=t.symbol,
                               side="BUY", order_type="LIMIT", quantity=sized.quantity, time_in_force="DAY",
                               limit_price=round(sized.limit_price, 2), instrument_type=t.instrument_type)
            state = self.exec.submit(req, prop.decision_id, "ENTRY")
            self.ledger.set_pending_status(prop.decision_id, "SUBMITTED")
            rep.add(f"{t.symbol}: BUY {sized.quantity} @ {sized.limit_price} -> {state.value}")
            account = self._account()  # refresh so the next order sees this one's cash and exposure
        return rep

    @staticmethod
    def _pending_symbol(row) -> str:
        data = json.loads(row["proposal"])
        return data["symbol"] if "symbol" in data else data["trade"]["symbol"]

    @staticmethod
    def _previous_trading_day(d: date) -> date:
        prev = d - timedelta(days=1)
        while not is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev

    def _execute_close(self, row, account: AccountState, quotes: dict, today: date, rep: CycleReport) -> None:
        did = row["decision_id"].split(":")[1]
        trade = next((t for t in self.ledger.open_trades() if t["decision_id"] == did), None)
        q = quotes.get(trade["symbol"]) if trade else None
        if trade is None or q is None:
            self.ledger.set_pending_status(row["decision_id"], "DROPPED")
            return
        held = account.position(trade["symbol"])
        qty = int(min(trade["quantity"], held.quantity if held else 0))
        if qty > 0:
            state = self.exec.exit_trade(did, trade["symbol"], qty, q.bid, "agent_thesis_exit", today)
            rep.add(f"{trade['symbol']}: agent exit SELL {qty} -> {state.value}")
        self.ledger.set_pending_status(row["decision_id"], "SUBMITTED")

    def _cancel_working_entries(self, rep: CycleReport) -> None:
        for o in self.ledger.live_orders():
            if o["purpose"] == "ENTRY":
                self.exec.cancel(o["client_order_id"])
                rep.add(f"cancelled working entry {o['symbol']}")

    # -- scheduler entry point ------------------------------------------------------------

    def tick(self) -> list[CycleReport]:
        """Call every few minutes. Decides by US/Eastern time what is due, so DST needs no crontab edits."""
        now = self.now()
        td = to_trading_date(now)
        local = now.astimezone(ET).time()
        reports: list[CycleReport] = []
        if not is_trading_day(td):
            return reports
        if is_regular_session(now):
            if local >= EXECUTE_AFTER and self._claim_cycle("execute", td):
                reports.append(self.execute())
            reports.append(self.monitor())
        elif local >= RESEARCH_AFTER and self._claim_cycle("research", td):
            reports.append(self.research())
            reports.append(self.monitor())
        return reports

    def _claim_cycle(self, kind: str, td: date) -> bool:
        key = f"{kind}:{td.isoformat()}"
        if self.ledger.has_event("cycle_run", key):
            return False
        self.ledger.append("cycle_run", {"kind": kind, "trading_date": td.isoformat()}, decision_id=key)
        return True

    # -- monitor ------------------------------------------------------------------------

    def monitor(self) -> CycleReport:
        rep = CycleReport("monitor")
        now = self.now()
        account = self._account()
        reconciled, issues = self.reconcile(account)
        if not reconciled:
            self.kill.engage("reconciliation failure: " + "; ".join(issues)[:300], by="monitor")
            rep.add("KILL SWITCH ENGAGED: " + "; ".join(issues))
        peak = self.ledger.peak_equity()
        if peak and (account.equity / peak - 1) * 100 <= -self.limits.account.max_drawdown_pct:
            self.kill.engage(f"drawdown limit breached: equity {account.equity:.2f} vs peak {peak:.2f}", by="monitor")
            rep.add("KILL SWITCH ENGAGED: drawdown limit")
        if self.kill.engaged():
            self._cancel_working_entries(rep)
        if not is_regular_session(now):
            rep.add("market closed: exits not evaluated")
            return rep

        trades = self.ledger.open_trades()
        quotes = self.broker.get_quotes(sorted({t["symbol"] for t in trades})) if trades else {}
        today = to_trading_date(now)
        for t in trades:
            q = quotes.get(t["symbol"])
            held = account.position(t["symbol"])
            if q is None or held is None:
                continue
            reason = None
            if q.last <= t["stop_loss"]:
                reason = "stop_breached"
            elif q.last >= t["take_profit"]:
                reason = "take_profit"
            elif today >= date.fromisoformat(t["time_stop_date"]):
                reason = "time_stop"
            if reason:
                qty = int(min(t["quantity"], held.quantity))
                state = self.exec.exit_trade(t["decision_id"], t["symbol"], qty, q.bid, reason, today)
                rep.add(f"{t['symbol']}: {reason} -> SELL {qty} {state.value}")
        rep.add(f"equity {account.equity:.2f}, {len(trades)} open system trades, reconciled={reconciled}")
        return rep
