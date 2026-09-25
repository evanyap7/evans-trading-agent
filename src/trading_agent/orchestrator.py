"""The trading cycles.

research  (after the US close)   evidence -> agent -> verify -> size -> risk -> pending actions
execute   (early regular session) fresh quotes -> re-verify -> re-size -> full risk -> preview -> submit
monitor   (every few minutes)     reconcile -> circuit breakers -> stops / targets / time stops
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable

from .agents import AgentContext, ResearchAgent
from .broker.base import Broker, OrderRequest
from .config import Events, RiskLimits, Settings, TradingMode, Universe
from .execution import ExecutionEngine, client_order_id
from .features import build_evidence
from .killswitch import KillSwitch
from .ledger import Ledger
from .market_calendar import ET, is_regular_session, is_trading_day, to_trading_date
from .portfolio import size_order, trailed_stop
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
                 ledger: Ledger, agent: ResearchAgent, now: Callable[[], datetime] = utcnow, enable_news: bool = True,
                 continuous_trading: bool | None = None):
        self.settings = settings
        self.limits = limits
        self.universe = universe
        self.events = events
        self._manual_events = events  # config/events.yaml: always wins over broker data
        self.broker = broker
        self.ledger = ledger
        self.agent = agent
        self.now = now
        self.enable_news = enable_news
        self.continuous_trading = (
            continuous_trading if continuous_trading is not None else settings.continuous_trading
        )
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

    def _refresh_earnings(self, symbols, rep: CycleReport) -> None:
        """Merge broker earnings dates for stocks into the risk engine's events. On failure the stocks
        simply stay unknown, which blocks their entries while require_earnings_data_for_stocks is on."""
        fetch = getattr(self.broker, "get_earnings_dates", None)
        stocks = sorted(s for s in symbols if (sec := self.universe.get(s)) and sec.type == "EQUITY")
        if fetch is None or not stocks:
            return
        try:
            fetched = fetch(stocks, to_trading_date(self.now()))
        except Exception as e:
            rep.add(f"earnings calendar unavailable: {e}")
            return
        self.events = Events(earnings={**fetched, **self._manual_events.earnings})
        missing = [s for s in stocks if s not in self.events.earnings]
        if missing:
            rep.add(f"no earnings date for {', '.join(missing)} (new entries blocked)")

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

        held_equity_syms = {p.symbol for p in account.positions if p.symbol.isalpha() and len(p.symbol) <= 5}
        symbols = sorted(set(self.universe.symbols) | held_equity_syms)
        bars = self.broker.get_daily_bars(symbols, BAR_HISTORY)
        quotes = self.broker.get_quotes(symbols)
        evidence, features = build_evidence(bars, quotes, self.universe.regime_benchmark)
        self._refresh_earnings(symbols, rep)

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

        cycle_ev = {e.evidence_id: e.symbol for e in evidence}
        cycle_ev.update(self.ledger.evidence_ids(cycle_id))
        freed_cash = 0.0
        for ex in output.exits:
            held = account.position(ex.symbol)
            # Every id must exist, and at least one must be our own price/position data for this symbol:
            # a news article alone (untrusted, injectable text) can never trigger a sale.
            grounded = all(e in cycle_ev for e in ex.evidence_ids) and any(
                cycle_ev.get(e) == ex.symbol and e.startswith(("px_", "pos_")) for e in ex.evidence_ids)
            own_trade = ex.symbol in trades
            if held and grounded and not own_trade and not self.limits.live_trading.agent_may_close_manual_positions:
                rep.add(f"exit for {ex.symbol} ignored: not a system trade and agent_may_close_manual_positions is off")
                continue
            if held and grounded:
                did = trades[ex.symbol]["decision_id"] if ex.symbol in trades else f"portfolio_{ex.symbol}"
                self.ledger.add_pending_action(f"close:{did}:{cycle_id[:8]}", cycle_id,
                                               ex.model_dump_json())
                rep.add(f"exit queued for {ex.symbol}: {ex.reason[:80]}")
                q = quotes.get(ex.symbol)
                px = self._exit_price(q) or (held.last_price if held.last_price > 0 else held.avg_cost)
                freed_cash += held.quantity * px
            else:
                rep.add(f"exit for {ex.symbol} ignored (held={bool(held)}, grounded={grounded})")

        entry_account = account.model_copy(update={"cash": account.cash + freed_cash}) if freed_cash > 0 else account
        for trade in output.proposals:
            self._consider_entry(trade, cycle_id, cycle_ev, features, quotes, entry_account, reconciled, rep)
        try:
            from .alerts import alert_research_summary
            alert_research_summary(to_trading_date(self.now()).isoformat(), len(output.proposals), len(evidence), rep.notes)
        except Exception:
            pass
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
        try:
            from .alerts import alert_trade_queued
            alert_trade_queued(trade.symbol, sized.quantity, sized.limit_price, sized.stop_loss, sized.take_profit, trade.thesis)
        except Exception:
            pass

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
        self._refresh_earnings(symbols, rep)
        today = to_trading_date(now)

        closes = [r for r in pending if r["decision_id"].startswith("close:")]
        entries = [r for r in pending if not r["decision_id"].startswith("close:")]

        for row in closes:
            self._execute_close(row, account, quotes, today, rep)

        if closes:
            import time
            time.sleep(2)
            account = self._account()
            reconciled, issues = self.reconcile(account)

        for row in entries:
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
            try:
                from .alerts import alert_order_submitted
                alert_order_submitted(t.symbol, "BUY", sized.quantity, sized.limit_price, is_shadow=not self.sends_real_orders_to_prod)
            except Exception:
                pass
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
        parts = row["decision_id"].split(":")
        did = parts[1]
        trade = next((t for t in self.ledger.open_trades() if t["decision_id"] == did), None)
        symbol = trade["symbol"] if trade else self._pending_symbol(row)
        q = quotes.get(symbol)
        if q is None:
            self.ledger.set_pending_status(row["decision_id"], "DROPPED")
            return
        held = account.position(symbol)
        if held is None or held.quantity <= 0:
            self.ledger.set_pending_status(row["decision_id"], "DROPPED")
            return
        if trade is None and not self.limits.live_trading.agent_may_close_manual_positions:
            self.ledger.set_pending_status(row["decision_id"], "DROPPED")
            rep.add(f"{symbol}: agent exit dropped (manual position, agent_may_close_manual_positions is off)")
            return
        ref = self._exit_price(q)
        if ref is None:
            self.ledger.set_pending_status(row["decision_id"], "DROPPED")
            rep.add(f"{symbol}: agent exit dropped (no fresh usable price)")
            return
        qty = int(min(trade["quantity"] if trade else held.quantity, held.quantity))
        if qty > 0:
            state = self.exec.exit_trade(did, symbol, qty, ref, "agent_thesis_exit", today)
            rep.add(f"{symbol}: agent exit SELL {qty} -> {state.value}")
            try:
                from .alerts import alert_trade_exited
                alert_trade_exited(symbol, qty, ref, f"agent_thesis_exit ({state.value})")
            except Exception:
                pass
        self.ledger.set_pending_status(row["decision_id"], "SUBMITTED")

    def _cancel_working_entries(self, rep: CycleReport) -> None:
        for o in self.ledger.live_orders():
            if o["purpose"] == "ENTRY":
                self.exec.cancel(o["client_order_id"])
                rep.add(f"cancelled working entry {o['symbol']}")

    # -- scheduler entry point ------------------------------------------------------------

    def morning_briefing(self) -> CycleReport:
        rep = CycleReport("morning_briefing")
        account = self._account()
        try:
            from .alerts import send_daily_morning_briefing
            ok = send_daily_morning_briefing(account, self.ledger)
            rep.add(f"morning briefing dispatched: {ok}")
        except Exception as e:
            rep.add(f"morning briefing error: {e}")
        return rep

    def tick(self) -> list[CycleReport]:
        """Call every few minutes. Decides by US/Eastern time what is due, so DST needs no crontab edits."""
        now = self.now()
        td = to_trading_date(now)
        local = now.astimezone(ET).time()
        reports: list[CycleReport] = []

        # 9:00 AM Singapore Time briefing check (SGT is UTC+8)
        sgt_now = now.astimezone(timezone(timedelta(hours=8)))
        if sgt_now.hour == 9 and self._claim_cycle("morning_briefing", sgt_now.date()):
            reports.append(self.morning_briefing())

        if not is_trading_day(td):
            return reports
        if is_regular_session(now):
            if self.ledger.pending():
                reports.append(self._guarded("execute", self.execute))
            elif local >= EXECUTE_AFTER and self._claim_cycle("execute", td):
                reports.append(self._guarded("execute", self.execute))

            # Continuous intraday trading: a fresh scan every scan_interval_minutes during regular hours
            if self.continuous_trading:
                interval = self.settings.scan_interval_minutes
                intraday_slot = f"{td.isoformat()}:{local.hour}:{local.minute // interval}"
                if local >= EXECUTE_AFTER and self._claim_cycle("intraday_trade", intraday_slot):
                    reports.append(self._guarded("research", self.research))
                    if self.ledger.pending():
                        reports.append(self._guarded("execute", self.execute))

            reports.append(self._guarded("monitor", self.monitor))
        elif local >= RESEARCH_AFTER and self._claim_cycle("research", td):
            reports.append(self._guarded("research", self.research))
            reports.append(self._guarded("monitor", self.monitor))
        return reports

    def _guarded(self, kind: str, cycle: Callable[[], CycleReport]) -> CycleReport:
        """A crash in one cycle (e.g. execute) must never stop the monitor pass that protects open positions."""
        try:
            return cycle()
        except Exception as e:
            self.ledger.append("cycle_error", {"kind": kind, "error": f"{type(e).__name__}: {e}"[:500]})
            try:
                from .alerts import alert_cycle_error
                alert_cycle_error(kind, f"{type(e).__name__}: {e}")
            except Exception:
                pass
            rep = CycleReport(kind)
            rep.add(f"CYCLE FAILED: {type(e).__name__}: {e}")
            return rep

    MAX_EXIT_QUOTE_AGE_SECONDS = 900

    def _exit_price(self, q) -> float | None:
        """Reference price for a sell: the bid, else last. None if the quote is too old to act on."""
        if q is None or q.age_seconds(self.now()) > self.MAX_EXIT_QUOTE_AGE_SECONDS:
            return None
        return q.bid if q.bid > 0 else (q.last if q.last > 0 else None)

    def _claim_cycle(self, kind: str, td_or_slot: date | str) -> bool:
        slot_str = td_or_slot.isoformat() if isinstance(td_or_slot, date) else str(td_or_slot)
        key = f"{kind}:{slot_str}"
        if self.ledger.has_event("cycle_run", key):
            return False
        self.ledger.append("cycle_run", {"kind": kind, "trading_date": slot_str}, decision_id=key)
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
            if held is None:
                continue
            if self.exec.send_orders and self.limits.execution.broker_side_stops:
                self.exec.ensure_protective_stop(t["decision_id"], t["symbol"], int(min(t["quantity"], held.quantity)),
                                                 t["stop_loss"])
            ref = self._exit_price(q)
            if ref is None:
                rep.add(f"{t['symbol']}: no fresh quote; software exits skipped (broker stop still active)")
                continue
            qty = int(min(t["quantity"], held.quantity))
            reason = None
            if q.last <= t["stop_loss"]:
                reason = "stop_breached"
            elif q.last >= t["take_profit"]:
                reason = "take_profit"
            elif today >= date.fromisoformat(t["time_stop_date"]):
                reason = "time_stop"
            if reason is None:
                initial = t["initial_stop"] if t["initial_stop"] is not None else t["stop_loss"]
                new_stop = trailed_stop(t["entry_price"], initial, t["stop_loss"], q.last, self.limits.execution)
                if new_stop is not None:
                    outcome = self.exec.raise_protective_stop(t["decision_id"], t["symbol"], qty, new_stop, current_price=q.last)
                    rep.add(f"{t['symbol']}: trail stop {t['stop_loss']} -> {new_stop}: {outcome}")
            if reason:
                state = self.exec.exit_trade(t["decision_id"], t["symbol"], qty, ref, reason, today)
                rep.add(f"{t['symbol']}: {reason} -> SELL {qty} {state.value}")
                try:
                    from .alerts import alert_trade_exited
                    alert_trade_exited(t["symbol"], qty, ref, f"{reason} ({state.value})")
                except Exception:
                    pass
        rep.add(f"equity {account.equity:.2f}, {len(trades)} open system trades, reconciled={reconciled}")
        return rep
