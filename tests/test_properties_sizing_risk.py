"""Property tests for sizing, the trailing stop, the risk engine and the cash sweep.

The sizer must never hand the risk engine an order the risk engine's own size checks would refuse, whatever the
limits, account or proposal look like. The limits are generated, not pinned to the shipped YAML.
"""

from datetime import datetime

import pytest
from hypothesis import assume, event, given
from hypothesis import strategies as st

from trading_agent.config import Events, load_risk_limits, load_universe
from trading_agent.market_calendar import ET
from trading_agent.portfolio import kelly_risk_fraction, size_order, trailed_stop, trailed_stop_short
from trading_agent.risk import OpenRisk, RiskContext, evaluate
from trading_agent.schemas import AccountState, Position, SizedOrder, TradeProposal
from trading_agent.sweep import SweepState, trading_view

BASE = load_risk_limits()
UNIVERSE = load_universe()
NOW = datetime(2026, 9, 22, 10, 30, tzinfo=ET)
SYMBOL = "XLK"
assert UNIVERSE.get(SYMBOL) is not None

pct = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# ---------------------------------------------------------------- generated configuration

@st.composite
def risk_limits(draw):
    a = BASE.account.model_copy(update=dict(
        max_position_pct=draw(st.floats(0.5, 100)),
        max_risk_per_trade_pct=draw(st.floats(0.05, 5)),
        max_order_value_usd=draw(st.floats(50, 1_000_000)),
        max_adv_participation_pct=draw(st.floats(0.1, 5)),
        max_total_exposure_pct=draw(st.floats(10, 100)),
    ))
    ex = BASE.execution.model_copy(update=dict(
        slippage_bps=draw(st.floats(0, 50)), fee_per_order_usd=draw(st.floats(0, 20))))
    sig = BASE.signal.model_copy(update=dict(kelly_fraction=draw(st.floats(0.05, 1))))
    return BASE.model_copy(update=dict(account=a, execution=ex, signal=sig))


@st.composite
def execution_limits(draw):
    start = draw(st.floats(0.1, 10))
    return BASE.execution.model_copy(update=dict(
        trailing_stops=draw(st.booleans()),
        trail_breakeven_r=draw(st.floats(0.1, 5)),
        trail_start_r=start,
        trail_distance_r=draw(st.floats(0.05, 1)) * start,  # the config validator forbids distance > start
        trail_min_step_r=draw(st.floats(0.01, 2)),
    ))


@st.composite
def accounts(draw, min_cash=0.0):
    equity = draw(st.floats(100, 10_000_000))
    cash = draw(st.floats(min_cash, equity))
    return AccountState(account_id="A", equity=equity, cash=cash, buying_power=cash, positions=[],
                        open_orders=[], as_of=NOW)


@st.composite
def proposals(draw):
    short = draw(st.booleans())
    # US equities at or above $1 tick in whole cents (Reg NMS Rule 612), so every price here is a cent multiple.
    limit_c = draw(st.integers(100, 500_000))
    stop_gap_c = draw(st.integers(1, max(1, limit_c // 2)))
    target_gap_c = draw(st.integers(1, max(1, limit_c * 99 // 100)))
    limit = limit_c / 100
    stop = (limit_c + stop_gap_c) / 100 if short else (limit_c - stop_gap_c) / 100
    target = (limit_c - target_gap_c) / 100 if short else (limit_c + target_gap_c) / 100
    return TradeProposal(
        symbol=SYMBOL, instrument_type="ETF", side="SELL_SHORT" if short else "BUY", strategy="generated",
        thesis="A generated thesis long enough to validate.", holding_period_days=draw(st.integers(1, 40)),
        confidence=draw(pct), expected_return_pct=draw(st.floats(-50, 50)), invalidation_price=stop,
        entry={"order_type": "LIMIT", "limit_price": limit},
        exit={"take_profit": target, "stop_loss": stop, "time_stop_days": 10},
        requested_risk_pct=draw(st.floats(0.01, 2.5)), evidence_ids=["e1"])


adv = st.floats(0, 1e9)


def ctx_for(account, **kw):
    base = dict(now=NOW, trading_date=NOW.date(), account=account, quote=None, kill_switch_engaged=False,
                sends_real_orders_to_prod=False, require_session=False, reconciled=True, entries_today=0,
                start_of_day_equity=None, peak_equity=None, decision_already_executed=False)
    base.update(kw)
    return RiskContext(**base)


# ---------------------------------------------------------------- Kelly

@given(st.floats(0, 1), st.floats(0.01, 1e4), st.floats(0.01, 1e4))
def test_kelly_is_positive_exactly_when_confidence_beats_break_even(p, up, down):
    break_even = down / (up + down)
    f = kelly_risk_fraction(p, up, down)
    assume(abs(p - break_even) > 1e-9)  # at the boundary the sign is decided by rounding
    assert (f > 0) == (p > break_even)
    assert f <= p  # you never stake more than the win probability


@given(st.floats(0, 1), st.floats(0, 1), st.floats(0.01, 1e4), st.floats(0.01, 1e4))
def test_kelly_grows_with_confidence(a, b, up, down):
    lo, hi = sorted((a, b))
    assert kelly_risk_fraction(lo, up, down) <= kelly_risk_fraction(hi, up, down)


@given(st.floats(0, 1), st.floats(max_value=0.0, allow_nan=False), st.floats(0.01, 1e4))
def test_kelly_without_upside_or_downside_is_zero(p, bad, other):
    assert kelly_risk_fraction(p, bad, other) == 0.0
    assert kelly_risk_fraction(p, other, bad) == 0.0


# ---------------------------------------------------------------- sizing

@given(proposals(), accounts(), adv, risk_limits())
def test_sized_order_respects_every_documented_limit(p, account, avg_volume, limits):
    out = size_order("d1", p, account, avg_volume, limits)
    event("sized" if isinstance(out, SizedOrder) else "refused")
    if not isinstance(out, SizedOrder):
        assert out  # a refusal always says why
        return
    a, ex = limits.account, limits.execution
    limit = p.entry.limit_price
    assert out.quantity >= 1
    assert out.quantity * limit <= a.max_order_value_usd + 1e-6
    assert out.quantity * limit <= account.equity * a.max_position_pct / 100 + 1e-6
    assert out.quantity * limit + ex.fee_per_order_usd <= account.cash + 1e-6
    assert out.quantity <= avg_volume * a.max_adv_participation_pct / 100 + 1e-9


@given(proposals(), accounts(), adv, risk_limits())
def test_anything_the_sizer_emits_passes_the_risk_engines_size_checks(p, account, avg_volume, limits):
    out = size_order("d1", p, account, avg_volume, limits)
    event("sized" if isinstance(out, SizedOrder) else "refused")
    if not isinstance(out, SizedOrder):
        return
    d = evaluate(out, "ETF", p.holding_period_days, ctx_for(account), limits, UNIVERSE, Events())
    size_checks = {"order_value_ok", "position_pct_ok", "risk_per_trade_ok", "cash_covers_order"}
    assert not size_checks & set(d.failed_checks), d.failed_checks


def test_a_clear_edge_on_a_funded_account_is_sized():
    """Guards the conditional properties above against passing only because everything is refused."""
    p = TradeProposal(
        symbol=SYMBOL, instrument_type="ETF", side="BUY", strategy="generated", thesis="A clear 3:1 setup, 70% sure.",
        holding_period_days=10, confidence=0.7, expected_return_pct=3.0, invalidation_price=98.0,
        entry={"order_type": "LIMIT", "limit_price": 100.0}, exit={"take_profit": 106.0, "stop_loss": 98.0,
                                                                    "time_stop_days": 10},
        requested_risk_pct=0.5, evidence_ids=["e1"])
    account = AccountState(account_id="A", equity=100_000, cash=100_000, buying_power=100_000, positions=[],
                           open_orders=[], as_of=NOW)
    assert isinstance(size_order("d1", p, account, 5_000_000, BASE), SizedOrder)


@given(proposals(), accounts(), adv, risk_limits(), st.floats(0, 1e7))
def test_more_cash_never_buys_fewer_shares(p, account, avg_volume, limits, extra):
    richer = account.model_copy(update=dict(cash=account.cash + extra, buying_power=account.buying_power + extra))
    poor = size_order("d1", p, account, avg_volume, limits)
    rich = size_order("d1", p, richer, avg_volume, limits)
    poor_qty = poor.quantity if isinstance(poor, SizedOrder) else 0
    rich_qty = rich.quantity if isinstance(rich, SizedOrder) else 0
    assert rich_qty >= poor_qty


@given(proposals(), st.floats(min_value=-1e6, max_value=0, allow_nan=False), adv, risk_limits())
def test_nothing_is_sized_for_an_account_without_equity(p, equity, avg_volume, limits):
    broke = AccountState(account_id="A", equity=equity, cash=0, buying_power=0, positions=[], open_orders=[],
                         as_of=NOW)
    assert isinstance(size_order("d1", p, broke, avg_volume, limits), str)


# ---------------------------------------------------------------- trailing stop

price = st.floats(0.01, 1e5)


@given(price, price, price, price, execution_limits())
def test_a_long_stop_only_ever_moves_up_and_stays_below_the_market(entry, initial, current, last, ex):
    new = trailed_stop(entry, initial, current, last, ex)
    if new is not None:
        assert current < new < last


@given(price, price, price, price, execution_limits())
def test_a_short_stop_only_ever_moves_down_and_stays_above_the_market(entry, initial, current, last, ex):
    new = trailed_stop_short(entry, initial, current, last, ex)
    if new is not None:
        assert last < new < current


@given(price, st.floats(0.001, 0.5), st.lists(st.floats(0.5, 3), min_size=1, max_size=40), execution_limits())
def test_ratcheting_through_a_price_path_never_loosens_the_stop(entry, risk_frac, path, ex):
    initial = entry * (1 - risk_frac)
    stop = initial
    for mult in path:
        new = trailed_stop(entry, initial, stop, entry * mult, ex)
        if new is not None:
            assert new > stop
            stop = new
    assert stop >= initial


@given(*[st.floats(-1e5, 1e5)] * 4, execution_limits())
def test_short_trailing_is_the_exact_mirror_of_long_trailing(entry, initial, current, last, ex):
    long_side = trailed_stop(-entry, -initial, -current, -last, ex)
    short_side = trailed_stop_short(entry, initial, current, last, ex)
    assert short_side == (None if long_side is None else -long_side)


@given(price, price, price, price, execution_limits())
def test_a_disabled_trail_never_moves_a_stop(entry, initial, current, last, ex):
    off = ex.model_copy(update={"trailing_stops": False})
    assert trailed_stop(entry, initial, current, last, off) is None
    assert trailed_stop_short(entry, initial, current, last, off) is None


# ---------------------------------------------------------------- risk engine

@st.composite
def sized_orders(draw, short=None):
    short = draw(st.booleans()) if short is None else short
    limit = draw(st.floats(1, 5000))
    qty = draw(st.integers(1, 100_000))
    return SizedOrder(decision_id="d", symbol=SYMBOL, side="SELL_SHORT" if short else "BUY", quantity=qty,
                      limit_price=limit, stop_loss=limit * (1.1 if short else 0.9),
                      take_profit=limit * (0.8 if short else 1.2), notional=qty * limit,
                      risk_usd=qty * limit * draw(st.floats(0.001, 0.2)))


@st.composite
def open_risks(draw):
    out = []
    for sym in draw(st.lists(st.sampled_from(["AAPL", "MSFT", "NVDA", "XLF", "XLE"]), unique=True, max_size=4)):
        qty = draw(st.floats(1, 5000)) * draw(st.sampled_from([1, -1]))
        px = draw(st.floats(1, 1000))
        out.append(OpenRisk(sym, qty, px, draw(st.none() | st.floats(0.5, 2) .map(lambda m, px=px: px * m))))
    return out


@given(sized_orders(), accounts(), open_risks(), st.booleans())
def test_approval_is_exactly_the_absence_of_failed_checks(order, account, risks, permit_shorting):
    limits = BASE.model_copy(update={"live_trading": BASE.live_trading.model_copy(
        update={"permit_shorting": permit_shorting})})
    d = evaluate(order, "ETF", 10, ctx_for(account, open_risks=risks), limits, UNIVERSE, Events())
    assert d.approved == (not d.failed_checks)
    assert not set(d.failed_checks) & set(d.passed_checks)


@given(sized_orders(), accounts(), open_risks())
def test_an_engaged_kill_switch_always_blocks(order, account, risks):
    d = evaluate(order, "ETF", 10, ctx_for(account, open_risks=risks, kill_switch_engaged=True), BASE, UNIVERSE,
                 Events())
    assert not d.approved and "kill_switch_off" in d.failed_checks


@given(sized_orders(short=False), accounts(), open_risks(), st.floats(1, 20))
def test_a_bigger_order_never_fixes_a_size_check_that_a_smaller_one_failed(order, account, risks, factor):
    bigger = order.model_copy(update=dict(
        quantity=order.quantity, notional=order.notional * factor, risk_usd=order.risk_usd * factor))
    ctx = ctx_for(account, open_risks=risks)
    small = set(evaluate(order, "ETF", 10, ctx, BASE, UNIVERSE, Events()).failed_checks)
    big = set(evaluate(bigger, "ETF", 10, ctx, BASE, UNIVERSE, Events()).failed_checks)
    scaling = {"order_value_ok", "position_pct_ok", "sector_pct_ok", "gross_exposure_ok", "total_exposure_ok",
               "risk_per_trade_ok", "portfolio_risk_ok", "cash_covers_order"}
    assert (small & scaling) <= big


@given(sized_orders(short=True), accounts())
def test_shorting_is_blocked_unless_the_limits_permit_it(order, account):
    assert not BASE.live_trading.permit_shorting
    d = evaluate(order, "ETF", 10, ctx_for(account), BASE, UNIVERSE, Events())
    assert not d.approved and "long_only" in d.failed_checks


# ---------------------------------------------------------------- cash sweep

@st.composite
def sweep_cases(draw):
    last = draw(st.floats(1, 1000))
    qty = draw(st.integers(0, 10_000))
    held = draw(st.floats(max(qty, 0.001), qty + 5000))  # a position the broker shows is never empty
    others = [Position(symbol=s, quantity=draw(st.floats(1, 1000)), avg_cost=10, last_price=draw(st.floats(1, 500)))
              for s in draw(st.lists(st.sampled_from(["AAPL", "MSFT", "NVDA"]), unique=True, max_size=3))]
    sweep_pos = Position(symbol="SGOV", quantity=held, avg_cost=last, last_price=last)
    cash = draw(st.floats(0, 1e6))
    account = AccountState(account_id="A", equity=cash + held * last, cash=cash, buying_power=cash,
                           positions=[*others, sweep_pos], open_orders=[], as_of=NOW)
    return account, SweepState(ledger_qty=qty, qty=qty, net_cost=draw(st.floats(0, 1e6)), last=last)


def total_value(a: AccountState) -> float:
    return a.cash + sum(p.market_value for p in a.positions)


@given(sweep_cases())
def test_the_trading_view_moves_value_from_sweep_shares_into_cash_and_loses_none(case):
    account, state = case
    view = trading_view(account, state, "SGOV")
    assert view.cash >= account.cash and view.buying_power >= account.buying_power
    assert view.equity == account.equity
    assert total_value(view) == pytest.approx(total_value(account), rel=1e-9)
    assert all(p.quantity > 0 for p in view.positions)
    assert {p.symbol: p for p in view.positions if p.symbol != "SGOV"} == {
        p.symbol: p for p in account.positions if p.symbol != "SGOV"}


@given(sweep_cases())
def test_a_sweep_with_no_shares_leaves_the_account_untouched(case):
    account, state = case
    empty = SweepState(ledger_qty=0, qty=0, net_cost=state.net_cost, last=state.last)
    assert trading_view(account, empty, "SGOV") == account
