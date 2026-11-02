"""Property tests for the NYSE calendar: invariants that must hold for every date, not a handful of examples."""

from datetime import date, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trading_agent.market_calendar import (
    CALENDAR_LAST_DAY,
    EARLY_CLOSES,
    ET,
    HOLIDAYS,
    add_trading_days,
    is_regular_session,
    is_trading_day,
    session_bounds,
)

covered_dates = st.dates(max_value=CALENDAR_LAST_DAY)
# Far enough from the calendar's end that 30 trading days still land inside it.
roomy_dates = st.dates(min_value=date(2026, 1, 1), max_value=date(2027, 10, 31))
trading_days = covered_dates.filter(is_trading_day)
MICRO = timedelta(microseconds=1)


@given(covered_dates)
def test_weekends_are_never_trading_days(d):
    if d.weekday() >= 5:
        assert not is_trading_day(d)


@given(st.sampled_from(sorted(HOLIDAYS)))
def test_listed_holidays_are_never_trading_days(d):
    assert not is_trading_day(d)


@given(st.dates(min_value=CALENDAR_LAST_DAY + timedelta(days=1)))
def test_dates_past_the_calendar_raise_instead_of_guessing(d):
    with pytest.raises(RuntimeError):
        is_trading_day(d)


@given(covered_dates)
def test_session_exists_exactly_on_trading_days(d):
    assert (session_bounds(d) is not None) == is_trading_day(d)


@given(trading_days)
def test_session_is_ordered_and_closes_early_only_when_listed(d):
    opens, closes = session_bounds(d)
    assert opens < closes
    assert opens.tzinfo is not None and closes.tzinfo is not None
    assert (closes.hour, closes.minute) == ((13, 0) if d in EARLY_CLOSES else (16, 0))


@given(trading_days)
def test_session_boundaries_are_half_open(d):
    opens, closes = session_bounds(d)
    assert is_regular_session(opens)
    assert is_regular_session(closes - MICRO)
    assert not is_regular_session(opens - MICRO)
    assert not is_regular_session(closes)


@given(trading_days, st.integers(min_value=0, max_value=24 * 60 * 60 - 1))
def test_session_answer_does_not_depend_on_the_timezone_it_is_written_in(d, second_of_day):
    local = datetime.combine(d, datetime.min.time(), ET) + timedelta(seconds=second_of_day)
    assert is_regular_session(local) == is_regular_session(local.astimezone(timezone.utc))


@given(roomy_dates, st.integers(min_value=1, max_value=30))
def test_add_trading_days_lands_on_a_trading_day_exactly_n_sessions_later(d, n):
    out = add_trading_days(d, n)
    assert out > d
    assert is_trading_day(out)
    between = [d + timedelta(days=i) for i in range(1, (out - d).days + 1)]
    assert sum(is_trading_day(x) for x in between) == n


@given(roomy_dates, st.integers(min_value=0, max_value=15), st.integers(min_value=0, max_value=15))
def test_add_trading_days_is_additive(d, a, b):
    assert add_trading_days(add_trading_days(d, a), b) == add_trading_days(d, a + b)
