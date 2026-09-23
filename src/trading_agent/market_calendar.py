"""US equity regular-session calendar (NYSE). Holidays must be extended each year."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}
EARLY_CLOSES = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)}
CALENDAR_LAST_DAY = date(2027, 12, 31)


def to_trading_date(ts: datetime) -> date:
    return ts.astimezone(ET).date()


def is_trading_day(d: date) -> bool:
    if d > CALENDAR_LAST_DAY:
        raise RuntimeError("market calendar expired: add next year's NYSE holidays to market_calendar.py")
    return d.weekday() < 5 and d not in HOLIDAYS


def session_bounds(d: date) -> tuple[datetime, datetime] | None:
    if not is_trading_day(d):
        return None
    close = EARLY_CLOSE if d in EARLY_CLOSES else CLOSE
    return datetime.combine(d, OPEN, ET), datetime.combine(d, close, ET)


def is_regular_session(ts: datetime) -> bool:
    bounds = session_bounds(to_trading_date(ts))
    return bounds is not None and bounds[0] <= ts.astimezone(ET) < bounds[1]


def add_trading_days(d: date, n: int) -> date:
    out = d
    while n > 0:
        out += timedelta(days=1)
        if is_trading_day(out):
            n -= 1
    return out
