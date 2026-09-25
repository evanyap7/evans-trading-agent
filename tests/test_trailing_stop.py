"""Pure rules for the ratcheting stop."""

import pytest

from trading_agent.config import ExecutionLimits, load_risk_limits
from trading_agent.portfolio import trailed_stop, trailed_stop_short

EX = load_risk_limits().execution  # breakeven 1R, trail 1.5R from 2R, min step 0.25R


def test_no_move_before_breakeven_trigger():
    assert trailed_stop(100, 96, 96, 103.9, EX) is None  # +0.975R


def test_breakeven_at_one_r():
    assert trailed_stop(100, 96, 96, 104, EX) == 100


def test_trails_behind_price_from_two_r():
    assert trailed_stop(100, 96, 100, 110, EX) == 104  # +2.5R: 110 - 1.5 * 4


def test_never_moves_down():
    assert trailed_stop(100, 96, 105, 108, EX) is None  # trail would be 102, below the current stop


def test_skips_tiny_steps():
    assert trailed_stop(100, 96, 104, 110.5, EX) is None  # would move 0.5 < 0.25R (1.0)
    assert trailed_stop(100, 96, 104, 111.2, EX) == 105.2


def test_disabled_or_degenerate():
    off = EX.model_copy(update={"trailing_stops": False})
    assert trailed_stop(100, 96, 96, 110, off) is None
    assert trailed_stop(100, 100, 100, 110, EX) is None  # zero initial risk
    assert trailed_stop(100, 96, 96, 99, EX) is None     # losing position


def test_trail_wider_than_start_is_rejected():
    fields = EX.model_dump() | {"trail_start_r": 1.0, "trail_distance_r": 1.5}
    with pytest.raises(ValueError):
        ExecutionLimits(**fields)


# -- Short trailing stop tests --------------------------------------------------

def test_short_no_move_before_breakeven():
    # Short: entry=100, initial_stop=104 (R=4). Gain at 96.1 is 3.9 (0.975R)
    assert trailed_stop_short(100, 104, 104, 96.1, EX) is None


def test_short_breakeven_at_one_r():
    # Gain at 96 is 4 (+1.0R) -> stop moves to 100
    assert trailed_stop_short(100, 104, 104, 96, EX) == 100


def test_short_trails_behind_price_from_two_r():
    # Gain at 90 is 10 (+2.5R): candidate = 90 + 1.5 * 4 = 96
    assert trailed_stop_short(100, 104, 100, 90, EX) == 96


def test_short_never_moves_up():
    # Current stop at 95, candidate 98 -> should return None (only ratchets down)
    assert trailed_stop_short(100, 104, 95, 92, EX) is None


def test_short_skips_tiny_steps():
    # Current stop 96, candidate 95.5 -> move 0.5 < 0.25R (1.0) -> None
    assert trailed_stop_short(100, 104, 96, 89.5, EX) is None
    # Candidate 94.8 -> move 1.2 >= 1.0 -> 94.8
    assert trailed_stop_short(100, 104, 96, 88.8, EX) == 94.8


def test_short_disabled_or_degenerate():
    off = EX.model_copy(update={"trailing_stops": False})
    assert trailed_stop_short(100, 104, 104, 90, off) is None
    assert trailed_stop_short(100, 100, 100, 90, EX) is None
    assert trailed_stop_short(100, 104, 104, 101, EX) is None  # losing position (above entry)

