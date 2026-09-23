"""Pure rules for the ratcheting stop."""

import pytest

from trading_agent.config import ExecutionLimits, load_risk_limits
from trading_agent.portfolio import trailed_stop

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
