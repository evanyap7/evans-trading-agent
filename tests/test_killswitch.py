"""The kill switch is the emergency stop: it must fail closed and never be blocked by alerting."""
from __future__ import annotations

import json

import pytest

from trading_agent import alerts
from trading_agent.killswitch import KillSwitch


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(alerts, "alert_killswitch", lambda text, engaged=True: calls.append((text, engaged)))
    return calls


def test_starts_disengaged(tmp_path):
    ks = KillSwitch(tmp_path)
    assert not ks.engaged()
    assert ks.reason() == ""


def test_engage_records_reason_and_alerts(tmp_path, sent):
    ks = KillSwitch(tmp_path)
    ks.engage("1% daily loss", by="risk")
    assert ks.engaged()
    assert ks.reason() == "1% daily loss"
    record = json.loads((tmp_path / "KILL").read_text())
    assert record["by"] == "risk" and record["at"]
    assert sent == [("1% daily loss (by risk)", True)]


def test_engage_creates_missing_state_dir(tmp_path):
    ks = KillSwitch(tmp_path / "nested" / "state")
    ks.engage("x", by="operator")
    assert ks.engaged()


def test_engage_is_idempotent_and_keeps_first_reason(tmp_path, sent):
    ks = KillSwitch(tmp_path)
    ks.engage("first", by="a")
    ks.engage("second", by="b")
    assert ks.reason() == "first"
    assert len(sent) == 1


def test_touching_the_file_by_hand_engages_it(tmp_path):
    (tmp_path / "KILL").touch()  # `touch state/KILL` is the documented manual path
    ks = KillSwitch(tmp_path)
    assert ks.engaged()
    assert ks.reason() == "engaged (no reason recorded)"  # empty file is not valid JSON


def test_corrupt_file_still_engaged(tmp_path):
    (tmp_path / "KILL").write_text("{not json")
    ks = KillSwitch(tmp_path)
    assert ks.engaged()
    assert "engaged" in ks.reason()


def test_release_clears_and_alerts(tmp_path, sent):
    ks = KillSwitch(tmp_path)
    ks.engage("x", by="a")
    ks.release()
    assert not ks.engaged()
    assert sent[-1] == ("", False)


def test_release_when_not_engaged_is_a_silent_noop(tmp_path, sent):
    KillSwitch(tmp_path).release()
    assert sent == []


def test_alert_failure_never_blocks_engage_or_release(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(alerts, "alert_killswitch", boom)
    ks = KillSwitch(tmp_path)
    ks.engage("x", by="a")
    assert ks.engaged()
    ks.release()
    assert not ks.engaged()
