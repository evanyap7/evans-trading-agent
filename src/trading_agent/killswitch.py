"""Operator and automatic kill switch.

A file on disk, so it can be flipped from outside the trading process
(`trading-agent kill`, or simply `touch state/KILL`). While it exists no new
orders are sent; the execution service cancels open entry orders on its next pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import utcnow


class KillSwitch:
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "KILL"

    def engaged(self) -> bool:
        return self.path.exists()

    def reason(self) -> str:
        if not self.engaged():
            return ""
        try:
            return json.loads(self.path.read_text()).get("reason", "")
        except (ValueError, OSError):
            return "engaged (no reason recorded)"

    def engage(self, reason: str, by: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.engaged():
            self.path.write_text(json.dumps({"reason": reason, "by": by, "at": utcnow().isoformat()}))
            try:
                from .alerts import alert_killswitch
                alert_killswitch(f"{reason} (by {by})", engaged=True)
            except Exception:
                pass

    def release(self) -> None:
        was_engaged = self.engaged()
        self.path.unlink(missing_ok=True)
        if was_engaged:
            try:
                from .alerts import alert_killswitch
                alert_killswitch("", engaged=False)
            except Exception:
                pass
