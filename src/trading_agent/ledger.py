"""Append-only audit ledger plus small current-state projections.

`events` and `evidence` are insert-only (enforced by SQLite triggers) and
hash-chained so tampering is detectable. `orders`, `pending_actions` and
`trades` are mutable projections that can always be rebuilt from `events`
plus broker history.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from .market_calendar import to_trading_date
from .schemas import Evidence, OrderState, TERMINAL_STATES, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    decision_id TEXT,
    client_order_id TEXT,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    as_of TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL,
    untrusted_text INTEGER NOT NULL,
    checksum TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;

CREATE TABLE IF NOT EXISTS pending_actions (
    decision_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    proposal TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    limit_price REAL,
    stop_price REAL,
    time_in_force TEXT NOT NULL,
    state TEXT NOT NULL,
    broker_order_id TEXT,
    filled_quantity REAL NOT NULL DEFAULT 0,
    filled_price REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    decision_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    time_stop_date TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL,
    closed_at TEXT,
    exit_reason TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    equity REAL NOT NULL,
    sweep_pnl REAL NOT NULL DEFAULT 0
);
"""

GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Ledger:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(trades)")}
        if "initial_stop" not in cols:  # trades opened before trailing stops: current stop = initial stop
            self.db.execute("ALTER TABLE trades ADD COLUMN initial_stop REAL")
        if "side" not in cols:  # trades opened before long/short support default to long
            self.db.execute("ALTER TABLE trades ADD COLUMN side TEXT DEFAULT 'BUY'")
        if "sweep_pnl" not in {r["name"] for r in self.db.execute("PRAGMA table_info(equity_snapshots)")}:
            self.db.execute("ALTER TABLE equity_snapshots ADD COLUMN sweep_pnl REAL NOT NULL DEFAULT 0")

    # -- append-only event log ------------------------------------------------

    def append(self, kind: str, payload: dict, decision_id: str | None = None, client_order_id: str | None = None) -> str:
        ts = utcnow().isoformat()
        body = _canonical(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            digest = hashlib.sha256(_canonical([prev, ts, kind, decision_id, client_order_id, body]).encode()).hexdigest()
            self.db.execute(
                "INSERT INTO events (ts, kind, decision_id, client_order_id, payload, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
                (ts, kind, decision_id, client_order_id, body, prev, digest),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return digest

    def verify_chain(self) -> bool:
        prev = GENESIS
        for r in self.db.execute("SELECT * FROM events ORDER BY seq"):
            expected = hashlib.sha256(
                _canonical([prev, r["ts"], r["kind"], r["decision_id"], r["client_order_id"], r["payload"]]).encode()
            ).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != expected:
                return False
            prev = r["hash"]
        return True

    def events(self, kind: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if kind:
            q = self.db.execute("SELECT * FROM events WHERE kind=? ORDER BY seq DESC LIMIT ?", (kind, limit))
        else:
            q = self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,))
        return q.fetchall()

    def has_event(self, kind: str, decision_id: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM events WHERE kind=? AND decision_id=? LIMIT 1", (kind, decision_id)
        ).fetchone() is not None

    # -- evidence ---------------------------------------------------------------

    def add_evidence(self, cycle_id: str, ev: Evidence) -> None:
        payload = _canonical(ev.payload)
        self.db.execute(
            "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ev.evidence_id, cycle_id, ev.kind, ev.symbol, ev.as_of.isoformat(), ev.source,
                payload, int(ev.untrusted_text), hashlib.sha256(payload.encode()).hexdigest(),
            ),
        )

    def evidence_ids(self, cycle_id: str) -> dict[str, str | None]:
        rows = self.db.execute("SELECT evidence_id, symbol FROM evidence")
        return {r["evidence_id"]: r["symbol"] for r in rows}

    # -- pending entries (research -> execution handoff: OPEN and CLOSE requests) --------------------------

    def add_pending_action(self, decision_id: str, cycle_id: str, proposal_json: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO pending_actions VALUES (?,?,?,?,?)",
            (decision_id, cycle_id, proposal_json, utcnow().isoformat(), "PENDING"),
        )

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM pending_actions WHERE status='PENDING' ORDER BY created_at").fetchall()

    def set_pending_status(self, decision_id: str, status: str) -> None:
        self.db.execute("UPDATE pending_actions SET status=? WHERE decision_id=?", (status, decision_id))

    # -- orders -----------------------------------------------------------------

    def upsert_order(self, **o: Any) -> None:
        now = utcnow().isoformat()
        existing = self.get_order(o["client_order_id"])
        if existing is None:
            self.db.execute(
                """INSERT INTO orders (client_order_id, decision_id, purpose, symbol, side, order_type, quantity,
                   limit_price, stop_price, time_in_force, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    o["client_order_id"], o["decision_id"], o["purpose"], o["symbol"], o["side"], o["order_type"],
                    o["quantity"], o.get("limit_price"), o.get("stop_price"), o["time_in_force"],
                    o["state"].value, now, now,
                ),
            )
        self.append("order_state", {k: (v.value if isinstance(v, OrderState) else v) for k, v in o.items()},
                    decision_id=o["decision_id"], client_order_id=o["client_order_id"])

    def update_order(self, client_order_id: str, state: OrderState, **fields: Any) -> None:
        sets = ["state=?", "updated_at=?"]
        vals: list[Any] = [state.value, utcnow().isoformat()]
        for k in ("broker_order_id", "filled_quantity", "filled_price"):
            if k in fields and fields[k] is not None:
                sets.append(f"{k}=?")
                vals.append(fields[k])
        vals.append(client_order_id)
        self.db.execute(f"UPDATE orders SET {', '.join(sets)} WHERE client_order_id=?", vals)
        row = self.get_order(client_order_id)
        self.append("order_state", {"state": state.value, **fields}, decision_id=row["decision_id"] if row else None,
                    client_order_id=client_order_id)

    def get_order(self, client_order_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)).fetchone()

    def live_orders(self) -> list[sqlite3.Row]:
        terminal = tuple(s.value for s in TERMINAL_STATES)
        q = f"SELECT * FROM orders WHERE state NOT IN ({','.join('?' * len(terminal))})"
        return self.db.execute(q, terminal).fetchall()

    def orders_in_state(self, state: OrderState) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM orders WHERE state=?", (state.value,)).fetchall()

    def entries_submitted_on(self, trading_date: date) -> int:
        """Entry orders sent (or shadow-sent) on a US trading date; used for the trades-per-day cap."""
        rows = self.db.execute(
            "SELECT created_at FROM orders WHERE purpose='ENTRY' AND state NOT IN ('PROPOSED','RISK_APPROVED','REJECTED')"
        ).fetchall()
        return sum(1 for r in rows if to_trading_date(datetime.fromisoformat(r["created_at"])) == trading_date)

    # -- trades (system-opened positions) ---------------------------------------

    def open_trade(self, decision_id: str, symbol: str, quantity: float, entry_price: float,
                   stop_loss: float, take_profit: float, time_stop_date: date,
                   *, side: str = "BUY") -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO trades (decision_id, symbol, quantity, entry_price, stop_loss, take_profit,"
            " time_stop_date, opened_at, status, closed_at, exit_reason, initial_stop, side)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, symbol, quantity, entry_price, stop_loss, take_profit, time_stop_date.isoformat(),
             utcnow().isoformat(), "OPEN", None, None, stop_loss, side),
        )
        self.append("trade_opened", {"symbol": symbol, "quantity": quantity, "entry_price": entry_price,
                                     "stop_loss": stop_loss, "take_profit": take_profit,
                                     "time_stop_date": time_stop_date.isoformat(), "side": side},
                    decision_id=decision_id)

    def open_trades(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

    def reduce_trade(self, decision_id: str, remaining_quantity: float) -> None:
        self.db.execute("UPDATE trades SET quantity=? WHERE decision_id=?", (remaining_quantity, decision_id))
        self.append("trade_reduced", {"remaining_quantity": remaining_quantity}, decision_id=decision_id)

    def raise_stop(self, decision_id: str, new_stop: float) -> None:
        """Move a trade's stop toward safety. For longs: up. For shorts: down. Never the reverse."""
        row = self.db.execute("SELECT stop_loss, side FROM trades WHERE decision_id=?", (decision_id,)).fetchone()
        if row is None:
            return
        is_short = (row["side"] if "side" in row.keys() else "BUY") == "SELL_SHORT"
        if is_short:
            # Short stop: only allow it to move down (more protective)
            if new_stop >= row["stop_loss"]:
                return
        else:
            # Long stop: only allow it to move up
            if new_stop <= row["stop_loss"]:
                return
        # Pre-migration trades keep their original stop as the risk unit before it is overwritten.
        self.db.execute("UPDATE trades SET initial_stop=COALESCE(initial_stop, stop_loss), stop_loss=? WHERE decision_id=?",
                        (new_stop, decision_id))
        self.append("stop_raised", {"from": row["stop_loss"], "to": new_stop}, decision_id=decision_id)

    def close_trade(self, decision_id: str, reason: str) -> None:
        self.db.execute("UPDATE trades SET status='CLOSED', closed_at=?, exit_reason=? WHERE decision_id=?",
                        (utcnow().isoformat(), reason, decision_id))
        self.append("trade_closed", {"reason": reason}, decision_id=decision_id)

    # -- equity -----------------------------------------------------------------

    def snapshot_equity(self, ts: datetime, trading_date: date, equity: float, sweep_pnl: float = 0.0) -> None:
        self.db.execute("INSERT INTO equity_snapshots (ts, trading_date, equity, sweep_pnl) VALUES (?,?,?,?)",
                        (ts.isoformat(), trading_date.isoformat(), equity, sweep_pnl))

    def start_of_day_equity(self, trading_date: date) -> float | None:
        r = self.db.execute(
            "SELECT equity FROM equity_snapshots WHERE trading_date=? ORDER BY ts LIMIT 1", (trading_date.isoformat(),)
        ).fetchone()
        return r["equity"] if r else None

    def peak_equity(self) -> float | None:
        r = self.db.execute("SELECT MAX(equity) AS m FROM equity_snapshots").fetchone()
        return r["m"] if r and r["m"] is not None else None

    # Trading equity = equity minus the cash sweep's P&L: what the drawdown and daily-loss breakers watch.

    def start_of_day_trading_equity(self, trading_date: date) -> float | None:
        r = self.db.execute(
            "SELECT equity - sweep_pnl AS e FROM equity_snapshots WHERE trading_date=? ORDER BY ts LIMIT 1",
            (trading_date.isoformat(),)).fetchone()
        return r["e"] if r else None

    def peak_trading_equity(self) -> float | None:
        r = self.db.execute("SELECT MAX(equity - sweep_pnl) AS m FROM equity_snapshots").fetchone()
        return r["m"] if r and r["m"] is not None else None

    def iter_rows(self, sql: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
        yield from self.db.execute(sql, params)
