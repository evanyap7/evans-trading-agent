"""Typed contracts between every stage of the pipeline.

The LLM may only communicate through `AgentOutput`. Everything downstream
(verification, sizing, risk, execution) consumes these validated objects,
never free text.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SYMBOL_PATTERN = r"^[A-Z]{1,5}(\.[A-Z])?$"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Market and account data
# ---------------------------------------------------------------------------


class Bar(Strict):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Quote(Strict):
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: float = 0
    ask_size: float = 0
    volume: float = 0
    last_trade_time: datetime | None = None
    fetched_at: datetime
    source: str = "broker"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return math.inf
        return (self.ask - self.bid) / self.mid * 100

    @property
    def data_time(self) -> datetime:
        """When the market data itself was produced: the older of the trade time and our fetch time."""
        return min(self.fetched_at, self.last_trade_time) if self.last_trade_time else self.fetched_at

    def age_seconds(self, now: datetime) -> float:
        return (now - self.data_time).total_seconds()


class Position(Strict):
    symbol: str
    quantity: float
    avg_cost: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price


class BrokerOrderStatus(str, Enum):
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    NOT_FOUND = "NOT_FOUND"


class BrokerOrder(Strict):
    client_order_id: str
    broker_order_id: str | None = None
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: str
    quantity: float
    filled_quantity: float = 0
    filled_price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    status: BrokerOrderStatus
    raw_status: str = ""


class AccountState(Strict):
    account_id: str
    equity: float
    cash: float
    buying_power: float
    positions: list[Position]
    open_orders: list[BrokerOrder]
    as_of: datetime

    def position(self, symbol: str) -> Position | None:
        return next((p for p in self.positions if p.symbol == symbol and p.quantity != 0), None)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class Evidence(Strict):
    evidence_id: str
    kind: Literal["price_features", "quote", "regime", "position", "event", "document"]
    symbol: str | None
    as_of: datetime
    source: str
    payload: dict
    untrusted_text: bool = False  # true for news/filings: content may contain prompt injection


# ---------------------------------------------------------------------------
# LLM contract
# ---------------------------------------------------------------------------


class EntryOrder(Strict):
    order_type: Literal["LIMIT"]
    limit_price: float = Field(gt=0)


class ExitPlan(Strict):
    take_profit: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    time_stop_days: int = Field(ge=1, le=40)


class TradeProposal(Strict):
    """An LLM request to open a long position. It asks for risk, not a quantity."""

    action: Literal["OPEN"] = "OPEN"
    symbol: str = Field(pattern=SYMBOL_PATTERN)
    instrument_type: Literal["EQUITY", "ETF"]
    side: Literal["BUY"]
    strategy: str = Field(min_length=3, max_length=64)
    thesis: str = Field(min_length=10, max_length=1200)
    holding_period_days: int = Field(ge=1, le=40)
    confidence: float = Field(ge=0, le=1)
    expected_return_pct: float = Field(ge=-50, le=50)
    invalidation_price: float = Field(gt=0)
    entry: EntryOrder
    exit: ExitPlan
    requested_risk_pct: float = Field(gt=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _prices_ordered(self) -> "TradeProposal":
        limit = self.entry.limit_price
        if not self.exit.stop_loss < limit < self.exit.take_profit:
            raise ValueError("long trade needs stop_loss < limit_price < take_profit")
        if self.invalidation_price >= limit:
            raise ValueError("invalidation_price must be below the entry limit")
        return self


class ExitProposal(Strict):
    """An LLM request to close an existing position because the thesis changed."""

    action: Literal["CLOSE"] = "CLOSE"
    symbol: str = Field(pattern=SYMBOL_PATTERN)
    reason: str = Field(min_length=10, max_length=600)
    evidence_ids: list[str] = Field(min_length=1)


class AgentOutput(Strict):
    """The only shape the research agent may return. Empty `proposals` means NO_TRADE."""

    market_view: str = Field(max_length=1500)
    proposals: list[TradeProposal] = Field(default_factory=list, max_length=5)
    exits: list[ExitProposal] = Field(default_factory=list, max_length=10)
    no_trade_reason: str | None = None


class Proposal(Strict):
    """A TradeProposal as recorded by the system, with identity and provenance."""

    decision_id: str
    cycle_id: str
    created_at: datetime
    source: Literal["llm", "tiered-llm", "baseline"]
    model: str
    trade: TradeProposal

    @field_validator("decision_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("decision_id required")
        return v


# ---------------------------------------------------------------------------
# Verification, sizing and risk
# ---------------------------------------------------------------------------


class Verification(Strict):
    passed: bool
    failures: list[str]
    metrics: dict[str, float]


class SizedOrder(Strict):
    decision_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    limit_price: float
    stop_loss: float
    take_profit: float
    notional: float
    risk_usd: float
    sizing_notes: list[str] = Field(default_factory=list)


class RiskDecision(Strict):
    approved: bool
    failed_checks: list[str]
    passed_checks: list[str]
    limits_sha256: str


class OrderState(str, Enum):
    PROPOSED = "PROPOSED"
    RISK_APPROVED = "RISK_APPROVED"
    SHADOW = "SHADOW"  # would have been submitted; shadow mode sends nothing
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN_RECONCILE = "UNKNOWN_RECONCILE"


TERMINAL_STATES = {OrderState.SHADOW, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}
