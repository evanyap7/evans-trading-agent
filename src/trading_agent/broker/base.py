"""Broker interface shared by the Webull adapter and the simulator.

Backtest, shadow, UAT and live modes run the same pipeline; only the object
implementing this protocol changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..schemas import AccountState, Bar, BrokerOrder, Quote


class BrokerError(Exception):
    """A broker call failed. `ambiguous` means the request may have reached the broker."""

    def __init__(self, message: str, ambiguous: bool = False):
        super().__init__(message)
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["LIMIT", "STOP_LOSS", "MARKET"]
    quantity: int
    time_in_force: Literal["DAY", "GTC"]
    limit_price: float | None = None
    stop_price: float | None = None
    instrument_type: str = "EQUITY"


@dataclass
class PreviewResult:
    ok: bool
    estimated_cost: float | None
    estimated_fees: float | None
    raw: dict = field(default_factory=dict)
    error: str = ""


class Broker(Protocol):
    name: str

    def get_account(self) -> AccountState: ...

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]: ...

    def get_daily_bars(self, symbols: list[str], count: int) -> dict[str, list[Bar]]: ...

    def preview(self, order: OrderRequest) -> PreviewResult: ...

    def place(self, order: OrderRequest) -> str | None:
        """Submit an order. Returns the broker order id if known. Raises BrokerError."""
        ...

    def cancel(self, client_order_id: str) -> None: ...

    def get_order(self, client_order_id: str) -> BrokerOrder: ...
