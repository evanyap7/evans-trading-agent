"""Loads the human-controlled configuration.

Risk limits are frozen Pydantic models built from a YAML file under version
control. Nothing the LLM produces flows into these objects.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LiveTradingLimits(Frozen):
    enabled: bool = False
    allowed_instruments: tuple[str, ...] = ("EQUITY", "ETF")
    permit_shorting: bool = False
    permit_margin: bool = False
    permit_options: bool = False


class AccountLimits(Frozen):
    max_position_pct: float
    max_sector_pct: float
    max_total_exposure_pct: float
    max_risk_per_trade_pct: float
    max_portfolio_risk_pct: float
    unknown_stop_risk_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_new_trades_per_day: int
    max_order_value_usd: float
    max_adv_participation_pct: float


class SignalLimits(Frozen):
    min_confidence: float
    min_net_edge_pct: float
    safety_buffer_pct: float
    min_reward_risk: float
    min_stop_atr: float
    max_stop_atr: float


class ExecutionLimits(Frozen):
    max_spread_pct: float
    max_quote_age_seconds: float
    max_bar_age_days: int
    max_account_age_seconds: float
    price_collar_pct: float
    slippage_bps: float
    fee_per_order_usd: float
    broker_side_stops: bool = True


class EventLimits(Frozen):
    earnings_blackout_days: int
    require_earnings_data_for_stocks: bool = True


class RiskLimits(Frozen):
    live_trading: LiveTradingLimits
    account: AccountLimits
    signal: SignalLimits
    execution: ExecutionLimits
    events: EventLimits
    source_sha256: str = Field(default="", description="Hash of the YAML this was loaded from")


class Security(Frozen):
    symbol: str
    type: str
    sector: str


class Universe(Frozen):
    symbols: dict[str, Security]
    regime_benchmark: str

    def get(self, symbol: str) -> Security | None:
        return self.symbols.get(symbol)


class Events(Frozen):
    earnings: dict[str, date] = Field(default_factory=dict)


class TradingMode(str, Enum):
    SHADOW = "shadow"
    BROKER = "broker"


class Settings(Frozen):
    """Process settings from the environment. Credentials are read only by the broker adapter."""

    trading_mode: TradingMode
    webull_environment: str
    webull_region: str
    webull_account_id: str
    state_dir: Path
    llm_model: str = "claude-3-7-sonnet-latest"
    llm_model_fast: str = "claude-3-5-haiku-latest"

    @property
    def is_production(self) -> bool:
        return self.webull_environment == "prod"


def _read_yaml(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def load_risk_limits(path: Path | None = None) -> RiskLimits:
    data, digest = _read_yaml(path or CONFIG_DIR / "risk_limits.yaml")
    return RiskLimits(**data, source_sha256=digest)


def load_universe(path: Path | None = None) -> Universe:
    data, _ = _read_yaml(path or CONFIG_DIR / "universe.yaml")
    symbols = {s: Security(symbol=s, **v) for s, v in data["symbols"].items()}
    return Universe(symbols=symbols, regime_benchmark=data["regime_benchmark"])


def load_events(path: Path | None = None) -> Events:
    data, _ = _read_yaml(path or CONFIG_DIR / "events.yaml")
    return Events(earnings=data.get("earnings") or {})


def load_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ.get("WEBULL_ENVIRONMENT", "uat").lower()
    if env not in {"uat", "prod"}:
        raise ValueError("WEBULL_ENVIRONMENT must be 'uat' or 'prod'")
    return Settings(
        trading_mode=TradingMode(os.environ.get("TRADING_MODE", "shadow").lower()),
        webull_environment=env,
        webull_region=os.environ.get("WEBULL_REGION", "sg").lower(),
        webull_account_id=os.environ.get("WEBULL_ACCOUNT_ID", ""),
        state_dir=Path(os.environ.get("STATE_DIR", PROJECT_ROOT / "state")),
        llm_model=os.environ.get("LLM_MODEL", "claude-3-7-sonnet-latest"),
        llm_model_fast=os.environ.get("LLM_MODEL_FAST", "claude-3-5-haiku-latest"),
    )
