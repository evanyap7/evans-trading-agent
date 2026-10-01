"""Databento market data adapter for Evan's Trading Agent.

Provides institutional-grade Historical, Live, and Reference data from Databento:
- Historical: High-precision split/dividend adjusted daily and intraday OHLCV bars (`ohlcv-1d`, `ohlcv-1h`, `ohlcv-1m`).
- Live: Sub-microsecond live streaming quotes, trades, and market depth (`db.Live`).
- Reference: Symbology resolution, dataset catalog, schema inspection, and pre-query cost calculation.

Documentation: https://databento.com/docs/quickstart/build-first-app?historical=python&live=python&reference=python
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .schemas import Bar, Quote

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DatabentoConfig:
    api_key: str | None = None
    default_dataset: str = "EQUS.SUMMARY"  # US Equities summary / OHLCV-1d
    live_dataset: str = "EQUS.MINI"        # Real-time top-of-book / trades
    cost_limit_usd: float = 1.0            # Safety cap per query in USD


class DatabentoClient:
    """Institutional market data client wrapping Databento Historical, Live, and Reference APIs."""

    def __init__(self, config: DatabentoConfig | None = None):
        cfg = config or DatabentoConfig()
        self.api_key = cfg.api_key or os.environ.get("DATABENTO_API_KEY")
        self.default_dataset = os.environ.get("DATABENTO_DATASET", cfg.default_dataset)
        self.live_dataset = cfg.live_dataset
        self.cost_limit_usd = float(os.environ.get("DATABENTO_COST_LIMIT_USD", cfg.cost_limit_usd))

    @property
    def is_available(self) -> bool:
        """True if Databento SDK is installed and an API key is provided."""
        if not self.api_key:
            return False
        try:
            import databento  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_historical_client(self):
        if not self.is_available:
            raise ValueError(
                "Databento API key is missing. Set DATABENTO_API_KEY in .env or pass api_key."
            )
        import databento as db
        return db.Historical(key=self.api_key)

    # -------------------------------------------------------------------------
    # Reference & Metadata
    # -------------------------------------------------------------------------

    def list_datasets(self) -> list[str]:
        """List all available datasets in Databento."""
        client = self._get_historical_client()
        return client.metadata.list_datasets()

    def list_schemas(self, dataset: str | None = None) -> list[str]:
        """List available schemas for a dataset (e.g. ohlcv-1d, trades, mbo)."""
        client = self._get_historical_client()
        ds = dataset or self.default_dataset
        return client.metadata.list_schemas(dataset=ds)

    def estimate_cost(
        self,
        symbols: Sequence[str],
        start: date | datetime | str,
        end: date | datetime | str,
        schema: str = "ohlcv-1d",
        dataset: str | None = None,
    ) -> float:
        """Estimate the query cost in USD before executing."""
        client = self._get_historical_client()
        ds = dataset or self.default_dataset
        start_str = start.isoformat() if hasattr(start, "isoformat") else str(start)
        end_str = end.isoformat() if hasattr(end, "isoformat") else str(end)
        cost = client.metadata.get_cost(
            dataset=ds,
            symbols=list(symbols),
            schema=schema,
            start=start_str,
            end=end_str,
        )
        return float(cost)

    def resolve_symbology(
        self,
        symbols: Sequence[str],
        start_date: date | str,
        end_date: date | str,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Resolve ticker symbology to Databento instrument IDs."""
        client = self._get_historical_client()
        ds = dataset or self.default_dataset
        s_date = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
        e_date = end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date)
        return client.symbology.resolve(
            dataset=ds,
            symbols=list(symbols),
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=s_date,
            end_date=e_date,
        )

    # -------------------------------------------------------------------------
    # Historical Market Data (OHLCV)
    # -------------------------------------------------------------------------

    def get_daily_bars(
        self,
        symbols: Sequence[str],
        start: date | datetime,
        end: date | datetime,
        dataset: str | None = None,
        cache_dir: Path | None = None,
    ) -> dict[str, list[Bar]]:
        """Fetch split- and dividend-adjusted daily bars as `Bar` models.
        
        Args:
            symbols: Ticker symbols, e.g. ["SPY", "AAPL", "MSFT"]
            start: Start date
            end: End date
            dataset: Databento dataset (defaults to EQUS.SUMMARY)
            cache_dir: Optional disk cache directory
        """
        out: dict[str, list[Bar]] = {}
        missing_symbols: list[str] = []

        # Check local disk cache first if cache_dir is specified
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for s in symbols:
                p = cache_dir / f"db_bars_{s}_{start}_{end}.json"
                if p.exists():
                    try:
                        out[s] = [Bar.model_validate(b) for b in json.loads(p.read_text())]
                    except Exception:
                        missing_symbols.append(s)
                else:
                    missing_symbols.append(s)
        else:
            missing_symbols = list(symbols)

        if not missing_symbols:
            return out

        client = self._get_historical_client()
        ds = dataset or self.default_dataset
        s_str = start.isoformat() if hasattr(start, "isoformat") else str(start)
        e_str = end.isoformat() if hasattr(end, "isoformat") else str(end)

        # Pre-flight cost check
        try:
            cost = self.estimate_cost(missing_symbols, s_str, e_str, schema="ohlcv-1d", dataset=ds)
            if cost > self.cost_limit_usd:
                raise ValueError(
                    f"Databento query cost (${cost:.4f} USD) exceeds cost limit (${self.cost_limit_usd:.2f} USD). "
                    f"Raise DATABENTO_COST_LIMIT_USD or narrow your date range/symbols."
                )
        except Exception as e:
            if "exceeds cost limit" in str(e):
                raise
            logger.debug("Cost pre-check skipped: %s", e)

        data = client.timeseries.get_range(
            dataset=ds,
            symbols=missing_symbols,
            schema="ohlcv-1d",
            start=s_str,
            end=e_str,
        )

        df = data.to_df()
        if df.empty:
            return out

        # Convert DataFrame to dict of Bar models
        # Databento dataframe usually has 'symbol' column and 'ts_event' or DatetimeIndex
        parsed = self._df_to_bars(df, missing_symbols)
        for sym, bars in parsed.items():
            out[sym] = bars
            if cache_dir and bars:
                p = cache_dir / f"db_bars_{sym}_{start}_{end}.json"
                p.write_text(json.dumps([b.model_dump(mode="json") for b in bars]))

        return out

    @staticmethod
    def _df_to_bars(df: pd.DataFrame, target_symbols: Sequence[str]) -> dict[str, list[Bar]]:
        """Convert a Databento OHLCV DataFrame into a dictionary of Bar models."""
        out: dict[str, list[Bar]] = {s: [] for s in target_symbols}

        # Normalize symbol column and timestamp index/column
        if "symbol" not in df.columns and len(target_symbols) == 1:
            df["symbol"] = target_symbols[0]

        if not isinstance(df.index, pd.DatetimeIndex) and "ts_event" in df.columns:
            df.index = pd.to_datetime(df["ts_event"])

        for idx, row in df.iterrows():
            sym = str(row.get("symbol", ""))
            if sym not in out:
                # If symbology returned raw numeric ID, map to first target symbol if single request
                if len(target_symbols) == 1:
                    sym = target_symbols[0]
                else:
                    continue

            # Ensure timestamp is localized to America/New_York (market close: 16:00 ET)
            ts = idx if isinstance(idx, datetime) else row.get("ts_event")
            if ts is None:
                continue
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc).astimezone(ET)
            else:
                ts = ts.astimezone(ET)
            # Normalize daily close timestamp to 16:00 ET
            ts = datetime.combine(ts.date(), time(16, 0), ET)

            b = Bar(
                ts=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
            out[sym].append(b)

        # Sort bars chronologically
        for sym in out:
            out[sym] = sorted(out[sym], key=lambda b: b.ts)

        return out

    # -------------------------------------------------------------------------
    # Live Streaming
    # -------------------------------------------------------------------------

    def start_live_stream(
        self,
        symbols: Sequence[str],
        callback: Callable[[Any], None],
        schema: str = "trades",
        dataset: str | None = None,
    ):
        """Create and start a live streaming connection via Databento Live.
        
        Args:
            symbols: Symbols to stream
            callback: Function invoked on each incoming data record
            schema: Live schema ('trades', 'bbo-1s', 'tbbo', etc.)
            dataset: Dataset (e.g. 'EQUS.MINI' or 'XNAS.ITCH')
        
        Returns:
            The running `databento.Live` client. Call `.stop()` to terminate.
        """
        if not self.is_available:
            raise ValueError("Databento API key is missing. Set DATABENTO_API_KEY.")
        import databento as db

        live_client = db.Live(key=self.api_key)
        ds = dataset or self.live_dataset

        live_client.subscribe(
            dataset=ds,
            schema=schema,
            symbols=list(symbols),
        )
        live_client.add_callback(callback)
        live_client.start()
        return live_client
