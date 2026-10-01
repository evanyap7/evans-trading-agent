"""Unit tests for Databento market data adapter and CLI integration."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent.cli import main
from trading_agent.databento_data import DatabentoClient, DatabentoConfig
from trading_agent.schemas import Bar

ET = ZoneInfo("America/New_York")


@pytest.fixture
def mock_ohlcv_df():
    """Create a sample pandas DataFrame matching Databento ohlcv-1d output."""
    data = [
        {
            "ts_event": pd.Timestamp("2025-01-02 00:00:00+0000", tz="UTC"),
            "symbol": "SPY",
            "open": 580.0,
            "high": 585.0,
            "low": 578.0,
            "close": 583.5,
            "volume": 45000000.0,
        },
        {
            "ts_event": pd.Timestamp("2025-01-03 00:00:00+0000", tz="UTC"),
            "symbol": "SPY",
            "open": 584.0,
            "high": 588.0,
            "low": 582.0,
            "close": 586.2,
            "volume": 42000000.0,
        },
        {
            "ts_event": pd.Timestamp("2025-01-02 00:00:00+0000", tz="UTC"),
            "symbol": "QQQ",
            "open": 510.0,
            "high": 515.0,
            "low": 508.0,
            "close": 513.0,
            "volume": 30000000.0,
        },
    ]
    df = pd.DataFrame(data)
    df.set_index("ts_event", inplace=True)
    return df


class TestDatabentoClientConfig:
    def test_availability_without_key(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        client = DatabentoClient()
        assert not client.is_available
        with pytest.raises(ValueError, match="Databento API key is missing"):
            client._get_historical_client()

    def test_availability_with_explicit_key(self):
        client = DatabentoClient(DatabentoConfig(api_key="db-test-key-123"))
        assert client.is_available
        assert client.api_key == "db-test-key-123"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "db-env-key-999")
        monkeypatch.setenv("DATABENTO_DATASET", "XNAS.ITCH")
        monkeypatch.setenv("DATABENTO_COST_LIMIT_USD", "2.50")
        client = DatabentoClient()
        assert client.is_available
        assert client.api_key == "db-env-key-999"
        assert client.default_dataset == "XNAS.ITCH"
        assert client.cost_limit_usd == 2.50


class TestDatabentoReferenceAndMetadata:
    @patch("databento.Historical")
    def test_list_datasets(self, mock_hist_cls):
        mock_instance = MagicMock()
        mock_instance.metadata.list_datasets.return_value = ["EQUS.SUMMARY", "EQUS.MINI", "XNAS.ITCH"]
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key"))
        datasets = client.list_datasets()
        assert "EQUS.SUMMARY" in datasets
        assert len(datasets) == 3
        mock_hist_cls.assert_called_once_with(key="test-key")

    @patch("databento.Historical")
    def test_list_schemas(self, mock_hist_cls):
        mock_instance = MagicMock()
        mock_instance.metadata.list_schemas.return_value = ["mbo", "mbp-1", "ohlcv-1d", "trades"]
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key"))
        schemas = client.list_schemas("EQUS.SUMMARY")
        assert "ohlcv-1d" in schemas
        mock_instance.metadata.list_schemas.assert_called_once_with(dataset="EQUS.SUMMARY")

    @patch("databento.Historical")
    def test_estimate_cost(self, mock_hist_cls):
        mock_instance = MagicMock()
        mock_instance.metadata.get_cost.return_value = 0.045
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key"))
        cost = client.estimate_cost(["SPY", "QQQ"], "2025-01-01", "2025-01-31")
        assert cost == 0.045
        mock_instance.metadata.get_cost.assert_called_once_with(
            dataset="EQUS.SUMMARY",
            symbols=["SPY", "QQQ"],
            schema="ohlcv-1d",
            start="2025-01-01",
            end="2025-01-31",
        )

    @patch("databento.Historical")
    def test_resolve_symbology(self, mock_hist_cls):
        mock_instance = MagicMock()
        mock_instance.symbology.resolve.return_value = {"result": {"SPY": [{"instrument_id": 12345}]}}
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key"))
        res = client.resolve_symbology(["SPY"], "2025-01-01", "2025-01-05")
        assert "result" in res
        mock_instance.symbology.resolve.assert_called_once_with(
            dataset="EQUS.SUMMARY",
            symbols=["SPY"],
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date="2025-01-01",
            end_date="2025-01-05",
        )


class TestDatabentoHistoricalBars:
    def test_df_to_bars_conversion(self, mock_ohlcv_df):
        bars_dict = DatabentoClient._df_to_bars(mock_ohlcv_df, ["SPY", "QQQ"])
        assert "SPY" in bars_dict
        assert "QQQ" in bars_dict
        assert len(bars_dict["SPY"]) == 2
        assert len(bars_dict["QQQ"]) == 1

        spy_bar_0 = bars_dict["SPY"][0]
        assert isinstance(spy_bar_0, Bar)
        assert spy_bar_0.open == 580.0
        assert spy_bar_0.high == 585.0
        assert spy_bar_0.low == 578.0
        assert spy_bar_0.close == 583.5
        assert spy_bar_0.volume == 45000000.0
        # Check normalized time to 16:00 ET
        assert spy_bar_0.ts.tzinfo == ET
        assert spy_bar_0.ts.hour == 16
        assert spy_bar_0.ts.minute == 0

    @patch("databento.Historical")
    def test_get_daily_bars_fetching_and_cache(self, mock_hist_cls, mock_ohlcv_df, tmp_path):
        mock_instance = MagicMock()
        mock_instance.metadata.get_cost.return_value = 0.02
        mock_range_data = MagicMock()
        mock_range_data.to_df.return_value = mock_ohlcv_df
        mock_instance.timeseries.get_range.return_value = mock_range_data
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key", cost_limit_usd=1.0))
        cache_dir = tmp_path / "databento_cache"

        # First call fetches from mock client and saves cache
        bars = client.get_daily_bars(
            symbols=["SPY", "QQQ"],
            start=date(2025, 1, 1),
            end=date(2025, 1, 5),
            cache_dir=cache_dir,
        )
        assert len(bars["SPY"]) == 2
        assert len(bars["QQQ"]) == 1
        assert mock_instance.timeseries.get_range.call_count == 1

        # Check cache files were written
        spy_cache = cache_dir / "db_bars_SPY_2025-01-01_2025-01-05.json"
        assert spy_cache.exists()

        # Second call should read from disk cache and NOT call timeseries.get_range again
        mock_instance.timeseries.get_range.reset_mock()
        cached_bars = client.get_daily_bars(
            symbols=["SPY", "QQQ"],
            start=date(2025, 1, 1),
            end=date(2025, 1, 5),
            cache_dir=cache_dir,
        )
        assert len(cached_bars["SPY"]) == 2
        assert mock_instance.timeseries.get_range.call_count == 0

    @patch("databento.Historical")
    def test_get_daily_bars_cost_limit_guard(self, mock_hist_cls):
        mock_instance = MagicMock()
        mock_instance.metadata.get_cost.return_value = 5.50  # Exceeds limit of $1.00
        mock_hist_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key", cost_limit_usd=1.0))
        with pytest.raises(ValueError, match="exceeds cost limit"):
            client.get_daily_bars(
                symbols=["SPY"],
                start=date(2025, 1, 1),
                end=date(2025, 1, 5),
            )


class TestDatabentoLive:
    @patch("databento.Live")
    def test_start_live_stream(self, mock_live_cls):
        mock_instance = MagicMock()
        mock_live_cls.return_value = mock_instance

        client = DatabentoClient(DatabentoConfig(api_key="test-key", live_dataset="EQUS.MINI"))
        dummy_callback = MagicMock()

        live_client = client.start_live_stream(["AAPL", "TSLA"], callback=dummy_callback, schema="trades")
        mock_live_cls.assert_called_once_with(key="test-key")
        mock_instance.subscribe.assert_called_once_with(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=["AAPL", "TSLA"],
        )
        mock_instance.add_callback.assert_called_once_with(dummy_callback)
        mock_instance.start.assert_called_once()
        assert live_client == mock_instance

    def test_start_live_stream_without_key(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        client = DatabentoClient()
        with pytest.raises(ValueError, match="Databento API key is missing"):
            client.start_live_stream(["AAPL"], callback=lambda x: None)


class TestDatabentoCli:
    def test_databento_probe_no_key(self, monkeypatch, capsys):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        main(["databento", "probe"])
        out = capsys.readouterr().out
        assert "Databento is not configured" in out

    @patch("trading_agent.databento_data.DatabentoClient.list_datasets")
    @patch("trading_agent.databento_data.DatabentoClient.list_schemas")
    def test_databento_probe_with_key(self, mock_list_schemas, mock_list_ds, monkeypatch, capsys):
        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
        mock_list_ds.return_value = ["EQUS.SUMMARY", "GLBX.MDP3"]
        mock_list_schemas.return_value = ["ohlcv-1d", "trades"]
        main(["databento", "probe"])
        out = capsys.readouterr().out
        assert "Databento Client: CONNECTED" in out
        assert "EQUS.SUMMARY" in out

    @patch("trading_agent.databento_data.DatabentoClient.estimate_cost")
    def test_databento_cost(self, mock_est_cost, monkeypatch, capsys):
        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
        mock_est_cost.return_value = 0.0125
        main([
            "databento", "cost",
            "--symbols", "SPY,QQQ",
            "--start", "2025-01-01",
            "--end", "2025-01-31"
        ])
        out = capsys.readouterr().out
        assert "Estimated query cost" in out
        assert "0.0125" in out

    @patch("trading_agent.databento_data.DatabentoClient.get_daily_bars")
    def test_databento_bars(self, mock_get_bars, monkeypatch, capsys):
        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
        bar = Bar(
            ts=datetime(2025, 1, 2, 16, 0, tzinfo=ET),
            open=580.0,
            high=585.0,
            low=578.0,
            close=583.5,
            volume=100000.0,
        )
        mock_get_bars.return_value = {"SPY": [bar]}
        main([
            "databento", "bars",
            "--symbols", "SPY",
            "--start", "2025-01-01",
            "--end", "2025-01-05"
        ])
        out = capsys.readouterr().out
        assert "SPY: 1 bars loaded" in out
        assert "close=583.5" in out
