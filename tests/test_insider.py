from datetime import date

from trading_agent.agents import AgentContext, BaselineMomentumAgent
from trading_agent.insider import demo


def test_filing_date_windowing():
    demo()


def test_agent_buys_insider_cluster_only_when_enabled():
    from datetime import datetime, timezone

    from trading_agent.schemas import Evidence
    f = {"close": 100.0, "atr14": 2.0, "dist_sma200_pct": -5.0, "dist_sma50_pct": -5.0, "ret_60d_pct": -10.0}
    now = datetime(2024, 3, 1, tzinfo=timezone.utc)
    ev = [Evidence(evidence_id="px_X", kind="price_features", symbol="X", as_of=now, source="t", payload=f),
          Evidence(evidence_id="rg", kind="regime", symbol=None, as_of=now, source="t",
                   payload={"benchmark_above_sma200": True})]
    ctx = AgentContext(as_of=date(2024, 3, 1), evidence=ev, universe={"X": {"type": "EQUITY", "sector": "T"}},
                       account_summary={}, positions=[], known_earnings={}, features={"X": f},
                       insider={"X": {"usd": 500_000, "insiders": 2}})
    def longs(agent):
        return [p for p in agent.propose(ctx).proposals if p.side == "BUY"]

    assert longs(BaselineMomentumAgent()) == []  # downtrend: momentum won't buy it
    out = longs(BaselineMomentumAgent(insider_min_usd=100_000))
    assert [p.strategy for p in out] == ["insider_cluster"] and out[0].holding_period_days == 40
    assert longs(BaselineMomentumAgent(insider_min_usd=900_000)) == []
