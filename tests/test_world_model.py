import networkx as nx

from graph.temporal_graph import TemporalGraphSnapshot
from forecasting import ForecastEngine


def test_forecast_engine_runs():
    g = nx.DiGraph()
    g.add_node("10.0.0.1:1234", role="workstation", subnet="10.0.0.0/24", os="Windows")
    g.add_node("10.0.0.2:80", role="server", subnet="10.0.1.0/24", os="Linux")
    g.add_edge("10.0.0.1:1234", "10.0.0.2:80", protocol="TCP", duration=1.2)

    snapshot = TemporalGraphSnapshot(window_id=0, window_start=None, window_end=None, graph=g)

    engine = ForecastEngine()
    result = engine.forecast([snapshot])

    assert "attack_probability" in result
    assert "stage_logits" in result
    assert "target_score" in result