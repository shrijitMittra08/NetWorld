import networkx as nx
import torch

from graph.temporal_graph import TemporalGraphSnapshot
from forecasting import ForecastEngine


def test_forecast_returns_explanation():
    g = nx.DiGraph()
    g.add_node("host_a", role="workstation", subnet="10.0.0.0/24", os="Windows", active_edges=2)
    g.add_node("host_b", role="server", subnet="10.0.1.0/24", os="Linux", active_edges=1)
    g.add_edge("host_a", "host_b", count=3)

    snapshot = TemporalGraphSnapshot(window_id=0, window_start=None, window_end=None, graph=g)

    engine = ForecastEngine(rollout_steps=2)
    result = engine.forecast(
        [snapshot],
        feature_tensor=torch.randn(1, 8),
        feature_names=[f"f{i}" for i in range(8)],
        window_summaries=[{"a": 1}, {"a": 2}],
    )

    assert "explanation" in result
    assert "feature_attribution" in result["explanation"]
    assert "graph_highlight" in result["explanation"]
    assert "temporal_explanation" in result["explanation"]