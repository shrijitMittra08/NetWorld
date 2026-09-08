import networkx as nx
import torch

from graph.temporal_graph import TemporalGraphSnapshot
from forecasting import ForecastEngine


def test_model_stack_forecast_rollout():
    g = nx.DiGraph()
    g.add_node("host_a", role="workstation", subnet="10.0.0.0/24", os="Windows", active_edges=2)
    g.add_node("host_b", role="server", subnet="10.0.1.0/24", os="Linux", active_edges=1)
    g.add_edge("host_a", "host_b", count=3)

    snapshot = TemporalGraphSnapshot(window_id=0, window_start=None, window_end=None, graph=g)

    engine = ForecastEngine(encoder_type="sage", rollout_steps=2)
    result = engine.forecast([snapshot])

    assert "rollout" in result
    assert len(result["rollout"]) == 2
    assert "final_attack_probability" in result