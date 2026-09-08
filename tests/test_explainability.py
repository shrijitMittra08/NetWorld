import networkx as nx

from explainability import build_explanation


def test_build_explanation():
    g = nx.DiGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")

    forecast = {
        "attack_probability": 0.8,
        "target_score": 0.5,
        "stage_logits": [0.1, 0.2, 0.3],
    }

    explanation = build_explanation(
        forecast=forecast,
        graph=g,
        features={"new_peers": 5, "auth_failures": 2},
        window_summaries=[
            {"new_peers": 3, "auth_failures": 1},
            {"new_peers": 5, "auth_failures": 2},
        ],
    )

    assert "feature_attribution" in explanation
    assert "graph_explanation" in explanation
    assert "temporal_explanation" in explanation