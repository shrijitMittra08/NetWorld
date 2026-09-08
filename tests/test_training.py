import networkx as nx

from graph.temporal_graph import TemporalGraphSnapshot
from training import Trainer

def test_trainer_train_step():
    g = nx.DiGraph()
    g.add_node("a", role="workstation")
    g.add_node("b", role="server")
    g.add_edge("a", "b")

    snapshot = TemporalGraphSnapshot(window_id=0, window_start=None, window_end=None, graph=g)

    trainer = Trainer()
    loss = trainer.train_step(
        graphs=[snapshot],
        targets={"attack": 1, "stage": 2, "target": 0.5},
    )

    assert isinstance(loss, float)