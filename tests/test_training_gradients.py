import networkx as nx
import torch

from graph.temporal_graph import TemporalGraphSnapshot
from training import Trainer


def test_forecast_training_path_preserves_gradients():
    g = nx.DiGraph()

    g.add_node("a", role="workstation")
    g.add_node("b", role="server")

    g.add_edge("a", "b")

    snapshot = TemporalGraphSnapshot(
        window_id=0,
        window_start=None,
        window_end=None,
        graph=g,
    )

    trainer = Trainer()

    trainer.optimizer.zero_grad()

    forecast = trainer.engine.forecast_tensors(
        [snapshot],
    )

    rollout_preds = forecast["rollout"]

    attack_targets = [
        torch.tensor([1.0])
        for _ in rollout_preds
    ]

    stage_targets = [
        torch.tensor([2], dtype=torch.long)
        for _ in rollout_preds
    ]

    from training.losses import rollout_loss

    loss = rollout_loss(
        rollout_preds=rollout_preds,
        attack_targets=attack_targets,
        stage_targets=stage_targets,
    )

    loss.backward()

    assert loss.requires_grad

    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in trainer.engine.world_model.parameters()
    )