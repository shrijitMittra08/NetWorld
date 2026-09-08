from pathlib import Path

import networkx as nx

from graph.temporal_graph import TemporalGraphSnapshot
from training import Trainer
from policy import LearnedPolicy
from feedback import (
    SampleManager,
    monitor_drift,
    validate_fresh_decoy_set,
    FeedbackTrainer,
    ImprovementLog,
)


def test_sample_manager_dedup():
    sm = SampleManager()
    assert sm.add_sample({"a": 1}) is True
    assert sm.add_sample({"a": 1}) is False


def test_drift_and_validation():
    drift = monitor_drift([0.1, 0.2], [0.1, 0.9], threshold=0.25)
    assert drift.drift_detected is True

    val = validate_fresh_decoy_set([0.9, 0.2], [1, 0], min_score=0.5)
    assert val.passed is True


def test_feedback_trainer_and_logging(tmp_path: Path):
    g = nx.DiGraph()
    g.add_node("a", role="workstation", active_edges=1)
    g.add_node("b", role="server", active_edges=1)
    g.add_edge("a", "b")

    snap = TemporalGraphSnapshot(window_id=0, window_start=None, window_end=None, graph=g)

    trainer = Trainer()
    policy = LearnedPolicy(mode="bandit")
    log = ImprovementLog(tmp_path / "improvement.json")
    fb = FeedbackTrainer(trainer=trainer, learned_policy=policy, improvement_log=log)

    fb.ingest_feedback_samples([{"x": 1}, {"x": 1}, {"x": 2}])

    result = fb.retrain(
        graphs=[[snap]],
        targets_list=[{"attack": 1, "stage": 2, "target_index": 0}],
        baseline_probs=[0.1, 0.2],
        current_probs=[0.9, 0.8],
        validation_predictions=[0.9, 0.2],
        validation_labels=[1, 0],
        policy_rewards=[{"action_index": 1, "reward": 1.0}],
    )

    assert result.retrained is True
    assert len(log.read_all()) >= 1