from pathlib import Path

from policy import RulePolicy, LearnedPolicy, PolicyLogger, replay_policy_decisions


def test_rule_policy(tmp_path: Path):
    config = tmp_path / "policy_config.yaml"
    config.write_text(
        """
thresholds:
  monitor_max_attack_probability: 0.35
  divert_max_attack_probability: 0.75
  isolate_min_attack_probability: 0.75
weights:
  attack_probability: 0.45
  target_criticality: 0.20
  stage_confidence: 0.10
  soc_load: 0.10
  decoy_capacity: 0.05
  intel_value: 0.10
intel_value:
  monitor: 0.1
  divert: 0.8
  isolate: 0.3
action_cost:
  monitor: 0.05
  divert: 0.25
  isolate: 0.70
""",
        encoding="utf-8",
    )

    policy = RulePolicy(config)
    decision = policy.decide(
        {"attack_probability": 0.6, "target_score": 0.3},
        asset_criticality=4,
        stage_confidence=0.7,
        soc_load=0.2,
        decoy_capacity=0.9,
        human_mode="semi_autonomous",
    )
    assert decision.action in {"monitor", "divert", "isolate"}
    assert "utilities" in decision.metadata


def test_learned_policy_and_logging(tmp_path: Path):
    lp = LearnedPolicy(mode="bandit")
    dec = lp.decide({"attack_probability": 0.8})
    lp.update(dec.action_index, reward=1.0)
    assert dec.action in {"monitor", "divert", "isolate"}

    log_path = tmp_path / "policy_log.json"
    logger = PolicyLogger(log_path)
    logger.append({"timestamp": "2026-01-01T00:00:00Z", "action": "divert"})
    records = logger.read_all()
    assert len(records) == 1

    replayed = replay_policy_decisions(records)
    assert replayed[0]["action"] == "divert"