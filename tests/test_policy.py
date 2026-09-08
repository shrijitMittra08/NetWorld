from pathlib import Path

from policy import RulePolicy


def test_rule_policy_decisions(tmp_path: Path):
    config = tmp_path / "policy_config.yaml"
    config.write_text(
        """
thresholds:
  monitor_max_attack_probability: 0.35
  divert_max_attack_probability: 0.75
  isolate_min_attack_probability: 0.75
weights:
  attack_probability: 0.5
  target_criticality: 0.3
  intel_value: 0.2
intel_value:
  monitor: 0.1
  divert: 0.8
  isolate: 0.3
""",
        encoding="utf-8",
    )

    policy = RulePolicy(config)
    low = policy.decide({"attack_probability": 0.2, "target_score": 0.1}, asset_criticality=1)
    mid = policy.decide({"attack_probability": 0.5, "target_score": 0.3}, asset_criticality=2)
    high = policy.decide({"attack_probability": 0.9, "target_score": 0.8}, asset_criticality=5)

    assert low.action == "monitor"
    assert mid.action == "divert"
    assert high.action == "isolate"