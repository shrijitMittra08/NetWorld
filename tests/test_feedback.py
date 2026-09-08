from feedback import relabel_decoy_sessions, evaluate_retrain_trigger

def test_relabel_decoy_sessions():
    df = relabel_decoy_sessions(
        [
            {"flow.src_ip": "10.0.0.1", "label.attack_stage": "Lateral Movement"},
            {"flow.src_ip": "10.0.0.2"},
        ]
    )

    assert len(df) == 2
    assert df.loc[0, "label.class"] == "DECOY_MALICIOUS"
    assert df.loc[1, "label.is_attack"] == 1

def test_retrain_trigger():
    trigger = evaluate_retrain_trigger(new_decoy_rows=12, drift_score=0.1)
    assert trigger.should_retrain is True

    trigger2 = evaluate_retrain_trigger(new_decoy_rows=2, drift_score=0.8)
    assert trigger2.should_retrain is True