from deceptra import (
    DecoyManager,
    divert_session,
    build_iptables_dnat_rule,
    build_ovs_flow_rule,
    detect_evasion_signals,
    SessionCapture,
    evaluate_rollback,
    ImmutableDiversionLog,
    export_one_way_telemetry,
)


def test_diversion_builders():
    ipt = build_iptables_dnat_rule("10.0.0.1", "10.0.0.2", "10.0.9.2", dst_port=443)
    assert "DNAT" in ipt[0]

    ovs = build_ovs_flow_rule("1", "10.0.0.2", "10.0.9.2")
    assert "ovs-ofctl" in ovs


def test_evasion_and_rollback(tmp_path):
    assert detect_evasion_signals([{"msg": "This looks like a honeypot"}]) is True
    rb = evaluate_rollback(True, 0.1)
    assert rb.should_rollback is True
    assert rb.escalate_to_isolate is True

    log = ImmutableDiversionLog(tmp_path / "diversion.log")
    log.append({"action": "divert", "target": "server-03-clone"})
    records = log.read_all()
    assert records[0]["action"] == "divert"


def test_capture_and_telemetry():
    manager = DecoyManager()
    decoy = manager.register_decoy("db-01-clone", "database", "10.0.9.0/24", "Linux")

    capture = SessionCapture()
    sess = capture.start_session("s1", "10.0.0.1", decoy.decoy_id)
    capture.record_event("s1", {"cmd": "whoami"})
    capture.mark_evasion("s1")

    assert sess.evasion_detected is True

    tel = export_one_way_telemetry("decoy-net", "analysis-net", {"session": "s1"})
    assert tel.one_way is True