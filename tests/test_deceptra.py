from deceptra import DecoyManager, divert_session, SessionCapture

def test_decoy_manager():
    manager = DecoyManager()
    decoy = manager.register_decoy("db-01-clone", "database", "10.0.9.0/24", "Linux")
    assert decoy.decoy_id == "db-01-clone"
    assert len(manager.list_active_decoys()) == 1

def test_diversion_and_capture():
    diversion = divert_session("10.0.0.1:1234", "server-03", "server-03-clone")
    assert diversion.diverted is True
    assert diversion.decoy_target == "server-03-clone"

    capture = SessionCapture()
    capture.start_session("sess-1", diversion.source, diversion.decoy_target)
    capture.record_event("sess-1", {"cmd": "whoami"})
    session = capture.get_session("sess-1")

    assert session.events[0]["cmd"] == "whoami"