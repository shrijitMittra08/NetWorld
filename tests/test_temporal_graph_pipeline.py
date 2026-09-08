import pandas as pd

from preprocessing.windowing import add_time_windows
from graph import (
    build_temporal_graphs,
    save_temporal_sequence,
    load_temporal_sequence,
    temporal_snapshot_to_pyg,
)


def test_temporal_graph_sequence_and_serialization(tmp_path):
    df = pd.DataFrame(
        [
            {
                "network.timestamp": "2026-01-01 00:00:00",
                "flow.src_ip": "10.0.0.1",
                "flow.dst_ip": "10.0.0.2",
                "flow.protocol": "TCP",
                "flow.duration": 1.2,
                "flow.flow_bytes_per_sec": 100.0,
                "flow.flow_packets_per_sec": 10.0,
                "host.src_role": "workstation",
                "host.dst_role": "server",
                "host.src_subnet": "10.0.0.0/24",
                "host.dst_subnet": "10.0.1.0/24",
                "host.src_os": "Windows",
                "host.dst_os": "Linux",
                "behaviour.new_peers": 1,
                "behaviour.auth_failures": 0,
                "label.class": "BENIGN",
                "label.is_attack": 0,
            },
            {
                "network.timestamp": "2026-01-01 00:00:02",
                "flow.src_ip": "10.0.0.1",
                "flow.dst_ip": "10.0.0.3",
                "flow.protocol": "TCP",
                "flow.duration": 0.8,
                "flow.flow_bytes_per_sec": 120.0,
                "flow.flow_packets_per_sec": 12.0,
                "host.src_role": "workstation",
                "host.dst_role": "server",
                "host.src_subnet": "10.0.0.0/24",
                "host.dst_subnet": "10.0.1.0/24",
                "host.src_os": "Windows",
                "host.dst_os": "Linux",
                "behaviour.new_peers": 2,
                "behaviour.auth_failures": 1,
                "label.class": "ATTACK",
                "label.is_attack": 1,
            },
        ]
    )

    windowed = add_time_windows(df, window_seconds=5)
    sequence = build_temporal_graphs(windowed)

    assert len(sequence) == 1
    snapshot = sequence.latest()
    assert snapshot is not None
    assert snapshot.graph.number_of_nodes() >= 2
    assert snapshot.graph.number_of_edges() >= 1

    out = save_temporal_sequence(sequence, tmp_path / "sequence.json")
    loaded = load_temporal_sequence(out)
    assert len(loaded) == 1

    pyg_data = temporal_snapshot_to_pyg(loaded.latest())
    assert "edge_index" in pyg_data
    assert "node_attr" in pyg_data