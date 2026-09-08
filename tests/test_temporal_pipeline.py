import pandas as pd

from preprocessing.windowing import add_time_windows
from graph.graph_builder import build_temporal_graphs


def test_temporal_pipeline_builds_graphs():
    df = pd.DataFrame(
        [
            {
                "network.timestamp": "2026-01-01 00:00:00",
                "flow.src_ip": "10.0.0.1",
                "flow.dst_ip": "10.0.0.2",
                "flow.src_port": 1234,
                "flow.dst_port": 80,
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
                "label.class": "BENIGN",
                "label.is_attack": 0,
            },
            {
                "network.timestamp": "2026-01-01 00:00:02",
                "flow.src_ip": "10.0.0.1",
                "flow.dst_ip": "10.0.0.3",
                "flow.src_port": 1234,
                "flow.dst_port": 443,
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
                "label.class": "ATTACK",
                "label.is_attack": 1,
            },
        ]
    )

    windowed = add_time_windows(df, window_seconds=5)
    snapshots = build_temporal_graphs(windowed)

    assert len(snapshots) == 1
    assert snapshots[0].graph.number_of_nodes() == 3
    assert snapshots[0].graph.number_of_edges() == 2