import networkx as nx
import torch

from models.graph_encoder import GraphEncoder


def test_edge_features_affect_graph_embedding():
    graph1 = nx.DiGraph()

    graph1.add_node(
        "A",
        role="workstation",
        subnet="10.0.0.0/24",
        os="Windows",
    )

    graph1.add_node(
        "B",
        role="server",
        subnet="10.0.1.0/24",
        os="Linux",
    )

    graph1.add_edge(
        "A",
        "B",
        count=1,
        duration_mean=1.0,
        bytes_per_sec_mean=100.0,
        packets_per_sec_mean=10.0,
        fwd_packets_sum=10.0,
        bwd_packets_sum=5.0,
        ttl_mean=64.0,
        payload_size_mean=100.0,
        auth_failures_sum=0.0,
        connection_failures_sum=0.0,
    )

    graph2 = graph1.copy()

    graph2["A"]["B"]["bytes_per_sec_mean"] = 100000.0
    graph2["A"]["B"]["packets_per_sec_mean"] = 5000.0

    encoder = GraphEncoder(
        encoder_type="sage",
    )

    embedding1 = encoder(graph1)
    embedding2 = encoder(graph2)

    assert embedding1.shape == embedding2.shape
    assert embedding1.shape[-1] == 64
    assert not torch.allclose(
        embedding1,
        embedding2,
    )