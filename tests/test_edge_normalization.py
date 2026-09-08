import torch

from models.graph_encoder import _normalize_edge_features

def test_edge_feature_normalization_reduces_large_scale():
    edge_attr = torch.tensor(
        [
            [
                1.0,
                1.0,
                100.0,
                10.0,
                10.0,
                5.0,
                64.0,
                100.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                100000.0,
                5000.0,
                10000.0,
                5000.0,
                64.0,
                10000.0,
                100.0,
                100.0,
            ],
        ],
        dtype=torch.float32,
    )

    normalized = _normalize_edge_features(edge_attr)

    assert normalized.shape == edge_attr.shape
    assert torch.isfinite(normalized).all()

    # Large traffic values should be compressed.
    assert normalized[1, 2] < edge_attr[1, 2]
    assert normalized[1, 3] < edge_attr[1, 3]

    # Zero values must remain zero.
    assert normalized[0, 8] == 0.0
    assert normalized[0, 9] == 0.0