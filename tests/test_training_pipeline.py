from pathlib import Path

def test_temporal_split():
    from training import temporal_split
    items = list(range(10))
    train, val, test = temporal_split(items, 0.6, 0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_early_stopping():
    from training import EarlyStopping
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False
    assert es.step(1.1) is False
    assert es.step(1.2) is True