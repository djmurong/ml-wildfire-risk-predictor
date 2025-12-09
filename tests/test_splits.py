import pandas as pd
from pathlib import Path

def test_train_val_test_splits():
    p = Path("data/processed")
    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")
    # basic non-empty
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    # disjoint indexes (by default our split preserves distinct rows)
    ids_train = set(train.index)
    ids_val = set(val.index)
    ids_test = set(test.index)
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
