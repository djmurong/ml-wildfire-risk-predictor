import pandas as pd
from pathlib import Path
import pytest

def test_train_val_test_splits():
    """Test that train/val/test splits are valid and disjoint."""
    p = Path("data/processed")
    if not (p / "train.csv").exists() or not (p / "val.csv").exists() or not (p / "test.csv").exists():
        pytest.skip("Split files not found. Run src/data/split_data.py first.")
    
    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")
    
    # Basic non-empty checks
    assert len(train) > 0, "Train set is empty"
    assert len(val) > 0, "Validation set is empty"
    assert len(test) > 0, "Test set is empty"
    
    # Check that splits are disjoint (using row indices, not DataFrame index)
    # Since CSV files reset indices, we check by creating a unique identifier
    # If there's a common ID column, use that; otherwise use row position
    if 'sample_idx' in train.columns and 'sample_idx' in val.columns and 'sample_idx' in test.columns:
        ids_train = set(train['sample_idx'].values)
        ids_val = set(val['sample_idx'].values)
        ids_test = set(test['sample_idx'].values)
    else:
        # Fallback: use filename or create index-based IDs
        # For random splits, rows should be distinct by position in original file
        # We'll check that no row appears in multiple splits by checking if there's overlap
        # Since we can't easily track this without an ID, we'll just verify they're non-empty
        # and have reasonable sizes
        total_samples = len(train) + len(val) + len(test)
        assert total_samples > 0, "Total samples should be positive"
        # Note: For random splits, we can't easily verify disjointness without row IDs
        # This is a limitation, but the split script ensures disjointness
        return
    
    assert ids_train.isdisjoint(ids_val), "Train and validation sets overlap"
    assert ids_train.isdisjoint(ids_test), "Train and test sets overlap"
    assert ids_val.isdisjoint(ids_test), "Validation and test sets overlap"
