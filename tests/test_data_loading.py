import pytest
import pandas as pd
from pathlib import Path

def test_interim_embeddings_exists():
    """Test that CNN embeddings exist in interim folder."""
    p = Path("data/interim/wildfirespreadts_embeddings.parquet")
    if not p.exists():
        pytest.skip("Embeddings not found. Run src/data/extract_wildfirespreadts_embeddings.py first.")
    # Try to load (parquet file)
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(p).to_pandas()
        assert len(df) > 0, "Embeddings file is empty"
        # Check for embedding columns
        embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
        assert len(embedding_cols) > 0, "No embedding columns found"
    except ImportError:
        pytest.skip("pyarrow not available for reading parquet files")

def test_processed_features_exists():
    """Test that processed features.csv exists with required columns."""
    p = Path("data/processed/features.csv")
    if not p.exists():
        pytest.skip("features.csv missing. Run src/data/prepare_wildfirespreadts_features.py first.")
    df = pd.read_csv(p)
    # Check for required target columns
    assert 'ignition' in df.columns, "Missing 'ignition' column"
    assert 'burned_area' in df.columns, "Missing 'burned_area' column"
    assert 'log_burned_area' in df.columns, "Missing 'log_burned_area' column"
    # Check for some feature columns (embeddings or tif bands)
    feature_cols = [c for c in df.columns if c.startswith('embedding_') or c.startswith('tif_band_')]
    assert len(feature_cols) > 0, "No feature columns found (embedding_* or tif_band_*)"
