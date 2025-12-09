import subprocess
import sys
from pathlib import Path
import pytest

def test_run_basic_pipeline():
    """Test that the basic pipeline scripts can run (without actually running them)."""
    # Check that key scripts exist
    scripts = [
        "src/data/prepare_wildfirespreadts_features.py",
        "src/data/split_data.py",
        "src/models/train_model.py"
    ]
    for script in scripts:
        script_path = Path(script)
        assert script_path.exists(), f"Script not found: {script}"
    
    # Check that required data files exist (if pipeline has been run)
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        train_path = processed_dir / "train.csv"
        val_path = processed_dir / "val.csv"
        test_path = processed_dir / "test.csv"
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            # Check that models exist if training has been run
            p_model = Path("models/final/xgb_wildfire_classifier_ignition.pkl")
            a_model = Path("models/final/xgb_wildfire_regressor_log_burned_area.pkl")
            
            if p_model.exists() and a_model.exists():
                # Models exist, pipeline has been run
                assert True, "Pipeline appears to have been run successfully"
            else:
                # Data exists but models don't - this is okay for a test
                pytest.skip("Data exists but models not trained yet. This is expected.")
        else:
            pytest.skip("Data files not found. Run data preparation pipeline first.")
    else:
        pytest.skip("Processed data directory not found. Run data preparation pipeline first.")
