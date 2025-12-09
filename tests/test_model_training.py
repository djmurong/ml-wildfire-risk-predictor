import joblib
from pathlib import Path
import pytest
import pandas as pd

def test_xgb_models_exist_and_predict():
    """Test that both P-model and A-model exist and can make predictions."""
    # Check P-model (classifier)
    p_model_path = Path("models/final/xgb_wildfire_classifier_ignition.pkl")
    assert p_model_path.exists(), "P-model missing. Run src/models/train_model.py"
    
    # Check A-model (regressor)
    a_model_path = Path("models/final/xgb_wildfire_regressor_log_burned_area.pkl")
    assert a_model_path.exists(), "A-model missing. Run src/models/train_model.py"
    
    # Load models
    p_model = joblib.load(p_model_path)
    a_model = joblib.load(a_model_path)
    
    # Load test data
    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        pytest.skip("Test data not found. Run data preparation pipeline first.")
    
    df = pd.read_csv(test_path)
    
    # Get features from saved feature lists or model
    p_feature_path = Path("models/final/features_used_classifier_ignition.txt")
    a_feature_path = Path("models/final/features_used_regressor_log_burned_area.txt")
    
    if p_feature_path.exists():
        with open(p_feature_path, 'r') as f:
            p_features = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: get from model
        p_features = p_model.feature_names_in_ if hasattr(p_model, 'feature_names_in_') else []
    
    if a_feature_path.exists():
        with open(a_feature_path, 'r') as f:
            a_features = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: get from model
        a_features = a_model.feature_names_in_ if hasattr(a_model, 'feature_names_in_') else []
    
    # Use intersection if features differ
    common_features = [f for f in p_features if f in a_features and f in df.columns]
    
    if len(common_features) == 0:
        pytest.skip("No common features found between models and test data.")
    
    # Prepare test data (small sample)
    X = df[common_features].head(10).copy()
    
    # Handle missing values
    if X.isnull().any().any():
        X = X.fillna(X.median())
    
    # Make predictions
    p_preds = p_model.predict_proba(X)[:, 1]  # Probability of ignition
    a_preds = a_model.predict(X)  # Log burned area
    
    # Check predictions
    assert len(p_preds) == len(X), "P-model predictions length mismatch"
    assert len(a_preds) == len(X), "A-model predictions length mismatch"
    assert not any([pd.isnull(x) for x in p_preds]), "P-model has NaN predictions"
    assert not any([pd.isnull(x) for x in a_preds]), "A-model has NaN predictions"
