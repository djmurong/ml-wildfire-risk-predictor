import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from scipy.stats import spearmanr
import joblib
from pathlib import Path
import sys

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models/final"


def _get_feature_list(test_df, exclude_cols=None):
    """
    Get list of feature columns from test data.
    
    Args:
        test_df: Test DataFrame
        exclude_cols: List of columns to exclude (targets, metadata, etc.)
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = [
            'filename', 'date', 'event_id', 'tile',
            'ignition', 'burned_area', 'log_burned_area',
            'year', 'month', 'day_of_year', 'day_of_week',
            'matched'
        ]
    
    # Get all columns except excluded ones
    features = [col for col in test_df.columns if col not in exclude_cols]
    return features


def evaluate_model(
    model_path,
    features=None,
    target_col=None,
    model_type='regressor',
    test_data_path=None
):
    """
    Evaluate trained XGBoost model on test data.
    
    Args:
        model_path: Path to model file
        features: List of feature column names
        target_col: Target column name
        model_type: 'regressor' or 'classifier'
        test_data_path: Path to test data (None = use default)
    
    Returns:
        Dictionary of evaluation metrics
    """
    if test_data_path is None:
        test_data_path = PROCESSED_DIR / "test.csv"
    
    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    print(f"Test samples: {len(test_df)}")
    
    # Auto-detect target if not provided
    if target_col is None:
        # Try to infer from model filename
        model_name = Path(model_path).stem
        if 'ignition' in model_name:
            target_col = 'ignition'
        elif 'log_burned_area' in model_name or 'burned_area' in model_name:
            target_col = 'log_burned_area'
        else:
            raise ValueError("Cannot auto-detect target. Please specify --target")
    
    # Auto-detect features if not provided
    if features is None:
        # Try to load from saved feature list
        model_name = Path(model_path).stem
        # Extract model type and target from filename
        if 'classifier' in model_name:
            model_type_auto = 'classifier'
            target_auto = 'ignition'
        else:
            model_type_auto = 'regressor'
            target_auto = 'log_burned_area'
        
        feature_list_path = MODELS_DIR / f"features_used_{model_type_auto}_{target_auto}.txt"
        if feature_list_path.exists():
            print(f"Loading feature list from {feature_list_path}...")
            with open(feature_list_path, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
        else:
            # Auto-detect from test data
            print("Auto-detecting features from test data...")
            features = _get_feature_list(test_df)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Get feature names from the model (this ensures exact match with training)
    if hasattr(model, 'feature_names_in_'):
        # XGBoost 1.6+ stores feature names here
        model_features = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        # Older XGBoost versions
        try:
            model_features = model.get_booster().feature_names
        except:
            model_features = None
    else:
        model_features = None
    
    # Use model's feature names if available, otherwise use provided/auto-detected
    if model_features is not None:
        print(f"Using feature names from model ({len(model_features)} features)")
        features = model_features
    else:
        print(f"Warning: Could not extract feature names from model. Using provided/auto-detected features.")
    
    # Check if target exists
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(test_df.columns)}"
        )
    
    # Check if all features exist in test data
    missing = [f for f in features if f not in test_df.columns]
    if missing:
        raise ValueError(
            f"Features not found in test data: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"Model expects {len(features)} features, but {len(missing)} are missing.\n"
            f"Available test columns: {len(test_df.columns)}"
        )
    
    print(f"Using {len(features)} features (matching model's expected features)")
    
    # Extract features and target - use exact order from model
    X_test = test_df[features].copy()
    y_test = test_df[target_col].copy()
    
    # Handle missing values
    if X_test.isnull().any().any():
        X_test = X_test.fillna(X_test.median())
    
    if y_test.isnull().any():
        mask = ~y_test.isnull()
        X_test = X_test[mask]
        y_test = y_test[mask]
    
    if len(X_test) == 0:
        raise ValueError("No test data available after preprocessing")
    
    # Make predictions
    print("\nMaking predictions...")
    preds = model.predict(X_test)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("Test Set Evaluation Metrics")
    print("="*70)
    
    metrics = {}
    
    if model_type == 'classifier':
        # Classification metrics
        try:
            pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Binary classification metrics
            auc = roc_auc_score(y_test, pred_proba)
            pr_auc = average_precision_score(y_test, pred_proba)
            brier = brier_score_loss(y_test, pred_proba)
            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            print(f"ROC-AUC:     {auc:.4f}")
            print(f"PR-AUC:      {pr_auc:.4f}")
            print(f"Brier Score: {brier:.4f}")
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1-Score:    {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            print(f"\nConfusion Matrix:")
            print(f"  True Neg: {cm[0,0]:6d}  False Pos: {cm[0,1]:6d}")
            print(f"  False Neg: {cm[1,0]:6d}  True Pos:  {cm[1,1]:6d}")
            
            metrics = {
                'roc_auc': auc,
                'pr_auc': pr_auc,
                'brier_score': brier,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            print(f"Warning: Could not calculate all classification metrics: {e}")
    
    else:
        # Regression metrics
        # Calculate RMSE manually (squared=False may not be available in all sklearn versions)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Spearman correlation (for rank correlation)
        try:
            spearman_corr, spearman_p = spearmanr(y_test, preds)
            print(f"RMSE:        {rmse:.4f}")
            print(f"MAE:         {mae:.4f}")
            print(f"R²:          {r2:.4f}")
            print(f"Spearman ρ:  {spearman_corr:.4f} (p={spearman_p:.4f})")
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            }
        except Exception as e:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"R²:   {r2:.4f}")
            print(f"Warning: Could not calculate Spearman correlation: {e}")
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
    
    print("="*70)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate XGBoost model on test data')
    parser.add_argument('--model', required=True,
                        help='Path to model file')
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of feature column names (auto-detected if not provided)')
    parser.add_argument('--target', default=None,
                        help='Target column name (auto-detected from model filename if not provided)')
    parser.add_argument('--model-type', choices=['regressor', 'classifier'], default='regressor',
                        help='Model type')
    parser.add_argument('--test-data', default=None,
                        help='Path to test data CSV (default: data/processed/test.csv)')
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_model(
            model_path=args.model,
            features=args.features,
            target_col=args.target,
            model_type=args.model_type,
            test_data_path=args.test_data
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
