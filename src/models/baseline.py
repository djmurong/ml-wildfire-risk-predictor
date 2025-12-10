"""
Baseline models for wildfire prediction.
Implements simple baselines to compare against XGBoost models:
- P-model baseline: Logistic Regression for ignition classification
- A-model baseline: Linear Regression for log(burned_area) regression
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import spearmanr
from pathlib import Path
import sys

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models/final"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_feature_list(df, exclude_cols=None):
    """
    Get list of feature columns from data.
    
    Args:
        df: DataFrame
        exclude_cols: List of columns to exclude (targets, metadata, etc.)
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = [
            'filename', 'date', 'event_id', 'tile',
            'ignition', 'burned_area', 'log_burned_area',
            'year', 'month', 'day_of_year', 'day_of_week',
            'matched', 'split', 'sample_idx'  # Added split and sample_idx from dataloader
        ]
    
    # Get all columns except excluded ones
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return sorted(feature_cols)


def train_baseline_models(features=None):
    """
    Train baseline models for both P-model and A-model.
    
    Args:
        features: List of feature column names (if None, auto-detect from training data)
    
    Returns:
        Tuple of (p_baseline_model, a_baseline_model)
    """
    print("=" * 70)
    print("Training Baseline Models")
    print("=" * 70)
    
    # Load data
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    print(f"\nLoading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Training samples: {len(train_df)}")
    
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)
    print(f"Validation samples: {len(val_df)}")
    
    # Get features
    if features is None:
        features = _get_feature_list(train_df)
    
    print(f"\nUsing {len(features)} features")
    
    # Prepare data
    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    
    # Handle missing values
    if X_train.isnull().any().any():
        print("Handling missing values in training data...")
        X_train = X_train.fillna(X_train.median())
    
    if X_val.isnull().any().any():
        print("Handling missing values in validation data...")
        X_val = X_val.fillna(X_val.median())
    
    # ===================================================================
    # P-Model Baseline: Logistic Regression for Ignition Classification
    # ===================================================================
    print("\n" + "=" * 70)
    print("P-Model Baseline: Logistic Regression (Ignition Classification)")
    print("=" * 70)
    
    y_train_p = train_df['ignition'].values
    y_val_p = val_df['ignition'].values
    
    # Check for missing values
    mask_train = ~np.isnan(y_train_p)
    mask_val = ~np.isnan(y_val_p)
    X_train_p = X_train[mask_train].copy()
    y_train_p = y_train_p[mask_train]
    X_val_p = X_val[mask_val].copy()
    y_val_p = y_val_p[mask_val]
    
    print(f"Training samples (after removing NaNs): {len(X_train_p)}")
    print(f"Validation samples (after removing NaNs): {len(X_val_p)}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    p_baseline = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'  # Good for small to medium datasets
    )
    p_baseline.fit(X_train_p, y_train_p)
    
    # Make predictions
    pred_proba_p = p_baseline.predict_proba(X_val_p)[:, 1]
    pred_p = p_baseline.predict(X_val_p)
    
    # Calculate metrics
    print("\nValidation Metrics:")
    try:
        auc = roc_auc_score(y_val_p, pred_proba_p)
        pr_auc = average_precision_score(y_val_p, pred_proba_p)
        brier = brier_score_loss(y_val_p, pred_proba_p)
        accuracy = accuracy_score(y_val_p, pred_p)
        precision = precision_score(y_val_p, pred_p, zero_division=0)
        recall = recall_score(y_val_p, pred_p, zero_division=0)
        f1 = f1_score(y_val_p, pred_p, zero_division=0)
        
        print(f"  ROC-AUC:     {auc:.4f}")
        print(f"  PR-AUC:      {pr_auc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
    except Exception as e:
        print(f"Warning: Could not calculate all classification metrics: {e}")
    
    # ===================================================================
    # A-Model Baseline: Linear Regression for Log Burned Area
    # ===================================================================
    print("\n" + "=" * 70)
    print("A-Model Baseline: Linear Regression (Log Burned Area Regression)")
    print("=" * 70)
    
    y_train_a = train_df['log_burned_area'].values
    y_val_a = val_df['log_burned_area'].values
    
    # Filter to only samples with actual burned area
    mask_train = ~np.isnan(y_train_a) & (train_df['burned_area'].values > 0)
    mask_val = ~np.isnan(y_val_a) & (val_df['burned_area'].values > 0)
    
    X_train_a = X_train[mask_train].copy()
    y_train_a = y_train_a[mask_train]
    X_val_a = X_val[mask_val].copy()
    y_val_a = y_val_a[mask_val]
    
    print(f"Training samples (with burned area > 0): {len(X_train_a)}")
    print(f"Validation samples (with burned area > 0): {len(X_val_a)}")
    
    if len(X_train_a) == 0:
        print("Warning: No training samples with burned area > 0. Skipping A-model baseline.")
        return p_baseline, None
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    a_baseline = LinearRegression()
    a_baseline.fit(X_train_a, y_train_a)
    
    # Make predictions
    pred_a = a_baseline.predict(X_val_a)
    
    # Calculate metrics
    print("\nValidation Metrics:")
    mse = mean_squared_error(y_val_a, pred_a)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_a, pred_a)
    r2 = r2_score(y_val_a, pred_a)
    
    try:
        spearman_corr, spearman_p = spearmanr(y_val_a, pred_a)
        print(f"  RMSE:        {rmse:.4f}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  R²:          {r2:.4f}")
        print(f"  Spearman ρ:  {spearman_corr:.4f} (p={spearman_p:.4f})")
    except Exception as e:
        print(f"  RMSE:        {rmse:.4f}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  R²:          {r2:.4f}")
        print(f"Warning: Could not calculate Spearman correlation: {e}")
    
    # Save models and feature lists
    print("\n" + "=" * 70)
    print("Saving Baseline Models")
    print("=" * 70)
    
    import joblib
    
    p_baseline_path = MODELS_DIR / "baseline_classifier_ignition.pkl"
    joblib.dump(p_baseline, p_baseline_path)
    print(f"✓ P-model baseline saved to {p_baseline_path}")
    
    # Save feature list for P-model
    p_feature_list_path = MODELS_DIR / "features_used_baseline_classifier_ignition.txt"
    with open(p_feature_list_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"✓ P-model feature list saved to {p_feature_list_path}")
    
    if a_baseline is not None:
        a_baseline_path = MODELS_DIR / "baseline_regressor_log_burned_area.pkl"
        joblib.dump(a_baseline, a_baseline_path)
        print(f"✓ A-model baseline saved to {a_baseline_path}")
        
        # Save feature list for A-model
        a_feature_list_path = MODELS_DIR / "features_used_baseline_regressor_log_burned_area.txt"
        with open(a_feature_list_path, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        print(f"✓ A-model feature list saved to {a_feature_list_path}")
    
    print("\nBaseline training completed!")
    
    return p_baseline, a_baseline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of feature column names (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    try:
        train_baseline_models(features=args.features)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
