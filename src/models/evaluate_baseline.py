"""
Evaluate baseline models and combine their predictions to compute hazard scores.

This evaluates:
- P-model baseline: Logistic Regression for ignition classification
- A-model baseline: Linear Regression for log(burned_area) regression
- Combined: hazard_score = P(ignition) × exp(log_burned_area)
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import spearmanr

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models/final"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models/final/predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_feature_list(df, exclude_cols=None):
    """Get feature list excluding targets and metadata."""
    if exclude_cols is None:
        exclude_cols = [
            'filename', 'date', 'event_id', 'tile',
            'ignition', 'burned_area', 'log_burned_area',
            'year', 'month', 'day_of_year', 'day_of_week',
            'matched', 'split', 'sample_idx'  # Added split and sample_idx from dataloader
        ]
    return [col for col in df.columns if col not in exclude_cols]


def evaluate_baseline_models(data_path=None, output_path=None):
    """
    Evaluate baseline models and combine predictions to compute hazard scores.
    
    Args:
        data_path: Path to data CSV (default: test.csv)
        output_path: Path to save predictions (default: models/final/predictions/baseline_hazard_scores.csv)
    """
    if data_path is None:
        data_path = PROCESSED_DIR / "test.csv"
    
    if output_path is None:
        output_path = OUTPUT_DIR / "baseline_hazard_scores.csv"
    
    print("=" * 70)
    print("Evaluating Baseline Models and Combining Predictions")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data samples: {len(df)}")
    
    # Load baseline models
    print("\nLoading baseline models...")
    
    # P-model baseline (classifier)
    p_baseline_path = MODELS_DIR / "baseline_classifier_ignition.pkl"
    if not p_baseline_path.exists():
        raise FileNotFoundError(
            f"P-model baseline not found: {p_baseline_path}\n"
            "Please run: python src/models/baseline.py"
        )
    p_baseline = joblib.load(p_baseline_path)
    print(f"  ✓ Loaded P-model baseline from {p_baseline_path}")
    
    # A-model baseline (regressor)
    a_baseline_path = MODELS_DIR / "baseline_regressor_log_burned_area.pkl"
    if not a_baseline_path.exists():
        raise FileNotFoundError(
            f"A-model baseline not found: {a_baseline_path}\n"
            "Please run: python src/models/baseline.py"
        )
    a_baseline = joblib.load(a_baseline_path)
    print(f"  ✓ Loaded A-model baseline from {a_baseline_path}")
    
    # Get features from saved feature lists (baseline models don't store feature names)
    print("\nLoading feature names from saved lists...")
    
    p_feature_list_path = MODELS_DIR / "features_used_baseline_classifier_ignition.txt"
    a_feature_list_path = MODELS_DIR / "features_used_baseline_regressor_log_burned_area.txt"
    
    if p_feature_list_path.exists():
        with open(p_feature_list_path, 'r') as f:
            p_features = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Loaded P-model features from {p_feature_list_path} ({len(p_features)} features)")
    else:
        raise FileNotFoundError(
            f"P-model feature list not found: {p_feature_list_path}\n"
            "Please run: python src/models/baseline.py"
        )
    
    if a_feature_list_path.exists():
        with open(a_feature_list_path, 'r') as f:
            a_features = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Loaded A-model features from {a_feature_list_path} ({len(a_features)} features)")
    else:
        raise FileNotFoundError(
            f"A-model feature list not found: {a_feature_list_path}\n"
            "Please run: python src/models/baseline.py"
        )
    
    # Check if features match (they should be the same)
    if set(p_features) != set(a_features):
        print("Warning: P-model and A-model have different feature sets")
        # Use intersection
        common_features = [f for f in p_features if f in a_features]
        print(f"Using common features: {len(common_features)}")
        p_features = [f for f in p_features if f in common_features]
        a_features = [f for f in a_features if f in common_features]
    else:
        # Use P-model features (they should be the same)
        features = p_features
    
    # Check if all features exist in data
    missing_p = [f for f in p_features if f not in df.columns]
    missing_a = [f for f in a_features if f not in df.columns]
    
    if missing_p:
        raise ValueError(
            f"P-model features not found in data: {missing_p[:10]}{'...' if len(missing_p) > 10 else ''}"
        )
    if missing_a:
        raise ValueError(
            f"A-model features not found in data: {missing_a[:10]}{'...' if len(missing_a) > 10 else ''}"
        )
    
    # Prepare data - use exact feature order from saved lists
    X_p = df[p_features].copy()
    X_a = df[a_features].copy()
    
    print(f"\nUsing {len(p_features)} features for P-model")
    print(f"Using {len(a_features)} features for A-model")
    
    # Handle missing values
    if X_p.isnull().any().any() or X_a.isnull().any().any():
        print("\nHandling missing values...")
        X_p = X_p.fillna(X_p.median())
        X_a = X_a.fillna(X_a.median())
    
    # Make predictions
    print("\nMaking predictions...")
    
    # P-model baseline: probability of ignition
    # Use features in exact order from saved list
    p_ignition = p_baseline.predict_proba(X_p)[:, 1]  # Probability of class 1 (ignition)
    print(f"  ✓ P-model baseline predictions: {len(p_ignition)} samples")
    print(f"     Ignition probability range: [{p_ignition.min():.4f}, {p_ignition.max():.4f}]")
    
    # A-model baseline: conditional log(burned_area) given ignition
    # Use features in exact order from saved list
    log_burned_area = a_baseline.predict(X_a)
    print(f"  ✓ A-model baseline predictions: {len(log_burned_area)} samples")
    print(f"     Log(burned_area) range: [{log_burned_area.min():.4f}, {log_burned_area.max():.4f}]")
    
    # Combine: hazard = p_ignition × exp(log_burned_area)
    burned_area = np.exp(log_burned_area)
    hazard_score = p_ignition * burned_area
    
    print(f"  ✓ Combined hazard scores: {len(hazard_score)} samples")
    print(f"     Hazard score range: [{hazard_score.min():.4f}, {hazard_score.max():.4f}]")
    
    # Create results DataFrame
    results = pd.DataFrame({
        'p_ignition': p_ignition,
        'log_burned_area': log_burned_area,
        'burned_area': burned_area,
        'hazard_score': hazard_score
    })
    
    # Add actual values if available
    if 'ignition' in df.columns:
        results['actual_ignition'] = df['ignition'].values
    if 'burned_area' in df.columns:
        results['actual_burned_area'] = df['burned_area'].values
    if 'log_burned_area' in df.columns:
        results['actual_log_burned_area'] = df['log_burned_area'].values
    
    # Evaluate P-model baseline
    print("\n" + "=" * 70)
    print("P-Model Baseline Evaluation (Ignition Classification)")
    print("=" * 70)
    
    if 'actual_ignition' in results.columns:
        y_test_p = results['actual_ignition'].values
        mask_p = ~np.isnan(y_test_p)
        
        if mask_p.sum() > 0:
            y_test_p = y_test_p[mask_p]
            pred_proba_p = p_ignition[mask_p]
            pred_p = (pred_proba_p > 0.5).astype(int)
            
            try:
                auc = roc_auc_score(y_test_p, pred_proba_p)
                pr_auc = average_precision_score(y_test_p, pred_proba_p)
                brier = brier_score_loss(y_test_p, pred_proba_p)
                accuracy = accuracy_score(y_test_p, pred_p)
                precision = precision_score(y_test_p, pred_p, zero_division=0)
                recall = recall_score(y_test_p, pred_p, zero_division=0)
                f1 = f1_score(y_test_p, pred_p, zero_division=0)
                
                print(f"\nTest Set Metrics:")
                print(f"  ROC-AUC:     {auc:.4f}")
                print(f"  PR-AUC:      {pr_auc:.4f}")
                print(f"  Brier Score: {brier:.4f}")
                print(f"  Accuracy:    {accuracy:.4f}")
                print(f"  Precision:   {precision:.4f}")
                print(f"  Recall:      {recall:.4f}")
                print(f"  F1-Score:    {f1:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate all classification metrics: {e}")
        else:
            print("No actual ignition labels available for evaluation")
    else:
        print("No actual ignition labels in data")
    
    # Evaluate A-model baseline
    print("\n" + "=" * 70)
    print("A-Model Baseline Evaluation (Log Burned Area Regression)")
    print("=" * 70)
    
    if 'actual_log_burned_area' in results.columns and 'actual_burned_area' in results.columns:
        y_test_a = results['actual_log_burned_area'].values
        mask_a = ~np.isnan(y_test_a) & (results['actual_burned_area'].values > 0)
        
        if mask_a.sum() > 0:
            y_test_a = y_test_a[mask_a]
            pred_a = log_burned_area[mask_a]
            
            mse = mean_squared_error(y_test_a, pred_a)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_a, pred_a)
            r2 = r2_score(y_test_a, pred_a)
            
            try:
                spearman_corr, spearman_p = spearmanr(y_test_a, pred_a)
                
                print(f"\nTest Set Metrics:")
                print(f"  RMSE:        {rmse:.4f}")
                print(f"  MAE:         {mae:.4f}")
                print(f"  R²:          {r2:.4f}")
                print(f"  Spearman ρ:  {spearman_corr:.4f} (p={spearman_p:.4f})")
            except Exception as e:
                print(f"\nTest Set Metrics:")
                print(f"  RMSE:        {rmse:.4f}")
                print(f"  MAE:         {mae:.4f}")
                print(f"  R²:          {r2:.4f}")
                print(f"Warning: Could not calculate Spearman correlation: {e}")
        else:
            print("No actual burned area data available for evaluation")
    else:
        print("No actual burned area labels in data")
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"\n✓ Baseline predictions saved to {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(results[['p_ignition', 'log_burned_area', 'burned_area', 'hazard_score']].describe())
    
    # If actual values available, compute errors
    if 'actual_ignition' in results.columns and 'actual_burned_area' in results.columns:
        print("\n" + "=" * 70)
        print("Prediction Accuracy (where actual values available)")
        print("=" * 70)
        
        # Ignition prediction accuracy
        ignition_mask = results['actual_ignition'].notna()
        if ignition_mask.sum() > 0:
            ignition_pred_binary = (results.loc[ignition_mask, 'p_ignition'] > 0.5).astype(int)
            ignition_accuracy = (ignition_pred_binary == results.loc[ignition_mask, 'actual_ignition']).mean()
            print(f"Ignition prediction accuracy: {ignition_accuracy:.4f}")
        
        # Burned area prediction error
        area_mask = results['actual_burned_area'].notna() & (results['actual_burned_area'] > 0)
        if area_mask.sum() > 0:
            mae = np.abs(results.loc[area_mask, 'burned_area'] - results.loc[area_mask, 'actual_burned_area']).mean()
            rmse = np.sqrt(((results.loc[area_mask, 'burned_area'] - results.loc[area_mask, 'actual_burned_area']) ** 2).mean())
            print(f"Burned area MAE: {mae:.4f} km²")
            print(f"Burned area RMSE: {rmse:.4f} km²")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate baseline models and combine predictions')
    parser.add_argument('--data', default=None,
                        help='Path to data CSV (default: data/processed/test.csv)')
    parser.add_argument('--output', default=None,
                        help='Path to save predictions (default: models/final/predictions/baseline_hazard_scores.csv)')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_baseline_models(
            data_path=args.data,
            output_path=args.output
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

