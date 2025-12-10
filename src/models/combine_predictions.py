"""
Combine P-model and A-model predictions to compute hazard scores.

Hazard Score = P(ignition) × E[burned_area | ignition]

This implements the two-head approach where:
- P-model predicts probability of ignition
- A-model predicts conditional log(burned_area) given ignition
- Combined: hazard = p_ignition × exp(log_burned_area)
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def load_feature_list(model_type, target_col):
    """Load feature list from saved file."""
    feature_file = MODELS_DIR / f"features_used_{model_type}_{target_col}.txt"
    if not feature_file.exists():
        return None
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def combine_predictions(data_path=None, output_path=None):
    """
    Combine P-model and A-model predictions to compute hazard scores.
    
    Args:
        data_path: Path to data CSV (default: test.csv)
        output_path: Path to save predictions (default: models/final/predictions/hazard_scores.csv)
    """
    if data_path is None:
        data_path = PROCESSED_DIR / "test.csv"
    
    if output_path is None:
        output_path = OUTPUT_DIR / "hazard_scores.csv"
    
    print("=" * 70)
    print("Combining P-Model and A-Model Predictions")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data samples: {len(df)}")
    
    # Load models
    print("\nLoading models...")
    
    # P-model (classifier)
    p_model_path = MODELS_DIR / "xgb_wildfire_classifier_ignition.pkl"
    if not p_model_path.exists():
        raise FileNotFoundError(f"P-model not found: {p_model_path}")
    p_model = joblib.load(p_model_path)
    print(f"  ✓ Loaded P-model from {p_model_path}")
    
    # A-model (regressor)
    a_model_path = MODELS_DIR / "xgb_wildfire_regressor_log_burned_area.pkl"
    if not a_model_path.exists():
        raise FileNotFoundError(f"A-model not found: {a_model_path}")
    a_model = joblib.load(a_model_path)
    print(f"  ✓ Loaded A-model from {a_model_path}")
    
    # Get features from saved lists, but verify order with models
    print("\nLoading feature names...")
    
    # Try to load from saved feature lists first
    p_features_saved = load_feature_list('classifier', 'ignition')
    a_features_saved = load_feature_list('regressor', 'log_burned_area')
    
    # Get model's expected feature order (for verification)
    p_model_features = None
    if hasattr(p_model, 'feature_names_in_'):
        p_model_features = list(p_model.feature_names_in_)
    elif hasattr(p_model, 'get_booster'):
        try:
            p_model_features = p_model.get_booster().feature_names
        except:
            pass
    
    a_model_features = None
    if hasattr(a_model, 'feature_names_in_'):
        a_model_features = list(a_model.feature_names_in_)
    elif hasattr(a_model, 'get_booster'):
        try:
            a_model_features = a_model.get_booster().feature_names
        except:
            pass
    
    # Use saved list if available, otherwise use model's features
    if p_features_saved is not None:
        print("  ✓ Loaded P-model features from saved list")
        # Reorder saved features to match model's expected order
        if p_model_features is not None:
            # Reorder saved features to match model order
            p_features = [f for f in p_model_features if f in p_features_saved]
            # Add any features in saved list but not in model (shouldn't happen, but be safe)
            p_features.extend([f for f in p_features_saved if f not in p_model_features])
        else:
            p_features = p_features_saved
    elif p_model_features is not None:
        print("  ✓ Using P-model features from model")
        p_features = p_model_features
    else:
        raise ValueError(
            "Cannot determine P-model features. Please run:\n"
            "  python scripts/save_feature_lists_from_model.py\n"
            "or ensure the model has feature_names_in_ attribute."
        )
    
    if a_features_saved is not None:
        print("  ✓ Loaded A-model features from saved list")
        # Reorder saved features to match model's expected order
        if a_model_features is not None:
            a_features = [f for f in a_model_features if f in a_features_saved]
            a_features.extend([f for f in a_features_saved if f not in a_model_features])
        else:
            a_features = a_features_saved
    elif a_model_features is not None:
        print("  ✓ Using A-model features from model")
        a_features = a_model_features
    else:
        raise ValueError(
            "Cannot determine A-model features. Please run:\n"
            "  python scripts/save_feature_lists_from_model.py\n"
            "or ensure the model has feature_names_in_ attribute."
        )
    
    print(f"  P-model expects {len(p_features)} features")
    print(f"  A-model expects {len(a_features)} features")
    
    # Check if models have the same features
    p_features_set = set(p_features)
    a_features_set = set(a_features)
    
    if p_features_set == a_features_set:
        # Models use same features - use them in P-model order
        common_features = p_features
        print(f"  ✓ Both models use the same {len(common_features)} features")
        X_p = df[p_features].copy()
        X_a = df[a_features].copy()
    else:
        # Models have different features - use intersection
        common_features = [f for f in p_features if f in a_features]
        print(f"\nWarning: Models have different feature sets")
        print(f"  P-model features: {len(p_features)}")
        print(f"  A-model features: {len(a_features)}")
        print(f"  Common features: {len(common_features)}")
        
        # Check which features are missing
        p_only = p_features_set - a_features_set
        a_only = a_features_set - p_features_set
        if p_only:
            print(f"  Features only in P-model: {list(p_only)[:5]}{'...' if len(p_only) > 5 else ''}")
        if a_only:
            print(f"  Features only in A-model: {list(a_only)[:5]}{'...' if len(a_only) > 5 else ''}")
        
        # Use common features, but in the order each model expects
        X_p = df[[f for f in p_features if f in common_features]].copy()
        X_a = df[[f for f in a_features if f in common_features]].copy()
    
    # Check if all required features exist in data
    missing_p = [f for f in X_p.columns if f not in df.columns]
    missing_a = [f for f in X_a.columns if f not in df.columns]
    
    if missing_p:
        raise ValueError(
            f"P-model features not found in data: {missing_p[:10]}{'...' if len(missing_p) > 10 else ''}"
        )
    if missing_a:
        raise ValueError(
            f"A-model features not found in data: {missing_a[:10]}{'...' if len(missing_a) > 10 else ''}"
        )
    
    # Handle missing values
    if X_p.isnull().any().any() or X_a.isnull().any().any():
        print("\nHandling missing values...")
        X_p = X_p.fillna(X_p.median())
        X_a = X_a.fillna(X_a.median())
    
    # Make predictions
    print("\nMaking predictions...")
    
    # P-model: probability of ignition
    # Use features in the exact order P-model expects
    p_ignition = p_model.predict_proba(X_p)[:, 1]  # Probability of class 1 (ignition)
    print(f"  ✓ P-model predictions: {len(p_ignition)} samples")
    print(f"     Ignition probability range: [{p_ignition.min():.4f}, {p_ignition.max():.4f}]")
    
    # A-model: conditional log(burned_area) given ignition
    # Use features in the exact order A-model expects
    log_burned_area = a_model.predict(X_a)
    print(f"  ✓ A-model predictions: {len(log_burned_area)} samples")
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
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(results.describe())
    
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
            print(f"Burned area MAE: {mae:.4f}")
            print(f"Burned area RMSE: {rmse:.4f}")
        
        # Combined hazard score evaluation
        print("\n" + "=" * 70)
        print("Combined Hazard Score Evaluation")
        print("=" * 70)
        
        # Compute actual hazard: 0 if no ignition, burned_area if ignition occurred
        results['actual_hazard'] = np.where(
            results['actual_ignition'] == 1,
            results['actual_burned_area'],
            0.0
        )
        
        # Filter to samples with valid actual values
        valid_mask = results['actual_ignition'].notna() & results['actual_burned_area'].notna()
        if valid_mask.sum() > 0:
            y_pred = results.loc[valid_mask, 'hazard_score'].values
            y_true = results.loc[valid_mask, 'actual_hazard'].values
            
            # Overall metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Spearman correlation (rank correlation)
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            
            print(f"\nOverall Metrics (all {valid_mask.sum()} samples):")
            print(f"  RMSE: {rmse:.4f} hectares")
            print(f"  MAE:  {mae:.4f} hectares")
            print(f"  R²:   {r2:.4f}")
            print(f"  Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine P-model and A-model predictions')
    parser.add_argument('--data', default=None,
                        help='Path to data CSV (default: data/processed/test.csv)')
    parser.add_argument('--output', default=None,
                        help='Path to save predictions (default: models/final/predictions/hazard_scores.csv)')
    
    args = parser.parse_args()
    
    try:
        results = combine_predictions(
            data_path=args.data,
            output_path=args.output
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

