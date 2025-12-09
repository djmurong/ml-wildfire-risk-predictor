import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
import joblib
from pathlib import Path
import sys

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models/final"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_feature_list(train_df, exclude_cols=None):
    """
    Get list of feature columns from training data.
    
    Args:
        train_df: Training DataFrame
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
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    return sorted(feature_cols)


def train_xgb(
    features=None,
    tree_method='hist'
):
    """
    Train both P-model (ignition probability) and A-model (conditional log burned area).
    
    Args:
        features: List of feature column names (if None, auto-detect from training data)
        tree_method: 'hist' (CPU) or 'gpu_hist' (GPU)
    
    Returns:
        dict with 'p_model' and 'a_model' keys containing trained models
    """
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Get feature list if not provided
    if features is None:
        print("\nDetecting features from training data...")
        features = _get_feature_list(train_df)
        embedding_count = len([c for c in features if c.startswith('embedding_')])
        tif_count = len([c for c in features if c.startswith('tif_band_')])
        other_count = len(features) - embedding_count - tif_count
        print(f"  Found {len(features)} features:")
        print(f"    - {embedding_count} embedding features")
        print(f"    - {tif_count} tif band features")
        if other_count > 0:
            print(f"    - {other_count} other features")
    else:
        print(f"\nUsing {len(features)} explicitly provided features")
    
    print("\n" + "="*70)
    print("TRAINING BOTH MODELS: P-model (Ignition) + A-model (Log Burned Area)")
    print("="*70)
    
    models = {}
    
    # Train P-model (ignition classifier)
    print("\n" + "="*70)
    print("P-MODEL: Ignition Probability (Binary Classification)")
    print("="*70)
    p_model = _train_single_model(
        train_df, val_df, features, 'ignition', 'classifier', tree_method
    )
    models['p_model'] = p_model
    
    # Train A-model (log_burned_area regressor)
    print("\n" + "="*70)
    print("A-MODEL: Conditional Log Burned Area (Regression)")
    print("="*70)
    a_model = _train_single_model(
        train_df, val_df, features, 'log_burned_area', 'regressor', tree_method
    )
    models['a_model'] = a_model
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Both models saved")
    print("="*70)
    
    return models


def _train_single_model(train_df, val_df, features, target_col, model_type, tree_method):
    """
    Train a single XGBoost model.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        features: List of feature column names
        target_col: Name of target column
        model_type: 'regressor' or 'classifier'
        tree_method: 'hist' (CPU) or 'gpu_hist' (GPU)
    
    Returns:
        Trained model
    """
    # Check if target exists
    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(train_df.columns)}"
        )
    
    # Check if all features exist
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Features not found: {missing}")
    
    print(f"Using {len(features)} features")
    
    # Extract features and target
    X_train = train_df[features].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[features].copy()
    y_val = val_df[target_col].copy()
    
    # Handle missing values
    if X_train.isnull().any().any():
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())
    
    if y_train.isnull().any():
        mask = ~y_train.isnull()
        X_train = X_train[mask]
        y_train = y_train[mask]
    
    # Calculate class weights for imbalanced classification
    scale_pos_weight = None
    if model_type == 'classifier':
        # Calculate imbalance ratio: negative_samples / positive_samples
        positive_count = (y_train == 1).sum()
        negative_count = (y_train == 0).sum()
        
        if positive_count > 0 and negative_count > 0:
            scale_pos_weight = negative_count / positive_count
            positive_pct = (positive_count / len(y_train)) * 100
            
            print(f"\nClass balance:")
            print(f"  Positive (ignition=1): {positive_count:,} ({positive_pct:.2f}%)")
            print(f"  Negative (ignition=0): {negative_count:,} ({100-positive_pct:.2f}%)")
            print(f"  Imbalance ratio: {scale_pos_weight:.2f}:1")
            
            if positive_pct < 5:
                print(f"  ⚠️  Imbalanced dataset - using scale_pos_weight={scale_pos_weight:.2f}")
            elif positive_pct < 20:
                print(f"  → Moderately imbalanced - using scale_pos_weight={scale_pos_weight:.2f}")
            else:
                print(f"  ✓ Reasonably balanced - scale_pos_weight={scale_pos_weight:.2f}")
    
    # Create and train model
    # Note: early_stopping_rounds must be in constructor for newer XGBoost versions
    # 20 rounds is a good balance: stops if no improvement, but allows some exploration
    early_stopping_rounds = 20
    
    if model_type == 'classifier':
        model = XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            tree_method=tree_method,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            eval_metric='logloss',  # Use loss for early stopping (what model optimizes)
            early_stopping_rounds=early_stopping_rounds
        )
    else:  # regressor
        model = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            tree_method=tree_method,
            eval_metric='rmse',  # Explicitly use RMSE (loss) for early stopping
            early_stopping_rounds=early_stopping_rounds
        )
    
    print(f"\nTraining {model_type}...")
    print(f"  Early stopping: stops if validation metric doesn't improve for {early_stopping_rounds} rounds")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make predictions
    preds = model.predict(X_val)
    if model_type == 'classifier':
        pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    print("\n" + "-"*70)
    print(f"Validation Metrics ({target_col})")
    print("-"*70)
    
    if model_type == 'classifier':
        try:
            auc = roc_auc_score(y_val, pred_proba)
            pr_auc = average_precision_score(y_val, pred_proba)
            brier = brier_score_loss(y_val, pred_proba)
            print(f"ROC-AUC: {auc:.4f}")
            print(f"PR-AUC:  {pr_auc:.4f}")
            print(f"Brier Score: {brier:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate classification metrics: {e}")
    else:
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
    
    # Save model
    model_name = f"xgb_wildfire_{model_type}_{target_col}.pkl"
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save feature list used for this model
    feature_list_path = MODELS_DIR / f"features_used_{model_type}_{target_col}.txt"
    with open(feature_list_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"Feature list saved to {feature_list_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train both P-model (ignition probability) and A-model (conditional log burned area)'
    )
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of feature column names (if not provided, auto-detect from training data)')
    parser.add_argument('--tree-method', choices=['hist', 'gpu_hist'], default='hist',
                        help='Tree method: hist (CPU) or gpu_hist (GPU)')
    
    args = parser.parse_args()
    
    try:
        train_xgb(
            features=args.features,
            tree_method=args.tree_method
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
