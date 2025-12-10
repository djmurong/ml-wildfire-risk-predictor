import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
from pathlib import Path
import sys
import torch
import random

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
            'matched', 'split', 'sample_idx'  # Added split and sample_idx from dataloader
        ]
    
    # Get all columns except excluded ones
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    return sorted(feature_cols)


def train_xgb(
    features=None,
    tree_method='gpu_hist',
    random_seed=42
):
    """
    Train both P-model (ignition probability) and A-model (conditional log burned area).
    
    Args:
        features: List of feature column names (if None, auto-detect from training data)
        tree_method: 'hist' (CPU) or 'gpu_hist' (GPU)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        dict with 'p_model' and 'a_model' keys containing trained models
    """
    # Set random seeds for full reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"Random seed set to {random_seed} for reproducibility")
    
    # Standard GPU detection using PyTorch (device = 'cuda' if available else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_available = torch.cuda.is_available()
    
    # Auto-detect GPU and set tree_method accordingly
    use_gpu = False
    if tree_method == 'gpu_hist':
        if gpu_available:
            # Use 'hist' with device='cuda' for GPU support (newer XGBoost versions)
            tree_method = 'hist'
            use_gpu = True
            print("✓ GPU detected and enabled (using 'hist' with device='cuda')")
        else:
            print("⚠️  GPU requested but not available. Falling back to CPU (tree_method='hist')")
            tree_method = 'hist'
            use_gpu = False
    elif tree_method == 'hist':
        if gpu_available:
            # Auto-enable GPU if available and using 'hist'
            use_gpu = True
            print("✓ GPU detected and enabled (using 'hist' with device='cuda')")
        else:
            print(f"Using CPU (tree_method='hist')")
            use_gpu = False
    
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print(f"Device: {device}")
    
    # Get feature list if not provided
    if features is None:
        print("\nDetecting features from training data...")
        features = _get_feature_list(train_df)
        embedding_count = len([c for c in features if c.startswith('embedding_')])
        tif_feature_count = len([c for c in features if c.startswith('tif_feature_')])
        other_count = len(features) - embedding_count - tif_feature_count
        print(f"  Found {len(features)} features:")
        print(f"    - {embedding_count} embedding features")
        if tif_feature_count > 0:
            print(f"    - {tif_feature_count} TIF features (from dataloader)")
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
        train_df, val_df, features, 'ignition', 'classifier', tree_method, device, use_gpu, random_seed
    )
    models['p_model'] = p_model
    
    # Train A-model (log_burned_area regressor)
    print("\n" + "="*70)
    print("A-MODEL: Conditional Log Burned Area (Regression)")
    print("="*70)
    a_model = _train_single_model(
        train_df, val_df, features, 'log_burned_area', 'regressor', tree_method, device, use_gpu, random_seed
    )
    models['a_model'] = a_model
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Both models saved")
    print("="*70)
    
    return models


def _train_single_model(train_df, val_df, features, target_col, model_type, tree_method, device, use_gpu, random_seed=42):
    """
    Train a single XGBoost model.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        features: List of feature column names
        target_col: Name of target column
        model_type: 'regressor' or 'classifier'
        tree_method: 'hist' (CPU) or 'gpu_hist' (GPU)
        device: PyTorch device object (for info only)
        use_gpu: Boolean indicating whether to use GPU
        random_seed: Random seed for reproducibility (default: 42)
    
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
    # Note: XGBoost doesn't use .to(device) like PyTorch
    # Data stays as pandas/numpy, XGBoost handles GPU internally via tree_method and device params
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
    early_stopping_rounds = 20
    
    # L2 regularization strength
    reg_lambda = 1.0
    
    # Configure device for GPU training (standard XGBoost GPU setup)
    # In newer XGBoost versions, use 'hist' with device='cuda' for GPU support
    device_params = {}
    if use_gpu:
        # XGBoost GPU configuration: use 'cuda' device
        # This works with 'hist' tree_method in newer XGBoost versions
        device_params['device'] = 'cuda'
    
    if model_type == 'classifier':
        model = XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=reg_lambda,  # L2 regularization
            random_state=random_seed,
            tree_method=tree_method,
            **device_params,  # Add GPU device parameters if using GPU
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
            reg_lambda=reg_lambda,  # L2 regularization
            random_state=random_seed,
            tree_method=tree_method,
            **device_params,  # Add GPU device parameters if using GPU
            eval_metric='rmse',  # Explicitly use RMSE (loss) for early stopping
            early_stopping_rounds=early_stopping_rounds
        )
    
    print(f"\nTraining {model_type}...")
    if use_gpu:
        print(f"  Device: GPU (CUDA)")
    else:
        print(f"  Device: CPU")
    print(f"  Tree method: {tree_method}")
    print(f"  L2 regularization (reg_lambda): {reg_lambda}")
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
            accuracy = accuracy_score(y_val, preds)
            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)
            print(f"ROC-AUC:     {auc:.4f}")
            print(f"PR-AUC:      {pr_auc:.4f}")
            print(f"Brier Score: {brier:.4f}")
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1-Score:    {f1:.4f}")
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
    
    # Save feature list directly from the trained model (ensures exact order)
    feature_list_path = MODELS_DIR / f"features_used_{model_type}_{target_col}.txt"
    
    # Extract feature names from the trained model to ensure exact order
    if hasattr(model, 'feature_names_in_'):
        # XGBoost 1.6+ stores feature names here
        model_features = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        # Older XGBoost versions
        try:
            model_features = model.get_booster().feature_names
        except:
            # Fallback to original features list
            model_features = features
    else:
        # Fallback to original features list
        model_features = features
    
    with open(feature_list_path, 'w') as f:
        for feature in model_features:
            f.write(f"{feature}\n")
    print(f"Feature list saved to {feature_list_path} ({len(model_features)} features)")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train both P-model (ignition probability) and A-model (conditional log burned area)'
    )
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of feature column names (if not provided, auto-detect from training data)')
    parser.add_argument('--tree-method', choices=['hist', 'gpu_hist'], default='gpu_hist',
                        help='Tree method: hist (CPU/GPU auto-detect) or gpu_hist (force GPU). Default: gpu_hist (auto-detects GPU)')
    
    args = parser.parse_args()
    
    try:
        train_xgb(
            features=args.features,
            tree_method=args.tree_method
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
