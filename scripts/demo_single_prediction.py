"""
Demo script for making a single prediction with the wildfire risk model.

This script demonstrates the full pipeline:
1. Load a single sample from test data
2. Extract features
3. Run P-model (ignition probability)
4. Run A-model (conditional burned area)
5. Combine into hazard score
6. Display results

Usage:
    python scripts/demo_single_prediction.py [sample_index]
    
    If sample_index is not provided, a random sample will be selected.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import random

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data/processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models/final"
RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/wildfirespreadts"


def load_feature_list(model_type, target_col):
    """Load feature list from saved file."""
    feature_file = MODELS_DIR / f"features_used_{model_type}_{target_col}.txt"
    if not feature_file.exists():
        return None
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def load_sample(data_path, sample_idx=None):
    """Load a single sample from the dataset."""
    print("=" * 70)
    print("Loading Sample Data")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(df)}")
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(df) - 1)
        print(f"Randomly selected sample index: {sample_idx}")
    else:
        sample_idx = int(sample_idx)
        if sample_idx < 0 or sample_idx >= len(df):
            print(f"Error: Sample index {sample_idx} out of range [0, {len(df)-1}]")
            sys.exit(1)
    
    sample = df.iloc[sample_idx].copy()
    
    print(f"\nSample Information:")
    print(f"  Index: {sample_idx}")
    if 'filename' in sample:
        print(f"  Filename: {sample['filename']}")
    if 'date' in sample:
        print(f"  Date: {sample['date']}")
    if 'event_id' in sample:
        print(f"  Event ID: {sample['event_id']}")
    
    # Show actual values if available
    if 'ignition' in sample:
        ignition_status = "YES" if sample['ignition'] == 1 else "NO"
        print(f"  Actual Ignition: {ignition_status}")
    if 'burned_area' in sample:
        print(f"  Actual Burned Area: {sample['burned_area']:.2f} hectares")
    
    return sample, sample_idx


def make_prediction(sample, verbose=True):
    """Make prediction for a single sample."""
    print("\n" + "=" * 70)
    print("Making Predictions")
    print("=" * 70)
    
    # Load models
    if verbose:
        print("\n1. Loading trained models...")
    
    p_model_path = MODELS_DIR / "xgb_wildfire_classifier_ignition.pkl"
    a_model_path = MODELS_DIR / "xgb_wildfire_regressor_log_burned_area.pkl"
    
    if not p_model_path.exists():
        raise FileNotFoundError(f"P-model not found: {p_model_path}")
    if not a_model_path.exists():
        raise FileNotFoundError(f"A-model not found: {a_model_path}")
    
    p_model = joblib.load(p_model_path)
    a_model = joblib.load(a_model_path)
    
    if verbose:
        print("   ✓ P-model (ignition classifier) loaded")
        print("   ✓ A-model (burned area regressor) loaded")
    
    # Load feature lists
    if verbose:
        print("\n2. Loading feature lists...")
    
    p_features = load_feature_list('classifier', 'ignition')
    a_features = load_feature_list('regressor', 'log_burned_area')
    
    if p_features is None or a_features is None:
        raise ValueError("Feature lists not found. Please ensure models were trained.")
    
    if verbose:
        print(f"   ✓ P-model expects {len(p_features)} features")
        print(f"   ✓ A-model expects {len(a_features)} features")
    
    # Extract features
    if verbose:
        print("\n3. Extracting features from sample...")
    
    # Convert sample to DataFrame for easier feature extraction
    sample_df = pd.DataFrame([sample])
    
    # Get features for P-model
    X_p = sample_df[p_features].copy()
    
    # Get features for A-model
    X_a = sample_df[a_features].copy()
    
    # Handle missing values
    if X_p.isnull().any().any():
        if verbose:
            print("   ⚠ Missing values detected, filling with median...")
        # For single sample, use 0 or mean from training (simplified)
        X_p = X_p.fillna(0)
    
    if X_a.isnull().any().any():
        if verbose:
            print("   ⚠ Missing values detected, filling with median...")
        X_a = X_a.fillna(0)
    
    if verbose:
        print("   ✓ Features extracted and prepared")
    
    # Make predictions
    if verbose:
        print("\n4. Running P-model (ignition probability)...")
    
    p_ignition = p_model.predict_proba(X_p)[0, 1]  # Probability of ignition
    
    if verbose:
        print(f"   → Ignition Probability: {p_ignition:.4f} ({p_ignition*100:.2f}%)")
        if p_ignition > 0.5:
            print(f"   → Prediction: FIRE LIKELY (probability > 50%)")
        else:
            print(f"   → Prediction: Fire unlikely (probability ≤ 50%)")
    
    if verbose:
        print("\n5. Running A-model (conditional burned area)...")
    
    log_burned_area = a_model.predict(X_a)[0]
    burned_area = np.exp(log_burned_area)
    
    if verbose:
        print(f"   → Log(Burned Area): {log_burned_area:.4f}")
        print(f"   → Expected Burned Area (if ignition): {burned_area:.2f} hectares")
    
    # Combine predictions
    if verbose:
        print("\n6. Computing hazard score...")
    
    hazard_score = p_ignition * burned_area
    
    if verbose:
        print(f"   → Hazard Score = P(ignition) × E[burned_area | ignition]")
        print(f"   → Hazard Score = {p_ignition:.4f} × {burned_area:.2f}")
        print(f"   → Final Hazard Score: {hazard_score:.2f} hectares")
    
    # Create results dictionary
    results = {
        'p_ignition': p_ignition,
        'log_burned_area': log_burned_area,
        'burned_area': burned_area,
        'hazard_score': hazard_score
    }
    
    # Add actual values if available
    if 'ignition' in sample:
        results['actual_ignition'] = sample['ignition']
    if 'burned_area' in sample:
        results['actual_burned_area'] = sample['burned_area']
    
    return results


def display_results(sample, results, sample_idx):
    """Display prediction results in a formatted way."""
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    print(f"\nSample #{sample_idx}")
    if 'filename' in sample:
        print(f"File: {sample['filename']}")
    if 'date' in sample:
        print(f"Date: {sample['date']}")
    
    print("\n" + "-" * 70)
    print("MODEL PREDICTIONS:")
    print("-" * 70)
    print(f"  P-Model (Ignition Probability): {results['p_ignition']:.4f} ({results['p_ignition']*100:.2f}%)")
    print(f"  A-Model (Expected Burned Area): {results['burned_area']:.2f} hectares")
    print(f"  Combined Hazard Score: {results['hazard_score']:.2f} hectares")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    
    if results['p_ignition'] > 0.7:
        risk_level = "HIGH RISK"
        risk_color = "🔴"
    elif results['p_ignition'] > 0.3:
        risk_level = "MODERATE RISK"
        risk_color = "🟡"
    else:
        risk_level = "LOW RISK"
        risk_color = "🟢"
    
    print(f"  Risk Level: {risk_color} {risk_level}")
    print(f"  Expected Impact: {results['hazard_score']:.2f} hectares if fire occurs")
    
    if results['hazard_score'] > 1000:
        print(f"  → This represents a MAJOR wildfire threat")
    elif results['hazard_score'] > 100:
        print(f"  → This represents a SIGNIFICANT wildfire threat")
    else:
        print(f"  → This represents a MODERATE wildfire threat")
    
    # Compare with actual if available
    if 'actual_ignition' in results and 'actual_burned_area' in results:
        print("\n" + "-" * 70)
        print("ACTUAL VALUES (for comparison):")
        print("-" * 70)
        actual_ignition = "YES" if results['actual_ignition'] == 1 else "NO"
        print(f"  Actual Ignition: {actual_ignition}")
        print(f"  Actual Burned Area: {results['actual_burned_area']:.2f} hectares")
        
        # Check prediction accuracy
        predicted_ignition = "YES" if results['p_ignition'] > 0.5 else "NO"
        ignition_correct = (predicted_ignition == actual_ignition)
        print(f"  Ignition Prediction: {'✓ CORRECT' if ignition_correct else '✗ INCORRECT'}")
        
        if results['actual_burned_area'] > 0:
            area_error = abs(results['burned_area'] - results['actual_burned_area'])
            area_error_pct = (area_error / results['actual_burned_area']) * 100
            print(f"  Burned Area Error: {area_error:.2f} hectares ({area_error_pct:.1f}%)")
    
    print("\n" + "=" * 70)


def main():
    """Main demo function."""
    # Parse arguments
    sample_idx = None
    if len(sys.argv) > 1:
        try:
            sample_idx = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid sample index: {sys.argv[1]}")
            print("Usage: python scripts/demo_single_prediction.py [sample_index]")
            sys.exit(1)
    
    # Load test data
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Please run the data preparation pipeline first:")
        print("  1. python src/data/extract_wildfirespreadts_embeddings.py")
        print("  2. python src/data/prepare_wildfirespreadts_features.py")
        print("  3. python src/data/split_data.py")
        sys.exit(1)
    
    # Load sample
    sample, sample_idx = load_sample(test_path, sample_idx)
    
    # Make prediction
    results = make_prediction(sample, verbose=True)
    
    # Display results
    display_results(sample, results, sample_idx)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

