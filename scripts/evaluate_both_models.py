"""
Helper script to evaluate both P-model and A-model in one run.
This script automatically loads the feature lists saved during training.
"""
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models/final"

def load_feature_list(model_type, target_col):
    """Load feature list from saved file."""
    feature_file = MODELS_DIR / f"features_used_{model_type}_{target_col}.txt"
    if not feature_file.exists():
        print(f"Error: Feature list not found at {feature_file}")
        print("Please run training first to generate feature lists.")
        return None
    
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features

def evaluate_both_models():
    """Evaluate both P-model and A-model."""
    
    # P-model (classifier)
    print("=" * 60)
    print("Evaluating P-model (Ignition Classifier)")
    print("=" * 60)
    
    p_features = load_feature_list('classifier', 'ignition')
    if p_features:
        model_path = MODELS_DIR / "xgb_wildfire_classifier_ignition.pkl"
        if model_path.exists():
            cmd = [
                sys.executable, "src/models/evaluate.py",
                "--model", str(model_path),
                "--features"] + p_features + [
                "--target", "ignition",
                "--model-type", "classifier"
            ]
            subprocess.run(cmd)
        else:
            print(f"Error: Model not found at {model_path}")
    else:
        print("Skipping P-model evaluation due to missing feature list.")
    
    print("\n")
    
    # A-model (regressor)
    print("=" * 60)
    print("Evaluating A-model (Log Burned Area Regressor)")
    print("=" * 60)
    
    a_features = load_feature_list('regressor', 'log_burned_area')
    if a_features:
        model_path = MODELS_DIR / "xgb_wildfire_regressor_log_burned_area.pkl"
        if model_path.exists():
            cmd = [
                sys.executable, "src/models/evaluate.py",
                "--model", str(model_path),
                "--features"] + a_features + [
                "--target", "log_burned_area",
                "--model-type", "regressor"
            ]
            subprocess.run(cmd)
        else:
            print(f"Error: Model not found at {model_path}")
    else:
        print("Skipping A-model evaluation due to missing feature list.")

if __name__ == "__main__":
    evaluate_both_models()

