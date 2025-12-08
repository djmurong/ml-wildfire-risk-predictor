"""
Example script for making predictions with the trained wildfire risk model.
"""
import pandas as pd
import joblib
from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).resolve().parents[1] / "models/final"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data/processed"

def predict_example():
    """Load model and make example predictions."""
    
    # Check if model exists
    model_path = MODELS_DIR / "xgb_wildfire_model.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run training first: python src/models/train_model.py")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Create example data for prediction
    # Features: ['latitude', 'longitude', 'day_of_year', 'month', 'confidence', 'brightness_mean']
    examples = pd.DataFrame({
        'latitude': [37.7749, 34.0522, 40.7128],  # San Francisco, LA, NYC
        'longitude': [-122.4194, -118.2437, -74.0060],
        'day_of_year': [180, 200, 250],  # Mid-summer dates
        'month': [6, 7, 9],
        'confidence': [2, 1, 2],  # high, nominal, high
        'brightness_mean': [320.0, 310.0, 330.0]
    })
    
    print("\nExample input data:")
    print(examples)
    
    # Make predictions
    predictions = model.predict(examples)
    
    print("\nPredicted hazard scores:")
    for i, (_, row) in enumerate(examples.iterrows()):
        print(f"  Example {i+1} (lat={row['latitude']:.2f}, lon={row['longitude']:.2f}): {predictions[i]:.4f}")
    
    # If test data exists, show some real predictions
    test_path = PROCESSED_DIR / "test.csv"
    if test_path.exists():
        print("\n--- Making predictions on test set (first 5 samples) ---")
        test_df = pd.read_csv(test_path)
        features = ['latitude', 'longitude', 'day_of_year', 'month', 'confidence', 'brightness_mean']
        X_test = test_df[features].head(5)
        y_test = test_df['hazard_score'].head(5)
        
        test_predictions = model.predict(X_test)
        
        print("\nTest samples predictions:")
        for i in range(len(X_test)):
            print(f"  Sample {i+1}:")
            print(f"    Actual: {y_test.iloc[i]:.4f}")
            print(f"    Predicted: {test_predictions[i]:.4f}")
            print(f"    Error: {abs(y_test.iloc[i] - test_predictions[i]):.4f}")
    else:
        print("\nNote: Test data not found. Run the full pipeline to generate test data.")
    
    print("\nPrediction example completed!")

if __name__ == "__main__":
    predict_example()