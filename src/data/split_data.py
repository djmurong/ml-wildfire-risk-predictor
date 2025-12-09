import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def random_split(df, test_size=0.15, val_size=0.15):
    """
    Random split.
    
    Suitable for:
    - Short time periods (e.g., 2018-2021) where temporal split leaves insufficient training data
    - Next-day prediction tasks (operational forecasting) where temporal trends are less critical
    - When spatial/temporal structure is already captured in features
    
    Args:
        df: DataFrame to split
        test_size: Proportion for test set
        val_size: Proportion for validation set
    
    Returns:
        train_df, val_df, test_df
    """
    trainval_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(trainval_df, test_size=val_ratio, random_state=42)
    return train_df, val_df, test_df


def split_data(
    input_file,
    test_size=0.15,
    val_size=0.15
):
    """
    Split data into train/validation/test sets using random split.
    
    Args:
        input_file: Path to input CSV
        test_size: Proportion for test set (default: 0.15)
        val_size: Proportion for validation set (default: 0.15)
    
    Returns:
        train_df, val_df, test_df
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total samples: {len(df)}")
    
    # Perform random split
    print(f"\nSplitting data using random split")
    print("="*70)
    
    train_df, val_df, test_df = random_split(df, test_size, val_size)
    print("Random split")
    print("  Note: Suitable for short time periods (e.g., 2018-2021) or next-day prediction tasks")
    
    # Print split statistics
    print(f"\nSplit results:")
    print(f"  Train: {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
    
    # Verify splits are disjoint
    if len(train_df) + len(val_df) + len(test_df) != len(df):
        print("\n⚠ Warning: Split sizes don't sum to total (may have overlapping rows)")
    
    # Save splits
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Splits saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split data into train/val/test sets using random split')
    parser.add_argument('--input-file', default='data/processed/features.csv',
                        help='Input CSV file')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Proportion for test set (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Proportion for validation set (default: 0.15)')
    
    args = parser.parse_args()
    
    try:
        split_data(
            input_file=args.input_file,
            test_size=args.test_size,
            val_size=args.val_size
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
