"""
Split data into train/val/test sets using random split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def split_data(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data into train/validation/test sets using random split.
    
    Args:
        input_file: Path to input CSV file
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_df, val_df, test_df
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total samples: {len(df):,}")
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Random split
    print(f"\nUsing random split")
    print("="*70)
    print(f"  Train ratio: {train_ratio:.1%}")
    print(f"  Val ratio:   {val_ratio:.1%}")
    print(f"  Test ratio:  {test_ratio:.1%}")
    print(f"  Random seed: {random_seed}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test = n_total - n_train - n_val  # Remaining goes to test
    
    # Split data
    train_df = df_shuffled[:n_train].copy()
    val_df = df_shuffled[n_train:n_train + n_val].copy()
    test_df = df_shuffled[n_train + n_val:].copy()
    
    # Show year distribution if available
    if 'year' in df.columns:
        print(f"\nYear distribution (random split):")
        train_years = sorted(train_df['year'].unique())
        val_years = sorted(val_df['year'].unique())
        test_years = sorted(test_df['year'].unique())
        print(f"  Train years: {train_years} ({len(train_years)} unique years)")
        print(f"  Val years:   {val_years} ({len(val_years)} unique years)")
        print(f"  Test years:  {test_years} ({len(test_years)} unique years)")
    
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
    
    parser = argparse.ArgumentParser(
        description='Split data into train/val/test sets using random split.'
    )
    parser.add_argument('--input-file', default='data/processed/features.csv',
                        help='Input CSV file')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Proportion of data for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Proportion of data for testing (default: 0.15)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        split_data(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
