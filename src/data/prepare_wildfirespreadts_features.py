"""
Prepare WildfireSpreadTS features for XGBoost training.

This script uses the official WildfireSpreadTS dataloader (PyTorch Lightning)
which provides proper band labeling and prevents data leakage.

Workflow:
1. Uses PyTorch Lightning dataloader to extract features
2. Properly excludes label bands (band 22) from features (no data leakage)
3. Combines all data into a single features.csv
4. Next step: Run split_data.py to create train/val/test splits

This follows the original workflow but uses the dataloader for feature extraction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
INTERIM_DIR = PROJECT_ROOT / "data/interim"
RAW_DIR = PROJECT_ROOT / "data/raw/wildfirespreadts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare WildfireSpreadTS features for XGBoost training using the official dataloader'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to WildfireSpreadTS dataset directory (default: data/raw/wildfirespreadts)'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings parquet file (default: checks data/interim/wildfirespreadts_embeddings.parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for features.csv (default: data/processed/features.csv)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for dataloader (default: 1 for feature extraction)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of workers for dataloader (default: 4)'
    )
    parser.add_argument(
        '--data-fold-id',
        type=int,
        default=0,
        help='Data fold ID for dataloader internal splits (default: 0)'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max', 'sum'],
        help='How to aggregate spatial dimensions (default: mean)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Limit number of samples per split (for testing)'
    )
    parser.add_argument(
        '--load-from-hdf5',
        action='store_true',
        help='Load from HDF5 files instead of TIF (requires preprocessing)'
    )
    
    args = parser.parse_args()
    
    try:
        # Use dataloader approach - follows original workflow
        from src.data.extract_features_from_dataloader import extract_all_splits
        
        # Set default paths
        if args.data_dir is None:
            args.data_dir = str(RAW_DIR)
        if args.embeddings is None:
            embeddings_path = INTERIM_DIR / "wildfirespreadts_embeddings.parquet"
            if embeddings_path.exists():
                args.embeddings = str(embeddings_path)
            else:
                args.embeddings = None
        
        # Extract features from all splits using dataloader
        print("="*70)
        print("Preparing WildfireSpreadTS Features (Using Official Dataloader)")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Embeddings: {args.embeddings or 'None (skipping)'}")
        print(f"  Data fold ID: {args.data_fold_id} (for dataloader internal splits)")
        print(f"  Aggregation: {args.aggregation}")
        print(f"  Load from HDF5: {args.load_from_hdf5}")
        
        train_df, val_df, test_df = extract_all_splits(
            data_dir=args.data_dir,
            embeddings_path=args.embeddings,
            n_leading_observations=1,
            crop_side_length=256,
            load_from_hdf5=args.load_from_hdf5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_fold_id=args.data_fold_id,
            aggregation=args.aggregation,
            sample_size=None  # Removed: process all samples
        )
        
        # Combine all splits into single features.csv (original workflow)
        print(f"\nCombining all splits into features.csv (original workflow)...")
        
        # Add split indicator
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine all splits
        df_features = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Add temporal features if day_of_year is available
        if 'day_of_year' in df_features.columns:
            print(f"  Adding temporal features from day_of_year...")
            df_features['month'] = ((df_features['day_of_year'] - 1) // 30) + 1
            df_features['month'] = df_features['month'].clip(1, 12)
            df_features['day_of_month'] = ((df_features['day_of_year'] - 1) % 30) + 1
        
        # Set output path
        if args.output is None:
            output_path = PROCESSED_DIR / "features.csv"
        else:
            output_path = Path(args.output)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save combined features.csv
        df_features.to_csv(output_path, index=False)
        
        print(f"\n" + "="*70)
        print("✓ Feature preparation complete!")
        print("="*70)
        print(f"\nSaved features: {output_path}")
        print(f"  Total samples: {len(df_features):,}")
        print(f"  Total features: {len([c for c in df_features.columns if c.startswith('tif_feature_') or c.startswith('embedding_')])}")
        print(f"\nNext steps:")
        print(f"  1. Split data: python src/data/split_data.py")
        print(f"  2. Train P-model: python src/models/train_model.py")
        print(f"  3. Train A-model: python src/models/train_model.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
