"""
Extract features from WildfireSpreadTS dataloader for XGBoost training.

This module implements Option 2 from the integration guide:
1. Extract features from the dataloader
2. Convert tensors to tabular format (DataFrame)
3. Exclude label bands (band 22) from features
4. Extract labels separately (ignition, burned_area)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the dataloader
try:
    from src.dataloader.FireSpreadDataModule import FireSpreadDataModule
    from src.dataloader.FireSpreadDataset import FireSpreadDataset
except ImportError as e:
    print(f"Error importing dataloader: {e}")
    print("Make sure pytorch-lightning and einops are installed:")
    print("  pip install pytorch-lightning einops")
    sys.exit(1)

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
INTERIM_DIR = PROJECT_ROOT / "data/interim"
RAW_DIR = PROJECT_ROOT / "data/raw/wildfirespreadts"


def extract_features_from_dataloader(
    data_module: FireSpreadDataModule,
    split: str = 'train',
    embeddings_df: Optional[pd.DataFrame] = None,
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Extract tabular features from WildfireSpreadTS dataloader.
    
    This function:
    1. Extracts features from the dataloader (excluding label band 22)
    2. Converts tensors to tabular format (DataFrame)
    3. Extracts labels separately (ignition, burned_area)
    4. Optionally merges with CNN embeddings
    
    Args:
        data_module: Initialized FireSpreadDataModule
        split: Which split to extract ('train', 'val', 'test')
        embeddings_df: Optional DataFrame with CNN embeddings to merge
        aggregation: How to aggregate spatial dimensions ('mean', 'max', 'sum')
    
    Returns:
        DataFrame with features and labels
    """
    # Setup dataloader
    data_module.setup(stage='fit')
    
    # Get the appropriate dataloader
    if split == 'train':
        loader = data_module.train_dataloader()
    elif split == 'val':
        loader = data_module.val_dataloader()
    elif split == 'test':
        loader = data_module.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
    
    print(f"\nExtracting features from {split} split...")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Number of batches: {len(loader)}")
    
    records = []
    sample_idx = 0
    
    # Process batches
    for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"  Processing {split} batches")):
        # Handle different return types (with/without doy)
        # The dataloader returns tuples, so we need to handle them properly
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 2:
                x, y = batch_data
                doys = None
            elif len(batch_data) == 3:
                x, y, doys = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        else:
            # Single tensor (shouldn't happen, but handle gracefully)
            raise ValueError(f"Unexpected batch type: {type(batch_data)}")
        
        # x: (batch, time_steps, features, height, width) or (batch, features, height, width) if single time step
        # y: (batch, height, width) - binary fire mask (label)
        batch_size = x.shape[0]
        
        # Handle case where n_leading_observations=1 (no time dimension)
        if len(x.shape) == 4:
            # (batch, features, height, width) - single time step
            x = x.unsqueeze(1)  # Add time dimension: (batch, 1, features, height, width)
        
        for i in range(batch_size):
            # Step 1: Extract features (EXCLUDE last band = band 22 = fire detection label)
            # x[i] shape: (time_steps, features, height, width)
            # We want to exclude the last feature (band 22 = fire detection label)
            # Note: The dataloader may have already processed features, so we need to check
            # Based on the code, band 22 is the last feature before one-hot encoding
            # After preprocessing, the last feature is the binary fire mask
            features = x[i, :, :-1, :, :]  # (time_steps, features-1, height, width)
            
            # Step 2: Aggregate spatial dimensions
            if aggregation == 'mean':
                features_agg = features.mean(dim=(2, 3))  # (time_steps, features-1)
            elif aggregation == 'max':
                features_agg = features.max(dim=(2, 3))[0]  # (time_steps, features-1)
            elif aggregation == 'sum':
                features_agg = features.sum(dim=(2, 3))  # (time_steps, features-1)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            # Flatten temporal dimension to get tabular features
            features_flat = features_agg.flatten().numpy()  # (time_steps * (features-1),)
            
            # Step 3: Extract labels separately
            # y[i] shape: (height, width) - binary fire mask
            y_mask = y[i].numpy()
            
            # Ignition: any fire pixel = ignition
            ignition = int(np.any(y_mask > 0))
            
            # Burned area: sum of fire pixels (or mean * area, depending on interpretation)
            # For now, use sum as a proxy for burned area
            # Note: This is pixel count, not actual area. You may need to scale by pixel size.
            burned_area = float(np.sum(y_mask > 0))
            
            # Create record
            record = {
                'sample_idx': sample_idx,
                'split': split,
                'ignition': ignition,
                'burned_area': burned_area,
            }
            
            # Add feature columns
            for j, val in enumerate(features_flat):
                record[f'tif_feature_{j}'] = float(val)
            
            # Add day of year if available
            if doys is not None:
                # doys shape: (batch, time_steps)
                record['day_of_year'] = int(doys[i, -1].item()) if len(doys.shape) > 1 else int(doys[i].item())
            
            records.append(record)
            sample_idx += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    print(f"  ✓ Extracted {len(df):,} samples")
    print(f"  ✓ Features per sample: {len([c for c in df.columns if c.startswith('tif_feature_')])}")
    print(f"  ✓ Ignition rate: {df['ignition'].mean():.2%}")
    
    # Step 4: Merge with embeddings if provided
    if embeddings_df is not None:
        print(f"\n  Merging with CNN embeddings...")
        # Match by sample_idx and split (ensures perfect 1:1 matching)
        if 'sample_idx' in embeddings_df.columns and 'split' in embeddings_df.columns:
            # Merge on sample_idx and split
            embedding_cols = [c for c in embeddings_df.columns if c.startswith('embedding_')]
            df_merged = df.merge(
                embeddings_df[['sample_idx', 'split'] + embedding_cols],
                on=['sample_idx', 'split'],
                how='left'
            )
            # Check how many embeddings were matched
            n_matched = df_merged[embedding_cols[0]].notna().sum() if embedding_cols else 0
            if n_matched == len(df):
                print(f"  ✓ Matched all {n_matched} samples with embeddings")
                print(f"  ✓ Added {len(embedding_cols)} embedding features")
                df = df_merged
            else:
                print(f"  ⚠ Warning: Only matched {n_matched}/{len(df)} samples with embeddings")
                print(f"  Missing embeddings will be NaN. Check that embeddings were extracted with the same dataloader config.")
                df = df_merged
        elif len(df) == len(embeddings_df):
            # Fallback: sequential matching (for backward compatibility)
            embedding_cols = [c for c in embeddings_df.columns if c.startswith('embedding_')]
            for col in embedding_cols:
                df[col] = embeddings_df[col].values
            print(f"  ✓ Added {len(embedding_cols)} embedding features (sequential match)")
        else:
            print(f"  ⚠ Warning: Embeddings length ({len(embeddings_df)}) doesn't match dataloader length ({len(df)})")
            print(f"  Skipping embedding merge. Re-extract embeddings using:")
            print(f"    python src/data/extract_cnn_embeddings.py")
    
    return df


def extract_all_splits(
    data_dir: str,
    embeddings_path: Optional[str] = None,
    n_leading_observations: int = 1,
    crop_side_length: int = 256,
    load_from_hdf5: bool = False,
    batch_size: int = 1,
    num_workers: int = 4,
    data_fold_id: int = 0,
    aggregation: str = 'mean',
    sample_size: Optional[int] = None  # Deprecated: kept for backward compatibility, but not used
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract features from all splits (train, val, test) using the dataloader.
    
    Args:
        data_dir: Path to WildfireSpreadTS dataset directory
        embeddings_path: Optional path to embeddings parquet file
        n_leading_observations: Number of days to use as input
        crop_side_length: Side length for cropping (not used in test)
        load_from_hdf5: Whether to load from HDF5 files
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        data_fold_id: Which data fold to use (0 = train:2018-2019, val:2020, test:2021)
        aggregation: How to aggregate spatial dimensions
        sample_size: Deprecated - kept for backward compatibility but not used
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("="*70)
    print("Extracting Features from WildfireSpreadTS Dataloader")
    print("="*70)
    
    # Load embeddings if provided
    embeddings_df = None
    if embeddings_path:
        embeddings_path = Path(embeddings_path)
        if embeddings_path.exists():
            print(f"\nLoading embeddings from {embeddings_path}...")
            embeddings_df = pd.read_parquet(embeddings_path)
            print(f"  ✓ Loaded {len(embeddings_df):,} embeddings")
        else:
            print(f"\n⚠ Warning: Embeddings file not found: {embeddings_path}")
    
    # Initialize datamodule
    print(f"\nInitializing dataloader...")
    print(f"  Data directory: {data_dir}")
    print(f"  Data fold ID: {data_fold_id} (train/val/test split)")
    print(f"  Leading observations: {n_leading_observations}")
    print(f"  Load from HDF5: {load_from_hdf5}")
    
    data_module = FireSpreadDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        n_leading_observations=n_leading_observations,
        n_leading_observations_test_adjustment=n_leading_observations,
        crop_side_length=crop_side_length,
        load_from_hdf5=load_from_hdf5,
        num_workers=num_workers,
        remove_duplicate_features=False,
        features_to_keep=None,  # Use all features (except we exclude band 22 in extraction)
        return_doy=False,
        data_fold_id=data_fold_id
    )
    
    # Extract each split
    train_df = extract_features_from_dataloader(
        data_module, split='train', embeddings_df=embeddings_df, aggregation=aggregation
    )
    
    val_df = extract_features_from_dataloader(
        data_module, split='val', embeddings_df=embeddings_df, aggregation=aggregation
    )
    
    test_df = extract_features_from_dataloader(
        data_module, split='test', embeddings_df=embeddings_df, aggregation=aggregation
    )
    
    # Note: sample_size limit removed - process all samples
    
    # Add target variables
    print(f"\nCreating target variables...")
    for df in [train_df, val_df, test_df]:
        df['log_burned_area'] = np.log1p(df['burned_area'])
    
    # Summary statistics
    print(f"\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} Split:")
        print(f"  Samples: {len(df):,}")
        print(f"  Ignition rate: {df['ignition'].mean():.2%}")
        print(f"  Mean burned area: {df['burned_area'].mean():.2f}")
        print(f"  Features: {len([c for c in df.columns if c.startswith('tif_feature_') or c.startswith('embedding_')])}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract features from WildfireSpreadTS dataloader for XGBoost'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(RAW_DIR),
        help='Path to WildfireSpreadTS dataset directory'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings parquet file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROCESSED_DIR),
        help='Output directory for split CSVs'
    )
    parser.add_argument(
        '--n-leading-observations',
        type=int,
        default=1,
        help='Number of days to use as input (default: 1)'
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
        help='Data fold ID for train/val/test split (default: 0)'
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
        # Set default embeddings path if not provided
        if args.embeddings is None:
            embeddings_path = INTERIM_DIR / "wildfirespreadts_embeddings.parquet"
            if embeddings_path.exists():
                args.embeddings = str(embeddings_path)
        
        # Extract all splits
        train_df, val_df, test_df = extract_all_splits(
            data_dir=args.data_dir,
            embeddings_path=args.embeddings,
            n_leading_observations=args.n_leading_observations,
            crop_side_length=256,
            load_from_hdf5=args.load_from_hdf5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_fold_id=args.data_fold_id,
            aggregation=args.aggregation,
            sample_size=None  # Removed: process all samples
        )
        
        # Save splits
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n" + "="*70)
        print("✓ Feature extraction complete!")
        print("="*70)
        print(f"\nSaved splits:")
        print(f"  Train: {train_path} ({len(train_df):,} samples)")
        print(f"  Val:   {val_path} ({len(val_df):,} samples)")
        print(f"  Test:  {test_path} ({len(test_df):,} samples)")
        print(f"\nNext steps:")
        print(f"  1. Train P-model: python src/models/train_model.py")
        print(f"  2. Train A-model: python src/models/train_model.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

