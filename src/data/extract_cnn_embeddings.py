"""
Extract CNN embeddings from WildfireSpreadTS dataloader.

This script extracts embeddings using the same dataloader as feature extraction,
ensuring perfect 1:1 matching between embeddings and features.
"""

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the dataloader
try:
    from src.dataloader.FireSpreadDataModule import FireSpreadDataModule
except ImportError as e:
    print(f"Error importing dataloader: {e}")
    print("Make sure pytorch-lightning and einops are installed:")
    print("  pip install pytorch-lightning einops")
    sys.exit(1)

INTERIM_DIR = PROJECT_ROOT / "data/interim"
RAW_DIR = PROJECT_ROOT / "data/raw/wildfirespreadts"


def load_resnet50_embedding_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load ResNet50 and remove classification head to get embeddings."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    # Remove the classification head so output = 2048-dim embedding
    embedding_model = torch.nn.Sequential(*list(model.children())[:-1])
    return embedding_model.to(device)


def extract_embeddings_from_image_tensor(x, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extract CNN embeddings from a preprocessed image tensor.
    
    Args:
        x: Image tensor from dataloader, shape (time_steps, features, height, width)
           or (features, height, width) if single time step
        model: ResNet50 embedding model
        device: Device to run on
    
    Returns:
        numpy array of embeddings (2048-dim)
    """
    # Convert multi-channel image to RGB for ResNet
    # We'll use the first 3 channels (or take mean across channels if > 3)
    if len(x.shape) == 4:
        # (time_steps, features, height, width) - use last time step
        x_img = x[-1, :, :, :]  # (features, height, width)
    else:
        # (features, height, width)
        x_img = x
    
    # ResNet expects RGB (3 channels), but we have many feature channels
    # Strategy: take first 3 channels, or create RGB from key channels
    # For now, use first 3 channels and normalize to [0, 1] range
    if x_img.shape[0] >= 3:
        # Use first 3 channels (VIIRS bands)
        x_rgb = x_img[:3, :, :]  # (3, height, width)
    else:
        # Pad with zeros if less than 3 channels
        x_rgb = torch.zeros(3, x_img.shape[1], x_img.shape[2])
        x_rgb[:x_img.shape[0], :, :] = x_img
    
    # Normalize to [0, 1] range (ResNet expects this)
    x_rgb = x_rgb - x_rgb.min()
    if x_rgb.max() > 0:
        x_rgb = x_rgb / x_rgb.max()
    
    # Resize to 224x224 (ResNet input size)
    x_rgb = T.functional.resize(x_rgb.unsqueeze(0), (224, 224)).squeeze(0)
    
    # Normalize with ImageNet stats
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    x_rgb = normalize(x_rgb)
    
    # Add batch dimension and extract embedding
    x_batch = x_rgb.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x_batch)
        embedding = embedding.squeeze().cpu().numpy()
    
    return embedding


def extract_embeddings_from_dataloader(
    data_module: FireSpreadDataModule,
    split: str = 'train',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Extract CNN embeddings from WildfireSpreadTS dataloader.
    
    Args:
        data_module: Initialized FireSpreadDataModule
        split: Which split to extract ('train', 'val', 'test')
        device: Device to run on
    
    Returns:
        DataFrame with embeddings and sample_idx
    """
    # Load embedding model
    print(f"  Loading ResNet50 embedding model on {device}...")
    model = load_resnet50_embedding_model(device)
    
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
    
    print(f"\nExtracting embeddings from {split} split...")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Number of batches: {len(loader)}")
    
    records = []
    sample_idx = 0
    
    # Process batches
    for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"  Processing {split} batches")):
        # Handle different return types (with/without doy)
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 2:
                x, y = batch_data
            elif len(batch_data) == 3:
                x, y, doys = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        else:
            raise ValueError(f"Unexpected batch type: {type(batch_data)}")
        
        # x: (batch, time_steps, features, height, width) or (batch, features, height, width) if single time step
        batch_size = x.shape[0]
        
        # Handle case where n_leading_observations=1 (no time dimension)
        if len(x.shape) == 4:
            # (batch, features, height, width) - single time step
            x = x.unsqueeze(1)  # Add time dimension: (batch, 1, features, height, width)
        
        for i in range(batch_size):
            # Extract embedding from image
            x_sample = x[i]  # (time_steps, features, height, width)
            embedding = extract_embeddings_from_image_tensor(x_sample, model, device)
            
            # Create record
            record = {
                'sample_idx': sample_idx,
                'split': split,
            }
            
            # Add embedding dimensions
            for j, val in enumerate(embedding):
                record[f'embedding_{j}'] = float(val)
            
            records.append(record)
            sample_idx += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    print(f"  ✓ Extracted {len(df):,} embeddings")
    print(f"  ✓ Embedding dimension: {len([c for c in df.columns if c.startswith('embedding_')])}")
    
    return df


def extract_all_splits_embeddings(
    data_dir: str,
    n_leading_observations: int = 1,
    crop_side_length: int = 256,
    load_from_hdf5: bool = False,
    batch_size: int = 1,
    num_workers: int = 4,
    data_fold_id: int = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Extract embeddings from all splits (train, val, test) using the dataloader.
    
    Args:
        data_dir: Path to WildfireSpreadTS dataset directory
        n_leading_observations: Number of days to use as input
        crop_side_length: Side length for cropping
        load_from_hdf5: Whether to load from HDF5 files
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        data_fold_id: Which data fold to use
        device: Device to run on
    
    Returns:
        Combined DataFrame with embeddings from all splits
    """
    print("="*70)
    print("Extracting CNN Embeddings from WildfireSpreadTS Dataloader")
    print("="*70)
    
    # Initialize datamodule
    print(f"\nInitializing dataloader...")
    print(f"  Data directory: {data_dir}")
    print(f"  Data fold ID: {data_fold_id} (train/val/test split)")
    print(f"  Leading observations: {n_leading_observations}")
    print(f"  Load from HDF5: {load_from_hdf5}")
    print(f"  Device: {device}")
    
    data_module = FireSpreadDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        n_leading_observations=n_leading_observations,
        n_leading_observations_test_adjustment=n_leading_observations,
        crop_side_length=crop_side_length,
        load_from_hdf5=load_from_hdf5,
        num_workers=num_workers,
        remove_duplicate_features=False,
        features_to_keep=None,
        return_doy=False,
        data_fold_id=data_fold_id
    )
    
    # Extract each split
    train_df = extract_embeddings_from_dataloader(
        data_module, split='train', device=device
    )
    
    val_df = extract_embeddings_from_dataloader(
        data_module, split='val', device=device
    )
    
    test_df = extract_embeddings_from_dataloader(
        data_module, split='test', device=device
    )
    
    # Combine all splits
    df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Summary statistics
    print(f"\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} Split:")
        print(f"  Samples: {len(df):,}")
        print(f"  Embedding features: {len([c for c in df.columns if c.startswith('embedding_')])}")
    
    print(f"\nTotal embeddings: {len(df_combined):,}")
    
    return df_combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract CNN embeddings from WildfireSpreadTS dataloader'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(RAW_DIR),
        help='Path to WildfireSpreadTS dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(INTERIM_DIR / "wildfirespreadts_embeddings.parquet"),
        help='Output path for embeddings parquet file'
    )
    parser.add_argument(
        '--n-leading-observations',
        type=int,
        default=1,
        help='Number of days to use as input (default: 1)'
    )
    parser.add_argument(
        '--crop-side-length',
        type=int,
        default=256,
        help='Side length for cropping (default: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for dataloader (default: 1)'
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
        '--load-from-hdf5',
        action='store_true',
        help='Load from HDF5 files instead of TIF (requires preprocessing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract embeddings from all splits
        df_embeddings = extract_all_splits_embeddings(
            data_dir=args.data_dir,
            n_leading_observations=args.n_leading_observations,
            crop_side_length=args.crop_side_length,
            load_from_hdf5=args.load_from_hdf5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_fold_id=args.data_fold_id,
            device=args.device
        )
        
        # Save embeddings
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_embeddings.to_parquet(output_path, index=False)
        
        print(f"\n" + "="*70)
        print("✓ Embedding extraction complete!")
        print("="*70)
        print(f"\nSaved embeddings: {output_path}")
        print(f"  Total samples: {len(df_embeddings):,}")
        print(f"  Embedding dimension: {len([c for c in df_embeddings.columns if c.startswith('embedding_')])}")
        print(f"\nNext steps:")
        print(f"  1. Prepare features: python src/data/prepare_wildfirespreadts_features.py")
        print(f"     (Embeddings will be automatically merged)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

