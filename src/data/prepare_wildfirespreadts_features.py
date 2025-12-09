"""
Prepare WildfireSpreadTS features for XGBoost training.

This script:
1. Loads CNN embeddings from parquet file
2. Extracts labels from .tif files (active fire detections, burned area)
3. Matches labels to embeddings by filename
4. Combines embeddings + labels + other features
5. Creates target variables (ignition, log_burned_area)
6. Saves to features.csv for splitting and training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
import sys
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data/interim"
RAW_DIR = Path(__file__).resolve().parents[2] / "data/raw/wildfirespreadts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# Global variables for multiprocessing (set before processing)
_global_tif_dict = None
_global_fire_band = None
_global_area_band = None
_global_aggregation = None


def _process_single_row(args):
    """Process a single embedding row to extract labels and features (for multiprocessing)."""
    row_dict, tif_dict, fire_detection_band, burned_area_band, aggregation, extract_tif_features_flag = args
    emb_filename = row_dict.get('filename', '')
    
    # Try to find matching .tif file
    tif_path = None
    if emb_filename:
        emb_stem = Path(emb_filename).stem
        tif_path = tif_dict.get(emb_stem)
        
        if tif_path is None:
            # Try partial match
            for tif_stem, tif_file in tif_dict.items():
                if emb_stem in tif_stem or tif_stem in emb_stem:
                    tif_path = tif_file
                    break
    
    # Extract labels and features
    result = {'filename': emb_filename}
    
    if tif_path and tif_path.exists():
        # Extract labels (ignition, burned_area)
        labels = extract_labels_from_tif(
            tif_path,
            fire_detection_band=fire_detection_band,
            burned_area_band=burned_area_band,
            aggregation=aggregation
        )
        result.update(labels)
        result['matched'] = True
        
        # Extract additional features from .tif file bands (weather, fuel, topography, etc.)
        if extract_tif_features_flag:
            try:
                import rasterio
                with rasterio.open(tif_path) as src:
                    # Extract all bands as features (mean aggregation per band)
                    for band_idx in range(1, min(src.count + 1, 50)):  # Limit to 50 bands
                        band = src.read(band_idx)
                        band_clean = band[~np.isnan(band)]
                        if len(band_clean) > 0:
                            result[f'tif_band_{band_idx}'] = float(np.mean(band_clean))
            except Exception:
                pass  # Silently skip if can't read features
    else:
        # No match found - use default values
        result.update({
            'ignition': 0,
            'burned_area': 0.0,
            'fire_detection_band': None,
            'burned_area_band': None,
            'matched': False
        })
    
    return result


def extract_labels_from_tif(
    tif_path,
    fire_detection_band=None,
    burned_area_band=None,
    aggregation='any'
):
    """
    Extract labels from a .tif file.
    
    Note: WildfireSpreadTS data is preprocessed with labels already filtered
    to medium/high confidence fire detections. We just need to extract them.
    
    Args:
        tif_path: Path to .tif file
        fire_detection_band: Band index (1-based) containing active fire detections
                           If None, will try to auto-detect (binary 0/1 band)
        burned_area_band: Band index (1-based) containing burned area
                         If None, will try to auto-detect (continuous values)
        aggregation: How to aggregate pixel values ('mean', 'max', 'sum', 'any')
                    'any' = 1 if any pixel has fire, 0 otherwise (recommended for binary)
    
    Returns:
        dict with 'ignition' (binary) and 'burned_area' (continuous) values
    """
    try:
        with rasterio.open(tif_path) as src:
            # Auto-detect label bands if not specified
            if fire_detection_band is None or burned_area_band is None:
                # Sample a few bands to identify label bands
                for band_idx in range(1, min(src.count + 1, 24)):
                    band = src.read(band_idx)
                    band_clean = band[~np.isnan(band)]
                    
                    if len(band_clean) == 0:
                        continue
                    
                    band_min = np.nanmin(band)
                    band_max = np.nanmax(band)
                    
                    # Check if binary (likely fire detection)
                    if fire_detection_band is None:
                        unique_vals = np.unique(band_clean)
                        # More flexible: check if mostly binary (0/1) or small integers
                        # Could be 0, 1, or 0, 1, 2 (confidence levels)
                        if len(unique_vals) <= 5:
                            # Check if values are mostly 0/1 or small integers
                            if all(v in [0, 1] for v in unique_vals):
                                fire_detection_band = band_idx
                            elif all(v in [0, 1, 2] for v in unique_vals) and len(unique_vals) <= 3:
                                # Could be confidence levels: 0=no fire, 1=low, 2=high
                                # Use this as fire detection, treat >0 as fire
                                fire_detection_band = band_idx
                            elif len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                                fire_detection_band = band_idx
                    
                    # Check if continuous moderate values (likely burned area)
                    if burned_area_band is None:
                        if 0 <= band_min and band_max > 0 and band_max < 10000:
                            if not (len(unique_vals) <= 3 and all(v in [0, 1] for v in unique_vals)):
                                burned_area_band = band_idx
            
            # Extract fire detection (binary)
            # Since data is preprocessed with medium/high confidence only,
            # presence of fire detection = ignition = 1
            ignition = 0
            if fire_detection_band is not None:
                fire_band = src.read(fire_detection_band)
                fire_clean = fire_band[~np.isnan(fire_band)]
                
                if len(fire_clean) > 0:
                    # For preprocessed data: any detection = ignition
                    # Use 'any' aggregation by default (most appropriate for binary labels)
                    if aggregation == 'any':
                        ignition = 1 if np.any(fire_clean > 0) else 0
                    elif aggregation == 'mean':
                        ignition = 1 if np.mean(fire_clean) > 0.5 else 0
                    elif aggregation == 'max':
                        ignition = 1 if np.max(fire_clean) > 0 else 0
                    else:
                        ignition = 1 if np.sum(fire_clean) > 0 else 0
            
            # Extract burned area (continuous)
            burned_area = 0.0
            if burned_area_band is not None:
                area_band = src.read(burned_area_band)
                area_clean = area_band[~np.isnan(area_band)]
                
                if len(area_clean) > 0:
                    if aggregation == 'mean':
                        burned_area = float(np.mean(area_clean))
                    elif aggregation == 'max':
                        burned_area = float(np.max(area_clean))
                    elif aggregation == 'sum':
                        burned_area = float(np.sum(area_clean))
                    else:
                        burned_area = float(np.mean(area_clean))
            
            # Fallback: if no fire detection band found, infer ignition from burned area
            # If burned_area > 0, there must have been ignition
            # This is a reasonable fallback since burned area implies fire occurred
            if fire_detection_band is None:
                if burned_area > 0:
                    ignition = 1  # If there's burned area, there was ignition
                # Otherwise ignition stays 0
            
            return {
                'ignition': ignition,
                'burned_area': burned_area,
                'fire_detection_band': fire_detection_band,
                'burned_area_band': burned_area_band
            }
    
    except Exception as e:
        print(f"Warning: Error reading {tif_path.name}: {e}")
        return {
            'ignition': 0,
            'burned_area': 0.0,
            'fire_detection_band': None,
            'burned_area_band': None
        }


def match_tif_to_embedding_filename(emb_filename, tif_files):
    """
    Match embedding filename to corresponding .tif file.
    
    Args:
        emb_filename: Filename from embeddings (may have different extension)
        tif_files: List of .tif file paths
    
    Returns:
        Path to matching .tif file or None
    """
    # Remove extension from embedding filename
    emb_stem = Path(emb_filename).stem
    
    # Try exact match
    for tif_file in tif_files:
        if Path(tif_file).stem == emb_stem:
            return tif_file
    
    # Try partial match (filename contains emb_stem or vice versa)
    for tif_file in tif_files:
        tif_stem = Path(tif_file).stem
        if emb_stem in tif_stem or tif_stem in emb_stem:
            return tif_file
    
    return None


def prepare_wildfirespreadts_features(
    embeddings_path=None,
    fire_detection_band=None,
    burned_area_band=None,
    aggregation='any',
    sample_size=None,
    output_path=None,
    num_workers=None,
    extract_tif_features_flag=True
):
    """
    Prepare features for XGBoost training.
    
    Args:
        embeddings_path: Path to embeddings parquet file
        fire_detection_band: Band index (1-based) for fire detections (None = auto-detect)
        burned_area_band: Band index (1-based) for burned area (None = auto-detect)
        aggregation: How to aggregate pixel values ('mean', 'max', 'sum', 'any')
        sample_size: Limit number of samples for testing (None = all)
        output_path: Path to save features.csv
    
    Returns:
        DataFrame with combined features
    """
    # Set default paths
    if embeddings_path is None:
        embeddings_path = INTERIM_DIR / "wildfirespreadts_embeddings.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / "features.csv"
    
    print("="*70)
    print("Preparing WildfireSpreadTS Features for Training")
    print("="*70)
    
    # 1. Load embeddings
    print(f"\n1. Loading embeddings from {embeddings_path}...")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    df_emb = pd.read_parquet(embeddings_path)
    print(f"   Loaded {len(df_emb):,} embeddings")
    print(f"   Columns: {list(df_emb.columns[:10])}...")
    
    # Limit for testing if specified
    if sample_size is not None:
        df_emb = df_emb.head(sample_size)
        print(f"   TEST MODE: Using only {len(df_emb):,} samples")
    
    # 2. Find all .tif files
    print(f"\n2. Finding .tif files in {RAW_DIR}...")
    tif_files = list(RAW_DIR.rglob("*.tif"))
    print(f"   Found {len(tif_files):,} .tif files")
    
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {RAW_DIR}")
    
    # 3. Extract labels from .tif files and match to embeddings
    print(f"\n3. Extracting labels from .tif files...")
    print(f"   Fire detection band: {fire_detection_band or 'auto-detect'}")
    print(f"   Burned area band: {burned_area_band or 'auto-detect'}")
    print(f"   Aggregation method: {aggregation}")
    
    labels_list = []
    matched_count = 0
    
    # Create a mapping from filename to tif path for faster lookup
    tif_dict = {}
    for tif_file in tif_files:
        tif_stem = Path(tif_file).stem
        tif_dict[tif_stem] = tif_file
    
    # Process in parallel or sequential
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    num_workers = max(1, min(num_workers, cpu_count()))  # Ensure valid range
    
    # Convert DataFrame rows to dictionaries for multiprocessing
    rows_list = [row.to_dict() for _, row in df_emb.iterrows()]
    
    if num_workers > 1:
        print(f"   Using {num_workers} CPU cores for parallel processing")
        
        # Prepare arguments for parallel processing
        process_args = [
            (row_dict, tif_dict, fire_detection_band, burned_area_band, aggregation, extract_tif_features_flag)
            for row_dict in rows_list
        ]
        
        # Process in parallel
        with Pool(processes=num_workers) as pool:
            labels_list = list(tqdm(
                pool.imap(_process_single_row, process_args),
                total=len(process_args),
                desc="   Processing"
            ))
    else:
        print(f"   Using sequential processing (1 core)")
        # Sequential processing (for debugging or single-core systems)
        labels_list = []
        for row_dict in tqdm(rows_list, desc="   Processing"):
            labels = _process_single_row((
                row_dict, tif_dict, fire_detection_band, burned_area_band, aggregation, extract_tif_features_flag
            ))
            labels_list.append(labels)
    
    # Count matches
    matched_count = sum(1 for labels in labels_list if labels.get('matched', False))
    
    print(f"   Matched {matched_count:,} / {len(df_emb):,} embeddings to .tif files")
    
    # 4. Combine embeddings, labels, and .tif features
    print(f"\n4. Combining embeddings, labels, and .tif features...")
    df_labels = pd.DataFrame(labels_list)
    
    # Merge on filename - include all columns from df_labels (labels + tif features)
    merge_cols = ['filename', 'ignition', 'burned_area', 'matched']
    tif_feature_cols = [c for c in df_labels.columns if c.startswith('tif_band_')]
    merge_cols.extend(tif_feature_cols)
    
    df_features = df_emb.merge(
        df_labels[merge_cols],
        on='filename',
        how='left'
    )
    
    if tif_feature_cols:
        print(f"   Added {len(tif_feature_cols)} features from .tif file bands")
    
    # Fill missing values
    df_features['ignition'] = df_features['ignition'].fillna(0).astype(int)
    df_features['burned_area'] = df_features['burned_area'].fillna(0.0).astype(float)
    
    # 5. Create target variables
    print(f"\n5. Creating target variables...")
    
    # P-model target: ignition (binary, already created)
    df_features['ignition'] = df_features['ignition'].astype(int)
    
    # A-model target: log(1 + burned_area) for conditional burned area
    df_features['log_burned_area'] = np.log1p(df_features['burned_area'])
    
    # 6. Add temporal features if date is available
    if 'date' in df_features.columns:
        print(f"   6. Adding temporal features...")
        df_features['date'] = pd.to_datetime(df_features['date'], errors='coerce')
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
    
    # 7. Data cleaning
    print(f"\n7. Cleaning data...")
    
    # Remove rows with all NaN embeddings
    embedding_cols = [c for c in df_features.columns if c.startswith('embedding_')]
    if embedding_cols:
        df_features = df_features.dropna(subset=embedding_cols[:10])  # Check first 10
    
    # Remove duplicates
    initial_len = len(df_features)
    df_features = df_features.drop_duplicates(subset=['filename'], keep='first')
    print(f"   Removed {initial_len - len(df_features):,} duplicate rows")
    
    # 8. Summary statistics
    print(f"\n8. Summary statistics:")
    print(f"   Total samples: {len(df_features):,}")
    print(f"   Ignition rate: {df_features['ignition'].mean():.2%}")
    print(f"   Samples with fire: {df_features['ignition'].sum():,}")
    print(f"   Samples without fire: {(df_features['ignition'] == 0).sum():,}")
    print(f"   Mean burned area: {df_features['burned_area'].mean():.4f}")
    print(f"   Max burned area: {df_features['burned_area'].max():.4f}")
    print(f"   Matched to .tif files: {matched_count:,} ({100*matched_count/len(df_features):.1f}%)")
    
    # 9. Save features
    print(f"\n9. Saving features to {output_path}...")
    df_features.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_features):,} rows, {len(df_features.columns)} columns")
    
    # Show column summary
    print(f"\n   Column summary:")
    print(f"   - Embedding columns: {len(embedding_cols)}")
    print(f"   - Target columns: ignition, log_burned_area")
    print(f"   - Metadata columns: {len(df_features.columns) - len(embedding_cols) - 2}")
    
    return df_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare WildfireSpreadTS features for XGBoost training'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings parquet file (default: checks interim/ then processed/)'
    )
    parser.add_argument(
        '--fire-band',
        type=int,
        default=None,
        help='Band index (1-based) for fire detections (None = auto-detect)'
    )
    parser.add_argument(
        '--area-band',
        type=int,
        default=None,
        help='Band index (1-based) for burned area (None = auto-detect)'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max', 'sum', 'any'],
        help='How to aggregate pixel values (default: mean)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Limit number of samples for testing (None = all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for features.csv (default: data/processed/features.csv)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--extract-tif-features',
        action='store_true',
        default=True,
        help='Extract additional features from .tif file bands (default: True)'
    )
    parser.add_argument(
        '--no-extract-tif-features',
        dest='extract_tif_features',
        action='store_false',
        help='Skip extracting features from .tif file bands (only extract labels)'
    )
    
    args = parser.parse_args()
    
    try:
        df_features = prepare_wildfirespreadts_features(
            embeddings_path=args.embeddings,
            fire_detection_band=args.fire_band,
            burned_area_band=args.area_band,
            aggregation=args.aggregation,
            sample_size=args.sample_size,
            extract_tif_features_flag=args.extract_tif_features,
            output_path=args.output,
            num_workers=args.num_workers
        )
        
        print("\n" + "="*70)
        print("✓ Feature preparation complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Split data: python src/data/split_data.py")
        print("  2. Train P-model: python src/models/train_model.py --model-type classifier --target-col ignition")
        print("  3. Train A-model: python src/models/train_model.py --model-type regressor --target-col log_burned_area")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

