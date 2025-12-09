"""
extract_wildfirespreadts_embeddings.py

Extract CNN embeddings from WildfireSpreadTS dataset imagery
and save them into a Parquet file for downstream ML models (XGBoost).

This script extracts embeddings from the imagery already included in the
WildfireSpreadTS dataset, eliminating the need for separate MODIS downloads.

WildfireSpreadTS contains:
- 13,607 images across 607 fire events (Jan 2018 - Oct 2021)
- Daily observations with active fire detections
- 375m spatial resolution
- Multi-temporal, multi-modal format (likely NetCDF, HDF5, or GeoTIFF)
"""

import os
import glob
import torch
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import geospatial libraries for reading various formats
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import pyhdf.SD as SD
    HAS_PYHDF = True
except ImportError:
    HAS_PYHDF = False


# ------------------------------------------------------------
# 1. Load pretrained CNN (ResNet50)
# ------------------------------------------------------------
def load_resnet50_embedding_model():
    """Load ResNet50 and remove classification head to get embeddings."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    # Remove the classification head so output = 2048-dim embedding
    embedding_model = torch.nn.Sequential(*list(model.children())[:-1])
    return embedding_model


# ------------------------------------------------------------
# 2. Image preprocessing for ResNet
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])


# ------------------------------------------------------------
# 3. Read image from various formats (NetCDF, HDF5, GeoTIFF, etc.)
# ------------------------------------------------------------
def read_image_file(file_path):
    """
    Read image from various formats commonly used in remote sensing datasets.
    
    Supports:
    - GeoTIFF (.tif, .tiff)
    - NetCDF (.nc)
    - HDF5 (.h5, .hdf5)
    - HDF-EOS (.hdf)
    - PNG/JPEG (standard image formats)
    
    Returns:
        numpy array of shape (H, W, 3) with RGB values in [0, 1], or None if failed
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    # Method 1: Standard image formats (PNG, JPEG, etc.)
    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        try:
            img = Image.open(file_path).convert("RGB")
            return np.array(img) / 255.0
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name} as image: {e}")
            return None
    
    # Method 2: GeoTIFF
    if ext in ['.tif', '.tiff'] and HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                # Read RGB bands (assumes bands 1, 2, 3 or similar)
                # Try to find RGB bands
                if src.count >= 3:
                    # Read first 3 bands as RGB
                    red = src.read(1)
                    green = src.read(2) if src.count > 1 else red
                    blue = src.read(3) if src.count > 2 else red
                    
                    # Normalize to [0, 1]
                    def normalize_band(band):
                        band = np.nan_to_num(band, nan=0.0)
                        if band.max() > band.min():
                            return (band - band.min()) / (band.max() - band.min())
                        return band
                    
                    rgb = np.stack([
                        normalize_band(red),
                        normalize_band(green),
                        normalize_band(blue)
                    ], axis=-1)
                    return rgb
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name} as GeoTIFF: {e}")
    
    # Method 3: NetCDF
    if ext == '.nc' and HAS_XARRAY:
        try:
            ds = xr.open_dataset(file_path)
            
            # Try to find RGB bands or reflectance data
            # Common variable names in remote sensing datasets
            rgb_vars = []
            for var_name in ds.data_vars:
                var_lower = var_name.lower()
                if any(keyword in var_lower for keyword in ['red', 'r', 'b01', 'band1', 'refl_b01']):
                    rgb_vars.append((var_name, 'r'))
                elif any(keyword in var_lower for keyword in ['green', 'g', 'b04', 'band4', 'refl_b04']):
                    rgb_vars.append((var_name, 'g'))
                elif any(keyword in var_lower for keyword in ['blue', 'b', 'b03', 'band3', 'refl_b03']):
                    rgb_vars.append((var_name, 'b'))
            
            # If we found RGB variables, use them
            if len(rgb_vars) >= 3:
                # Sort by color
                rgb_dict = {color: None for color in ['r', 'g', 'b']}
                for var_name, color in rgb_vars:
                    if rgb_dict[color] is None:
                        rgb_dict[color] = ds[var_name].values
                
                # Get first available band if specific color not found
                if rgb_dict['r'] is None:
                    rgb_dict['r'] = list(ds.data_vars.values())[0].values
                if rgb_dict['g'] is None:
                    rgb_dict['g'] = rgb_dict['r']
                if rgb_dict['b'] is None:
                    rgb_dict['b'] = rgb_dict['r']
                
                # Normalize
                def normalize_band(band):
                    band = np.nan_to_num(band, nan=0.0)
                    if band.max() > band.min():
                        return (band - band.min()) / (band.max() - band.min())
                    return band
                
                rgb = np.stack([
                    normalize_band(rgb_dict['r']),
                    normalize_band(rgb_dict['g']),
                    normalize_band(rgb_dict['b'])
                ], axis=-1)
                
                ds.close()
                return rgb
            else:
                # Fallback: use first 3 data variables
                data_vars = list(ds.data_vars)
                if len(data_vars) >= 3:
                    red = ds[data_vars[0]].values
                    green = ds[data_vars[1]].values
                    blue = ds[data_vars[2]].values
                    
                    def normalize_band(band):
                        band = np.nan_to_num(band, nan=0.0)
                        if band.max() > band.min():
                            return (band - band.min()) / (band.max() - band.min())
                        return band
                    
                    rgb = np.stack([
                        normalize_band(red),
                        normalize_band(green),
                        normalize_band(blue)
                    ], axis=-1)
                    ds.close()
                    return rgb
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name} as NetCDF: {e}")
    
    # Method 4: HDF5
    if ext in ['.h5', '.hdf5'] and HAS_H5PY:
        try:
            with h5py.File(file_path, 'r') as f:
                # Try to find RGB datasets
                def find_rgb_datasets(group):
                    """Recursively find RGB datasets."""
                    rgb = {'r': None, 'g': None, 'b': None}
                    
                    def search(group, depth=0):
                        if depth > 3:  # Limit recursion
                            return
                        for key in group.keys():
                            item = group[key]
                            key_lower = key.lower()
                            
                            if isinstance(item, h5py.Dataset):
                                if any(kw in key_lower for kw in ['red', 'r', 'b01', 'refl_b01']) and rgb['r'] is None:
                                    rgb['r'] = np.array(item)
                                elif any(kw in key_lower for kw in ['green', 'g', 'b04', 'refl_b04']) and rgb['g'] is None:
                                    rgb['g'] = np.array(item)
                                elif any(kw in key_lower for kw in ['blue', 'b', 'b03', 'refl_b03']) and rgb['b'] is None:
                                    rgb['b'] = np.array(item)
                            elif isinstance(item, h5py.Group):
                                search(item, depth + 1)
                    
                    search(group)
                    return rgb
                
                rgb_dict = find_rgb_datasets(f)
                
                # If we found RGB, use it; otherwise use first datasets
                if all(rgb_dict.values()):
                    def normalize_band(band):
                        band = np.nan_to_num(band, nan=0.0)
                        if band.max() > band.min():
                            return (band - band.min()) / (band.max() - band.min())
                        return band
                    
                    rgb = np.stack([
                        normalize_band(rgb_dict['r']),
                        normalize_band(rgb_dict['g']),
                        normalize_band(rgb_dict['b'])
                    ], axis=-1)
                    return rgb
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name} as HDF5: {e}")
    
    # Method 5: HDF-EOS (MODIS format)
    if ext == '.hdf' and HAS_PYHDF:
        try:
            hdf = SD.SD(str(file_path), SD.SDC.READ)
            datasets = hdf.datasets()
            
            # Try to find RGB bands
            band_names = {
                'red': ['sur_refl_b01', 'sur_refl_b01_1', 'Band1', 'red'],
                'green': ['sur_refl_b04', 'sur_refl_b04_1', 'Band4', 'green'],
                'blue': ['sur_refl_b03', 'sur_refl_b03_1', 'Band3', 'blue']
            }
            
            bands = {}
            for color, names in band_names.items():
                for name in names:
                    if name in datasets:
                        sds = hdf.select(name)
                        data = sds.get()
                        attrs = sds.attributes()
                        scale = attrs.get('scale_factor', 0.0001)
                        data = np.clip(data * scale, 0, 1)
                        bands[color] = data
                        sds.endaccess()
                        break
            
            hdf.end()
            
            if len(bands) == 3:
                rgb = np.stack([bands['red'], bands['green'], bands['blue']], axis=-1)
                return rgb
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name} as HDF-EOS: {e}")
    
    return None


# ------------------------------------------------------------
# 4. PyTorch Dataset for batching images
# ------------------------------------------------------------
class WildfireImageDataset(Dataset):
    """Dataset for loading and preprocessing wildfire images."""
    
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        try:
            # Read image
            rgb = read_image_file(file_path)
            if rgb is None:
                # Return dummy image if read fails
                rgb = np.zeros((224, 224, 3), dtype=np.float32)
            
            # Convert to PIL Image
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(rgb_uint8)
            
            # Apply transforms
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transform(img)
            
            return img_tensor, str(file_path)
        except Exception as e:
            # Return dummy image on error
            dummy_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                img_tensor = self.transform(dummy_img)
            else:
                img_tensor = transform(dummy_img)
            return img_tensor, str(file_path)


# ------------------------------------------------------------
# 5. Extract embeddings in batches
# ------------------------------------------------------------
def extract_embeddings_batch(model, dataloader, device):
    """
    Extract embeddings from a batch of images.
    
    Args:
        model: Pretrained CNN model (ResNet50)
        dataloader: DataLoader with image batches
        device: torch device (cuda or cpu)
    
    Yields:
        (batch_paths, batch_embeddings) tuples where:
        - batch_paths: list of file paths
        - batch_embeddings: numpy array of shape [batch_size, 2048]
    """
    model.eval()
    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            batch_images = batch_images.to(device)
            
            # Extract embeddings
            embeddings = model(batch_images)
            
            # Flatten from [batch, 2048, 1, 1] → [batch, 2048]
            embeddings = embeddings.squeeze().cpu().numpy()
            
            # Handle single image case
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Convert batch_paths to list if it's a tensor
            if isinstance(batch_paths, torch.Tensor):
                batch_paths = [str(p) for p in batch_paths]
            elif not isinstance(batch_paths, list):
                batch_paths = list(batch_paths)
            
            yield batch_paths, embeddings


# ------------------------------------------------------------
# 6. Parse filename to extract metadata (date, event ID, etc.)
# ------------------------------------------------------------
def parse_filename_metadata(file_path):
    """
    Try to extract metadata from filename.
    Common patterns in wildfire datasets:
    - Date in filename (YYYY-MM-DD, YYYYMMDD, etc.)
    - Event ID or fire ID
    - Tile or location info
    
    Returns:
        dict with extracted metadata
    """
    filename = Path(file_path).stem  # filename without extension
    metadata = {
        'filename': Path(file_path).name,
        'date': None,
        'event_id': None,
        'tile': None
    }
    
    # Try to extract date (various formats)
    import re
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{4}\d{2}\d{2})',     # YYYYMMDD
        r'(\d{4})\.(\d{3})',      # YYYY.DOY (day of year)
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            if '-' in match.group(0):
                metadata['date'] = match.group(0)
            elif len(match.group(0)) == 8:
                # YYYYMMDD
                date_str = match.group(0)
                try:
                    metadata['date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                except:
                    pass
            break
    
    # Try to extract event/fire ID
    event_match = re.search(r'(?:event|fire|id)[_-]?(\d+)', filename, re.IGNORECASE)
    if event_match:
        metadata['event_id'] = event_match.group(1)
    
    # Try to extract tile (h##v## format)
    tile_match = re.search(r'[hH](\d+)[vV](\d+)', filename)
    if tile_match:
        metadata['tile'] = f"h{tile_match.group(1)}v{tile_match.group(2)}"
    
    return metadata


# ------------------------------------------------------------
# 7. Main pipeline: iterate over WildfireSpreadTS files
# ------------------------------------------------------------
def build_wildfirespreadts_embedding_table(
    dataset_dir="data/raw/wildfirespreadts",
    output_path="data/interim/wildfirespreadts_embeddings.parquet",
    image_extensions=None,
    max_images=None  # Limit for testing (None = process all)
):
    """
    Build embedding table from WildfireSpreadTS imagery.
    
    Args:
        dataset_dir: Directory containing WildfireSpreadTS dataset
        output_path: Path to save embeddings Parquet file
        image_extensions: List of file extensions to process (default: common image formats)
    
    Returns:
        pandas DataFrame with embeddings
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: WildfireSpreadTS directory not found: {dataset_path}")
        print("Please download the dataset first:")
        print("  python scripts/download_wildfirespreadts.py")
        return None
    
    if image_extensions is None:
        image_extensions = ['.tif', '.tiff', '.nc', '.h5', '.hdf5', '.hdf', '.png', '.jpg', '.jpeg']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = load_resnet50_embedding_model().to(device)
    print("Loaded ResNet50 embedding model")
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(f"*{ext}")))
        image_files.extend(list(dataset_path.rglob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} image files")
    
    # Limit for testing if specified
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"TEST MODE: Processing only first {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No image files found. The dataset may need to be explored first:")
        print("  python scripts/explore_wildfirespreadts.py")
        print("  or open notebooks/05_explore_wildfirespreadts.ipynb")
        return None
    
    records = []
    
    # Create dataset and dataloader for batching
    batch_size = 32 if device == "cuda" else 8  # Larger batches on GPU
    dataset = WildfireImageDataset(image_files, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device == "cuda" else 2,  # Parallel loading
        pin_memory=True if device == "cuda" else False
    )
    
    # Progress tracking
    import time
    start_time = time.time()
    
    print(f"\nStarting embedding extraction...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Total images: {len(image_files)}")
    print("")
    
    processed_count = 0
    for batch_idx, (batch_paths, batch_embeddings) in enumerate(extract_embeddings_batch(model, dataloader, device)):
        current_batch_size = len(batch_paths)
        processed_count += current_batch_size
        
        # Progress updates
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            if processed_count > 0:
                rate = processed_count / elapsed  # images per second
                remaining = (len(image_files) - processed_count) / rate if rate > 0 else 0
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                print(f"Processing batch {batch_idx+1} ({processed_count}/{len(image_files)} images)... "
                      f"({rate:.2f} img/s, ETA: {eta_min}m {eta_sec}s)")
        
        # Process each image in the batch
        for i, img_path_str in enumerate(batch_paths):
            img_path = Path(img_path_str)
            emb = batch_embeddings[i]
            
            # Parse filename metadata
            metadata = parse_filename_metadata(img_path)
            
            # Create record
            record = metadata.copy()
            
            # Add embedding dimensions
            for j, val in enumerate(emb):
                record[f'embedding_{j}'] = val
            
            records.append(record)
    
    total_time = time.time() - start_time
    print(f"\n✓ Completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Average: {total_time/len(records):.2f} seconds per image")
    
    if len(records) == 0:
        print("No embeddings extracted. Check image file formats.")
        return None
    
    df = pd.DataFrame(records)
    print(f"Extracted embeddings from {len(records)} images")
    print(f"Embedding dimension: {len(emb)}")
    
    return df


# ------------------------------------------------------------
# 8. Save embeddings as Parquet
# ------------------------------------------------------------
def save_embeddings(df, out_path="data/interim/wildfirespreadts_embeddings.parquet"):
    """Save embeddings DataFrame to Parquet file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)
    print(f"Saved WildfireSpreadTS embeddings → {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 8. Run the full script
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/wildfirespreadts"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/interim/wildfirespreadts_embeddings.parquet"
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print("=" * 70)
    print("WildfireSpreadTS Embeddings Extraction")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output path: {output_path}")
    if max_images:
        print(f"TEST MODE: Processing only {max_images} images")
    print("")
    
    df = build_wildfirespreadts_embedding_table(dataset_dir, output_path, max_images=max_images)
    
    if df is not None:
        save_embeddings(df, output_path)
        print("\n✓ Done! Embeddings ready for XGBoost feature engineering.")
        print("\nNote: You can now skip downloading separate MODIS files.")
        print("      The WildfireSpreadTS imagery provides all needed data.")
    else:
        print("\n✗ Failed to extract embeddings.")
        sys.exit(1)

