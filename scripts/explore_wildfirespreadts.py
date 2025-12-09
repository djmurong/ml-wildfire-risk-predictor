"""
Helper script to explore the structure of the downloaded WildfireSpreadTS dataset.

This script helps you understand:
- What files are in the dataset
- Dataset format (NetCDF, HDF5, etc.)
- Available variables/features
- Temporal and spatial coverage

Usage:
    python scripts/explore_wildfirespreadts.py [dataset_directory]
"""

import sys
from pathlib import Path
import json

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set dataset directory
if len(sys.argv) > 1:
    DATASET_DIR = Path(sys.argv[1])
else:
    DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "wildfirespreadts"

if not DATASET_DIR.exists():
    print(f"Error: Dataset directory not found: {DATASET_DIR}")
    print("Please download the dataset first using:")
    print("  python scripts/download_wildfirespreadts.py")
    sys.exit(1)


def get_file_info(filepath):
    """Get basic information about a file."""
    stat = filepath.stat()
    return {
        'name': filepath.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'extension': filepath.suffix.lower()
    }


def detect_file_type(filepath):
    """Try to detect the file type and suggest how to read it."""
    ext = filepath.suffix.lower()
    
    type_map = {
        '.nc': 'NetCDF (use xarray or netCDF4)',
        '.h5': 'HDF5 (use h5py or xarray)',
        '.hdf5': 'HDF5 (use h5py or xarray)',
        '.hdf': 'HDF/HDF-EOS (use h5py, pyhdf, or xarray)',
        '.zip': 'ZIP archive (extract first)',
        '.tar': 'TAR archive (extract first)',
        '.gz': 'GZIP compressed (may need extraction)',
        '.parquet': 'Parquet (use pandas or pyarrow)',
        '.csv': 'CSV (use pandas)',
        '.json': 'JSON (use json or pandas)'
    }
    
    return type_map.get(ext, f'Unknown format ({ext})')


def explore_directory(directory):
    """Explore the dataset directory structure."""
    print("=" * 70)
    print("WildfireSpreadTS Dataset Explorer")
    print("=" * 70)
    print(f"Directory: {directory}")
    print("")
    
    # Find all files
    all_files = list(directory.rglob('*'))
    files_only = [f for f in all_files if f.is_file()]
    dirs_only = [d for d in all_files if d.is_dir()]
    
    print(f"Found {len(files_only)} file(s) and {len(dirs_only)} directory/directories")
    print("")
    
    if not files_only:
        print("No files found. The dataset may need to be extracted.")
        return
    
    # Group files by extension
    by_ext = {}
    total_size = 0
    
    for filepath in files_only:
        info = get_file_info(filepath)
        ext = info['extension'] or '(no extension)'
        if ext not in by_ext:
            by_ext[ext] = []
        by_ext[ext].append(info)
        total_size += info['size_mb']
    
    print("Files by type:")
    print("-" * 70)
    for ext, files in sorted(by_ext.items()):
        count = len(files)
        total_mb = sum(f['size_mb'] for f in files)
        print(f"  {ext:15} {count:4} file(s)  {total_mb:10.2f} MB")
    
    print("-" * 70)
    print(f"  {'Total':15} {len(files_only):4} file(s)  {total_size:10.2f} MB")
    print("")
    
    # Show top-level files
    top_level_files = [f for f in files_only if f.parent == directory]
    if top_level_files:
        print("Top-level files:")
        print("-" * 70)
        for filepath in sorted(top_level_files):
            info = get_file_info(filepath)
            file_type = detect_file_type(filepath)
            print(f"  {info['name']:50} {info['size_mb']:8.2f} MB  ({file_type})")
        print("")
    
    # Show directory structure (first 2 levels)
    if dirs_only:
        print("Directory structure (showing first 2 levels):")
        print("-" * 70)
        top_dirs = sorted(set(d.parent for d in dirs_only if d.relative_to(directory).parts))
        for top_dir in top_dirs[:10]:  # Show first 10
            rel_path = top_dir.relative_to(directory)
            depth = len(rel_path.parts)
            if depth <= 2:
                indent = "  " * depth
                print(f"{indent}{rel_path.name}/")
        if len(top_dirs) > 10:
            print(f"  ... and {len(top_dirs) - 10} more directories")
        print("")
    
    # Try to identify main data files
    print("Suggested next steps:")
    print("-" * 70)
    
    # Check for common data formats
    nc_files = [f for f in files_only if f.suffix.lower() == '.nc']
    h5_files = [f for f in files_only if f.suffix.lower() in ['.h5', '.hdf5']]
    zip_files = [f for f in files_only if f.suffix.lower() == '.zip']
    
    if zip_files:
        print("⚠ Found ZIP files. Extract them first:")
        for zf in zip_files[:3]:
            print(f"  - {zf.name}")
        if len(zip_files) > 3:
            print(f"  ... and {len(zip_files) - 3} more")
        print("")
    
    if nc_files:
        print("✓ Found NetCDF files. You can explore them with:")
        print("  import xarray as xr")
        print(f"  ds = xr.open_dataset('{nc_files[0].relative_to(directory)}')")
        print("  print(ds)")
        print("")
    
    if h5_files:
        print("✓ Found HDF5 files. You can explore them with:")
        print("  import h5py")
        print(f"  f = h5py.File('{h5_files[0].relative_to(directory)}', 'r')")
        print("  print(list(f.keys()))")
        print("")
    
    # Check for README or documentation
    readme_files = [f for f in files_only if 'readme' in f.name.lower() or 'doc' in f.name.lower()]
    if readme_files:
        print("📄 Found documentation files:")
        for rf in readme_files:
            print(f"  - {rf.relative_to(directory)}")
        print("")
    
    print("For more information:")
    print("  - Zenodo: https://zenodo.org/records/8006177")
    print("  - GitHub: https://github.com/SebastianGer/WildfireSpreadTS")


def main():
    """Main exploration function."""
    explore_directory(DATASET_DIR)


if __name__ == "__main__":
    main()

