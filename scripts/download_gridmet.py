"""
Python script to download gridMET data from Northwest Knowledge Network.
This works on all platforms without needing wget.

Usage:
    py scripts/download_gridmet.py [output_directory]
    
If output_directory is not specified, files will be saved to data/raw/gridmet/
"""
import sys
import requests
from pathlib import Path
from time import sleep

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set output directory
if len(sys.argv) > 1:
    OUTPUT_DIR = Path(sys.argv[1])
else:
    OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gridmet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base URL for gridMET data (from https://www.climatologylab.org/gridmet.html)
BASE_URL = "http://www.northwestknowledge.net/metdata/data"

# Variables and years to download
VARIABLES = ["vpd", "pr", "rmin", "rmax", "srad", "tmmn", "tmmx", "vs", "erc", "bi", "fm100", "fm1000"]
YEARS = list(range(2010, 2020))

def download_file(filename, url, output_path):
    """Download a single file with progress indication."""
    # if output_path.exists():
    #     print(f"✓ Skipping {filename} (already exists)")
    #     return True
    
    try:
        print(f"Downloading {filename}...", end="", flush=True)
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading {filename}... {percent:.1f}%", end="", flush=True)
        
        size_mb = downloaded / 1024 / 1024
        print(f"\r✓ Downloaded {filename} ({size_mb:.1f} MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\r✗ Error downloading {filename}: {e}")
        return False

def main():
    """Download all gridMET files."""
    print(f"Downloading gridMET data to: {OUTPUT_DIR}")
    print("")
    
    total_files = len(VARIABLES) * len(YEARS)
    current = 0
    failed = 0
    
    for var in VARIABLES:
        for year in YEARS:
            current += 1
            filename = f"{var}_{year}.nc"
            url = f"{BASE_URL}/{filename}"
            output_path = OUTPUT_DIR / filename
            
            print(f"[{current}/{total_files}] Processing {filename}...")
            
            if not download_file(filename, url, output_path):
                failed += 1
            
            # Small delay to avoid overwhelming the server
            sleep(0.5)
    
    print("")
    print("=== Download Summary ===")
    print(f"Total files: {total_files}")
    print(f"Successful: {total_files - failed}")
    print(f"Failed: {failed}")
    print("")
    print(f"Files saved to: {OUTPUT_DIR}")
    
    if failed > 0:
        print("")
        print("Some downloads failed. You can run this script again to retry failed downloads.")
        sys.exit(1)

if __name__ == "__main__":
    main()

