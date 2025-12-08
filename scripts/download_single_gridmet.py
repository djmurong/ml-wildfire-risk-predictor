"""
Download a single gridMET file, with resume capability.
Usage: py scripts/download_single_file.py <filename>
Example: py scripts/download_single_file.py pr_2021.nc
"""
import sys
import requests
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gridmet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://www.northwestknowledge.net/metdata/data"

def download_file(filename, force=False):
    """Download a single file, resuming if partial."""
    url = f"{BASE_URL}/{filename}"
    output_path = OUTPUT_DIR / filename
    
    # Check if file exists
    existing_size = 0
    if output_path.exists() and not force:
        existing_size = output_path.stat().st_size
        print(f"Found existing file: {filename} ({existing_size / 1024 / 1024:.1f} MB)")
    
    try:
        # Get file info from server
        print(f"Checking server for {filename}...")
        head_response = requests.head(url, timeout=30, allow_redirects=True)
        head_response.raise_for_status()
        
        total_size = int(head_response.headers.get('content-length', 0))
        if total_size == 0:
            # HEAD might not return content-length, try GET with range
            print("Content-length not available, checking file size...")
            range_response = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=30)
            if 'content-range' in range_response.headers:
                total_size = int(range_response.headers['content-range'].split('/')[-1])
        
        if total_size > 0:
            print(f"Server file size: {total_size / 1024 / 1024:.1f} MB")
            
            if existing_size == total_size:
                print(f"[OK] File {filename} is already complete!")
                return True
            elif existing_size > 0:
                print(f"Resuming download from {existing_size} bytes...")
        
        # Download the file
        headers = {}
        if existing_size > 0 and total_size > existing_size:
            # Resume download
            headers['Range'] = f'bytes={existing_size}-'
            mode = 'ab'  # append mode
        else:
            # Full download
            mode = 'wb'
            existing_size = 0
        
        print(f"Downloading {filename}...", end="", flush=True)
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        downloaded = existing_size
        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading {filename}... {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        print(f"\r[OK] Downloaded {filename} ({downloaded / 1024 / 1024:.1f} MB)")
        
        # Verify file size
        final_size = output_path.stat().st_size
        if total_size > 0 and final_size != total_size:
            print(f"Warning: File size mismatch. Expected {total_size}, got {final_size}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\r[ERROR] Error downloading {filename}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py scripts/download_single_file.py <filename>")
        print("Example: py scripts/download_single_file.py pr_2021.nc")
        sys.exit(1)
    
    filename = sys.argv[1]
    force = '--force' in sys.argv
    
    if download_file(filename, force=force):
        print(f"\n[OK] Successfully downloaded {filename}")
    else:
        print(f"\n[ERROR] Failed to download {filename}")
        sys.exit(1)

