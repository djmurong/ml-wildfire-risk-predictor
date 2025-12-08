#!/usr/bin/env python3
"""
Downloads MODIS MOD09A1 (8-day Surface Reflectance) files for the United States
for the years 2010–2020.

Based on NASA LAADS DAAC download scripts:
https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/

You need a NASA LAADS Bearer Token.
Get your token here: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/
(Click "Token" button after logging in)

This script:
  - Selects only the MODIS tiles that cover the US
  - Iterates dates from 2010-01-01 to 2020-12-31
  - Downloads only MOD09A1 files that match the tile & date
  - Saves output under data/raw/modis/

MODIS MOD09A1 directory structure:
https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD09A1/YYYY/DDD/
where:
    YYYY = year
    DDD  = day-of-year (1–365/366)

"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# CONFIG
# --------------------------

# Get Bearer token from environment variables
# Get your token from: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/
LAADS_TOKEN = os.getenv("LAADS_TOKEN")

if not LAADS_TOKEN:
    raise ValueError(
        "LAADS_TOKEN must be set in .env file.\n"
        "Get your token from: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/\n"
        "See .env.example for template."
    )

OUT_DIR = Path("data/raw/modis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MODIS tiles that cover the continental US (CONUS)
US_TILES = [
    "h08v04", "h08v05",
    "h09v04", "h09v05",
    "h10v04", "h10v05",
    "h11v04", "h11v05",
    "h12v04", "h12v05"
]

# Product base URL
BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD09A1"

# Date range
START = datetime(2010, 1, 1)
END   = datetime(2020, 12, 31)


# --------------------------
# Helper: Make authenticated request with Bearer token
# --------------------------

def get_url(url, token, out=None):
    """
    Make an authenticated request to LAADS using Bearer token.
    Based on NASA LAADS Python example.
    
    Args:
        url: URL to request
        token: Bearer token for authentication
        out: Optional file handle to write response to
    
    Returns:
        Response content (if out is None) or None
    """
    headers = {
        'user-agent': 'MODIS-download.py_1.0',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        r = requests.get(url, headers=headers, stream=(out is not None), timeout=300)
        r.raise_for_status()
        
        if out is None:
            return r.text
        else:
            # Stream to file
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    out.write(chunk)
            return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(f"Authentication failed (401). Check your LAADS_TOKEN in .env file.")
        elif e.response.status_code == 404:
            # 404 could mean file doesn't exist or auth failed
            return None
        else:
            raise
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None


# --------------------------
# Main: loop through dates and tiles
# --------------------------

def download_modis():
    """
    Download MODIS MOD09A1 files using LAADS Bearer token authentication.
    Based on NASA LAADS Python download script pattern.
    """
    current = START
    total_downloaded = 0
    total_skipped = 0

    print(f"[INFO] Starting MODIS MOD09A1 download for US tiles {US_TILES}")
    print(f"[INFO] Time range: {START.date()} → {END.date()}")
    print(f"[INFO] Using Bearer token authentication")

    while current <= END:
        year = current.year
        doy = current.timetuple().tm_yday  # day-of-year
        doy_str = f"{doy:03d}"

        # URL for this date - use .json endpoint to get directory listing
        dir_url = f"{BASE_URL}/{year}/{doy_str}"
        json_url = f"{dir_url}.json"

        # Get directory listing as JSON
        json_content = get_url(json_url, LAADS_TOKEN)
        
        if json_content is None:
            # 404 or error - data doesn't exist for this date (normal for some dates)
            current += timedelta(days=8)
            continue
        
        try:
            # Parse JSON directory listing
            dir_listing = json.loads(json_content)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON for {json_url}")
            current += timedelta(days=8)
            continue

        # Process files in directory
        if 'content' not in dir_listing:
            current += timedelta(days=8)
            continue

        # Find files matching our tiles
        for item in dir_listing['content']:
            # Skip directories (size == 0)
            if int(item.get('size', 0)) == 0:
                continue
            
            filename = item['name']
            
            # Check if file matches our tiles
            matches_tile = any(tile in filename for tile in US_TILES)
            if not matches_tile:
                continue
            
            # Check if it's a MOD09A1 file
            if not filename.startswith('MOD09A1'):
                continue

            file_url = f"{dir_url}/{filename}"
            out_path = OUT_DIR / filename

            if out_path.exists() and out_path.stat().st_size > 0:
                # Already downloaded
                total_skipped += 1
                continue

            print(f"[DOWNLOAD] {filename}")
            try:
                with open(out_path, 'wb') as f:
                    get_url(file_url, LAADS_TOKEN, out=f)
                
                # Verify file was downloaded
                if out_path.exists() and out_path.stat().st_size > 0:
                    size_mb = out_path.stat().st_size / 1024 / 1024
                    print(f"  ✓ Downloaded {filename} ({size_mb:.1f} MB)")
                    total_downloaded += 1
                else:
                    print(f"  ✗ Download failed or incomplete: {filename}")
                    if out_path.exists():
                        out_path.unlink()  # Remove incomplete file
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
                if out_path.exists():
                    out_path.unlink()  # Remove incomplete file

        # Step forward by 8 days (MOD09A1 is 8-day composite)
        current += timedelta(days=8)

    print(f"\n[DONE] Download complete.")
    print(f"  Downloaded: {total_downloaded} files")
    print(f"  Skipped (already exist): {total_skipped} files")


# --------------------------
# Run
# --------------------------

if __name__ == "__main__":
    download_modis()
