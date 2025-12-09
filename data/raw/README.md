# Raw Data Directory

This directory contains raw datasets that are **excluded from Git** due to their large size (78GB+).

## Data Sources

### MODIS Active Fire Data
- **Source**: NASA FIRMS
- **How to download**: Run `python src/data/download_data.py` or `python scripts/download_modis_mod09a1.py`
- **Location**: `data/raw/MODIS_24h.csv` and `data/raw/modis/`

### GridMET Climate Data
- **Source**: [Climatology Lab GridMET](https://www.climatologylab.org/gridmet.html)
- **How to download**: Use the scripts in `scripts/` directory
- **Location**: `data/raw/gridmet/`
- **Variables**: pr, tmmx, tmmn, rmax, rmin, srad, vs, th, pet, etr, etc.

### FPA-FOD (Fire Program Analysis - Fire Occurrence Database)
- **Source**: [USGS](https://www.fs.usda.gov/rds/archive/Product/RDS-2013-0009.6/)
- **How to download**: Download from USGS website
- **Location**: `data/raw/fpa_fod/`

### Sentinel-2 Data
- **Source**: ESA Copernicus
- **How to download**: Use `scripts/download_sentinel2.py`
- **Location**: `data/raw/sentinel2/`

## Notes

- All data files (`.nc`, `.h5`, `.zip`, etc.) are excluded from Git via `.gitignore`
- Data should be downloaded/regenerated on each machine or cluster
- For cluster usage, transfer data separately or download directly on the cluster
- See project README.md for full setup instructions

