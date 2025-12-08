#!/usr/bin/env python3
"""
prepare_tabular.py

Cleans raw inputs (FOD .gdb, GridMET NetCDFs), creates a spatial grid,
aggregates daily GridMET variables to each grid cell, and constructs
a tabular dataset with labels (ignition) for supervised training.

Outputs:
    data/processed/features_tabular.parquet
    data/processed/grid_definition.parquet

Usage:
    python src/data/prepare_tabular.py
"""

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point, box
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
RAW_DIR = Path("data/raw")
GRIDMET_DIR = RAW_DIR / "gridmet"   # netcdf files
FOD_GDB_DIR = RAW_DIR / "fpa_fod"   # directory containing FPA_FOD_*.gdb folder(s), or path to .gdb folder itself
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# grid resolution in degrees (approx): e.g., 0.05 deg ~5km
GRID_RES = 0.05

# label window: we will only use labels up to this year (your FOD currently ~2020)
LABEL_END_YEAR = 2020

# GridMET variables to extract (should exist as NetCDFs)
GRIDMET_VARS = {
    "rmin": "rmin", "rmax": "rmax", "srad": "srad",
    "tmmx": "tmmx", "tmmn": "tmmn", "pr": "pr",
    "vpd": "vpd", "vs": "vs", "erc": "erc",
    "bi": "bi", "fm100": "fm100", "fm1000": "fm1000"
}

# ---------------------------
# Helpers
# ---------------------------
def find_fod_gdb(fod_gdb_dir):
    """
    Find FOD geodatabase (.gdb folder) in the specified directory.
    
    Args:
        fod_gdb_dir: Path to directory containing .gdb folder(s), or path to .gdb folder itself
    
    Returns:
        Path to .gdb folder
    """
    fod_gdb_dir = Path(fod_gdb_dir)
    
    # Check if the path itself is a .gdb folder
    if fod_gdb_dir.is_dir() and fod_gdb_dir.suffix == ".gdb":
        return fod_gdb_dir
    
    # Otherwise, search for .gdb folders inside the directory
    if not fod_gdb_dir.exists():
        raise FileNotFoundError(f"FOD directory does not exist: {fod_gdb_dir}")
    
    if not fod_gdb_dir.is_dir():
        raise FileNotFoundError(f"FOD path is not a directory: {fod_gdb_dir}")
    
    # Look for .gdb folders
    for p in fod_gdb_dir.iterdir():
        if p.is_dir() and p.suffix == ".gdb":
            return p
    
    # Also try glob pattern (handles cases where .gdb might be in subdirectories)
    candidates = list(fod_gdb_dir.glob("*.gdb"))
    if candidates:
        return candidates[0]
    
    # Try recursive search
    candidates = list(fod_gdb_dir.rglob("*.gdb"))
    if candidates:
        return candidates[0]
    
    raise FileNotFoundError(
        f"No .gdb folder found in {fod_gdb_dir}. "
        f"Please ensure FPA_FOD_*.gdb folder exists in this directory."
    )


def build_grid(bounds, res=GRID_RES):
    """Create a grid (square cells) covering bounds=(minx,miny,maxx,maxy)."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, res)
    ys = np.arange(miny, maxy, res)
    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cell_geom = box(x, y, x+res, y+res)
            cell_id = f"cell_{i}_{j}"
            rows.append({"cell_id": cell_id, "geometry": cell_geom, "xmin": x, "ymin": y,
                         "xmax": x+res, "ymax": y+res})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf


def process_single_variable(var, file_paths, lons, lats, cell_ids, lon_coord, lat_coord, gridmet_dir):
    """
    Process a single GridMET variable (worker function for multiprocessing).
    
    Returns:
        List of dictionaries with results for this variable
    """
    results = []
    print(f"[WORKER] ===== Processing variable: {var} ({len(file_paths)} files) =====", flush=True)
    
    total_days = 0
    for file_idx, fp in enumerate(file_paths):
        print(f"[WORKER-{var}] Opening file {file_idx+1}/{len(file_paths)}: {fp.name}", flush=True)
        try:
            # Open one file at a time
            var_ds = xr.open_dataset(fp)
            
            # Get the actual data variable name from the dataset
            data_vars = list(var_ds.data_vars.keys())
            if not data_vars:
                print(f"[WARN-{var}] No data variables found in {fp.name}, skipping")
                var_ds.close()
                continue
            
            actual_var_name = data_vars[0]
            data_var = var_ds[actual_var_name]
            
            # Get time coordinate
            time_dim = None
            for time_name in ['time', 'day', 'date']:
                if time_name in var_ds.coords or time_name in var_ds.dims:
                    time_dim = time_name
                    break
            
            if time_dim is None:
                print(f"[WARN-{var}] Could not find time dimension in {fp.name}, skipping")
                var_ds.close()
                continue
            
            times = pd.to_datetime(var_ds[time_dim].values)
            print(f"[WORKER-{var}] File {file_idx+1}/{len(file_paths)} ({fp.name}): {times.min()} -> {times.max()} ({len(times)} days)", flush=True)
            
            # Get coordinates once (before batching) to compute indices
            lon_coords = data_var[lon_coord].values
            lat_coords = data_var[lat_coord].values
            
            print(f"  [{var}] Computing indices for {len(lons)} points...", flush=True)
            
            lon_min, lon_max = lon_coords.min(), lon_coords.max()
            lat_min, lat_max = lat_coords.min(), lat_coords.max()
            
            if len(lon_coords) > 1:
                lon_spacing = (lon_max - lon_min) / (len(lon_coords) - 1)
            else:
                lon_spacing = 0.0
            if len(lat_coords) > 1:
                lat_spacing = (lat_max - lat_min) / (len(lat_coords) - 1)
            else:
                lat_spacing = 0.0
            
            lon_indices_float = (np.array(lons) - lon_min) / lon_spacing if lon_spacing > 0 else np.zeros(len(lons))
            lat_indices_float = (np.array(lats) - lat_min) / lat_spacing if lat_spacing > 0 else np.zeros(len(lats))
            
            lon_indices = np.clip(np.round(lon_indices_float).astype(np.int32), 0, len(lon_coords) - 1)
            lat_indices = np.clip(np.round(lat_indices_float).astype(np.int32), 0, len(lat_coords) - 1)
            
            print(f"  [{var}] Indices computed. Processing in small batches...", flush=True)
            
            # Process in smaller batches to reduce memory usage
            # Reduced from 100 to 5 days per batch to avoid memory errors
            BATCH_SIZE = 5
            num_batches = (len(times) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(times))
                batch_times = times[start_idx:end_idx]
                
                # Progress reporting
                if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
                    progress_pct = ((batch_idx + 1) / num_batches) * 100
                    print(f"  [{var}] Batch {batch_idx+1}/{num_batches} ({progress_pct:.1f}%) - Days {start_idx}-{end_idx-1}", flush=True)
                
                # Select time slice and immediately extract only the points we need
                # This avoids loading the full spatial grid into memory
                try:
                    time_slice = {time_dim: batch_times}
                    # Use isel for faster indexing, then select only the points we need
                    arr_batch = data_var.isel({time_dim: slice(start_idx, end_idx)})
                except (KeyError, ValueError):
                    arr_batch = data_var.isel({time_dim: slice(start_idx, end_idx)})
                
                # Extract only the values we need using vectorized indexing
                # This avoids loading the full (time, lat, lon) array
                try:
                    # Use xarray's isel with DataArray for efficient indexing
                    values_array = arr_batch.isel({lon_coord: xr.DataArray(lon_indices, dims='points'),
                                                   lat_coord: xr.DataArray(lat_indices, dims='points')}).values
                except Exception:
                    # Fallback: load array and use numpy advanced indexing
                    # But only if xarray method fails
                    data_array = arr_batch.values  # Shape: (time, lat, lon) or (time, y, x)
                    # Use vectorized indexing - much faster than loop
                    values_array = data_array[:, lat_indices, lon_indices]
                
                # Convert to numpy array and ensure it's 2D (time, points)
                if values_array.ndim == 1:
                    # Single time step
                    values_array = values_array.reshape(1, -1)
                
                # Store results vectorized
                batch_dates = pd.to_datetime(batch_times)
                num_days_batch = len(batch_times)
                num_cells = len(cell_ids)
                
                batch_cell_ids = np.tile(cell_ids, num_days_batch)
                batch_dates_list = np.repeat(batch_dates, num_cells)
                batch_values = values_array.flatten()
                
                # Extend results efficiently
                for i in range(len(batch_cell_ids)):
                    results.append({
                        "cell_id": batch_cell_ids[i],
                        "date": batch_dates_list[i],
                        var: float(batch_values[i]) if np.isfinite(batch_values[i]) else np.nan
                    })
                
                # Explicitly delete large arrays to free memory
                del arr_batch, values_array, batch_values
                
                total_days += len(batch_times)
            
            var_ds.close()
            
        except Exception as e:
            print(f"[ERROR-{var}] Failed to process {fp.name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[WORKER-{var}] COMPLETE: {total_days} days, {len(results)} records", flush=True)
    return results


def aggregate_gridmet_to_grid(grid_gdf, gridmet_dir, use_multiprocessing=True, n_workers=None):
    """
    For each grid cell and day, extract GridMET variables by nearest pixel.
    GridMET NetCDF files have dimensions: time, lat, lon (or time, y, x)
    Output: DataFrame with rows per (cell_id, date) and columns for each var.
    
    Args:
        use_multiprocessing: If True, process variables in parallel
        n_workers: Number of parallel workers (None = auto-detect)
    """
    # Determine grid cell centers
    cell_centers = grid_gdf.geometry.centroid
    lons = cell_centers.x.values
    lats = cell_centers.y.values
    cell_ids = grid_gdf["cell_id"].values

    # Find all GridMET files and group by variable
    all_paths = list(Path(gridmet_dir).glob("*.nc"))
    if not all_paths:
        raise FileNotFoundError(f"No GridMET NetCDF files found in {gridmet_dir}")
    
    # Group files by variable name
    var_files = {}
    for var in GRIDMET_VARS.keys():
        var_paths = [p for p in all_paths if p.stem.startswith(f"{var}_")]
        if var_paths:
            var_files[var] = sorted(var_paths)
        else:
            print(f"[WARN] GridMET var {var} not found; skipping")
    
    if not var_files:
        raise FileNotFoundError("No GridMET variables found in directory")

    # Open one file to inspect structure
    sample_path = next(iter(var_files.values()))[0]
    sample_ds = xr.open_dataset(sample_path)
    
    # Detect coordinate names
    coord_names = {}
    if 'lon' in sample_ds.coords and 'lat' in sample_ds.coords:
        coord_names = {'lon': 'lon', 'lat': 'lat'}
    elif 'x' in sample_ds.coords and 'y' in sample_ds.coords:
        coord_names = {'lon': 'x', 'lat': 'y'}
    else:
        dims = list(sample_ds.dims.keys())
        if 'lon' in dims and 'lat' in dims:
            coord_names = {'lon': 'lon', 'lat': 'lat'}
        elif 'x' in dims and 'y' in dims:
            coord_names = {'lon': 'x', 'lat': 'y'}
        else:
            raise RuntimeError(f"Cannot determine coordinate names. Available coords: {list(sample_ds.coords.keys())}, dims: {dims}")
    
    lon_coord = coord_names['lon']
    lat_coord = coord_names['lat']
    sample_ds.close()
    
    print(f"[INFO] Using coordinates: {lon_coord} (lon), {lat_coord} (lat)", flush=True)
    print(f"[INFO] Found {len(var_files)} variables to process", flush=True)

    # Process variables in parallel or sequentially
    if use_multiprocessing and len(var_files) > 1:
        if n_workers is None:
            n_workers = min(len(var_files), cpu_count())
        print(f"[INFO] Using multiprocessing with {n_workers} workers", flush=True)
        
        # Prepare arguments for worker function
        worker_args = [
            (var, file_paths, lons, lats, cell_ids, lon_coord, lat_coord, gridmet_dir)
            for var, file_paths in var_files.items()
        ]
        
        # Process in parallel
        with Pool(processes=n_workers) as pool:
            var_results_list = pool.starmap(process_single_variable, worker_args)
        
        # Combine all results
        results = []
        for var_results in var_results_list:
            results.extend(var_results)
    else:
        # Sequential processing (fallback or if disabled)
        print(f"[INFO] Processing variables sequentially", flush=True)
        results = []
        for var, file_paths in var_files.items():
            var_results = process_single_variable(var, file_paths, lons, lats, cell_ids, lon_coord, lat_coord, gridmet_dir)
            results.extend(var_results)
    
    # Convert results into DataFrame
    if not results:
        raise RuntimeError("No results sampled from GridMET; check dataset")
    
    df = pd.DataFrame(results)
    
    # Pivot so columns are var names (one row per cell_id, date)
    df_pivot = df.pivot_table(
        index=["cell_id", "date"],
        values=list(GRIDMET_VARS.keys()),
        aggfunc="first"
    ).reset_index()
    
    return df_pivot


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print("[STEP] Locate FOD geodatabase")
    fod_gdb = find_fod_gdb(FOD_GDB_DIR)
    print(f"[INFO] Using FOD GDB at: {fod_gdb}")

    print("[STEP] Read FOD points")
    # FOD geodatabase has multiple layers - we need the 'Fires' layer
    try:
        fod_layers = gpd.read_file(fod_gdb, layer='Fires')  # Read the Fires layer specifically
    except (ValueError, KeyError) as e:
        # Fallback: try reading without layer specification (may read default layer)
        print(f"[WARN] Could not read 'Fires' layer, trying default: {e}")
        fod_layers = gpd.read_file(fod_gdb)
    
    print(f"[INFO] Loaded {len(fod_layers)} FOD records")
    print(f"[INFO] Columns: {list(fod_layers.columns)[:10]}...")
    
    # Normalize column names to lowercase
    fod = fod_layers.rename(columns={c: c.lower() for c in fod_layers.columns})
    # We expect columns 'latitude','longitude','discovery_date'
    if 'latitude' not in fod.columns or 'longitude' not in fod.columns:
        # try geometry reading if lat/lon missing
        if 'geometry' in fod.columns:
            fod['longitude'] = fod.geometry.x
            fod['latitude'] = fod.geometry.y
        else:
            raise RuntimeError("FOD missing lat/lon and geometry")

    # parse discovery date
    if 'discovery_date' in fod.columns:
        fod['discovery_date'] = pd.to_datetime(fod['discovery_date'], errors='coerce')
    else:
        # try DISCOVERY_DATE uppercase
        if 'DISCOVERY_DATE' in fod.columns:
            fod['discovery_date'] = pd.to_datetime(fod['DISCOVERY_DATE'], errors='coerce')
        else:
            raise RuntimeError("FOD dataset missing DISCOVERY_DATE column")

    # filter labels up to LABEL_END_YEAR
    fod = fod[fod['discovery_date'].notna()]
    fod = fod[fod['discovery_date'].dt.year <= LABEL_END_YEAR].copy()
    print(f"[INFO] FOD records up to {LABEL_END_YEAR}: {len(fod)}")

    # Build bounding box over label points plus a small buffer
    bounds = fod.total_bounds  # minx,miny,maxx,maxy
    minx, miny, maxx, maxy = bounds
    # expand slightly
    pad = 0.5
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    print(f"[INFO] Spatial bounds for grid: {bounds}")

    print("[STEP] Build spatial grid")
    grid_gdf = build_grid(bounds, res=GRID_RES)
    print(f"[INFO] Grid cells created: {len(grid_gdf)}")
    grid_gdf.to_parquet(OUT_DIR / "grid_definition.parquet", index=False)

    print("[STEP] Assign FOD points to grid cells")
    # Create geometry for FOD points
    fod['geometry'] = fod.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
    fod_gdf = gpd.GeoDataFrame(fod, geometry='geometry', crs="EPSG:4326")
    
    # Use spatial join for better performance
    # Ensure grid_gdf has geometry set
    grid_gdf = grid_gdf.set_geometry("geometry")
    
    # Perform spatial join to assign cell_id to each FOD point
    fod_gdf = gpd.sjoin(
        fod_gdf,
        grid_gdf[['cell_id', 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Remove the index_right column added by sjoin
    if 'index_right' in fod_gdf.columns:
        fod_gdf = fod_gdf.drop(columns=['index_right'])
    
    # Filter out points that didn't match any cell
    fod_gdf = fod_gdf[fod_gdf['cell_id'].notna()]
    # build ignition labels: for each cell and date (date = discovery_date.date) mark ignited=1
    fod_gdf['date'] = fod_gdf['discovery_date'].dt.floor('D')
    label_df = fod_gdf.groupby(['cell_id','date']).size().reset_index(name='n_fires')
    label_df['ignited'] = 1
    print(f"[INFO] Label rows (cell_id,date) = {len(label_df)}")

    print("[STEP] Sample GridMET to grid")
    gridmet_df = aggregate_gridmet_to_grid(grid_gdf, GRIDMET_DIR)
    print(f"[INFO] GridMET sampled rows: {len(gridmet_df)}")

    print("[STEP] Merge features and labels")
    merged = gridmet_df.merge(label_df[['cell_id','date','ignited']], on=['cell_id','date'], how='left')
    merged['ignited'] = merged['ignited'].fillna(0).astype(int)
    # Save
    out_path = OUT_DIR / "features_tabular.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[SAVED] Tabular features with labels to {out_path}")


if __name__ == "__main__":
    main()
