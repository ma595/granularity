import os
from pathlib import Path
import pandas as pd
import xarray as xr

from standardise_variables import VARIABLE_ALIASES, standardize_variables
from gran_analysis import (
    analyze_what_is_possible_efficient,
)

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]


def find_time_dimension(data):
    """
    Find the time dimension in the dataset
    Different climate models use different names for time dimensions
    """
    possible_time_dims = ["time", "time_counter", "t", "T", "Time"]

    for dim in data.dims:
        if dim in possible_time_dims:
            return dim

    # If no exact match, look for anything with 'time' in the name
    for dim in data.dims:
        if "time" in dim.lower():
            return dim

    return None


import hashlib
import json


def get_cache_filename(var, source_gran, target_gran, cache_dir="./resampled_cache"):
    """Generate consistent cache filename"""
    cache_key = f"{var}_{source_gran}_to_{target_gran}"
    return os.path.join(cache_dir, f"{cache_key}.nc")


def get_file_hash(filepath):
    """Get hash of source file to detect changes"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read(1024 * 1024)).hexdigest()  # Hash first 1MB for speed


def load_resampled_from_cache(
    var, source_gran, target_gran, source_file, cache_dir="./resampled_cache"
):
    """Load resampled data from cache if available and valid"""
    cache_file = get_cache_filename(var, source_gran, target_gran, cache_dir)
    metadata_file = cache_file.replace(".nc", "_metadata.json")

    if not (os.path.exists(cache_file) and os.path.exists(metadata_file)):
        return None

    # Check if source file has changed
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        current_hash = get_file_hash(source_file)
        if metadata.get("source_hash") != current_hash:
            print(
                f"Source file changed, cache invalid for {var} {source_gran}→{target_gran}"
            )
            return None

        print(f"Loading {var} {source_gran}→{target_gran} from cache")
        return xr.open_dataarray(cache_file, chunks={"time": 100})

    except Exception as e:
        print(f"Cache load failed: {e}")
        return None


def save_resampled_to_cache(
    data, var, source_gran, target_gran, source_file, cache_dir="./resampled_cache"
):
    """Save resampled data to cache"""
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = get_cache_filename(var, source_gran, target_gran, cache_dir)
    metadata_file = cache_file.replace(".nc", "_metadata.json")

    try:
        # Save data
        print(f"Caching {var} {source_gran}→{target_gran} to {cache_file}")
        data.to_netcdf(cache_file)

        # Save metadata
        metadata = {
            "variable": var,
            "source_granularity": source_gran,
            "target_granularity": target_gran,
            "source_file": source_file,
            "source_hash": get_file_hash(source_file),
            "created": pd.Timestamp.now().isoformat(),
            "shape": list(data.shape),
            "chunks": str(data.chunks) if hasattr(data, "chunks") else None,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Cached {var} {source_gran}→{target_gran}")

    except Exception as e:
        print(f"Cache save failed: {e}")


def get_data_optimised_with_cache(
    var,
    granularity,
    variable_file_map,
    cache=None,
    allow_resampling=True,
    disk_cache_dir="./resampled_cache",
    save_to_cache=True,
):
    """
    Enhanced version with disk caching for resampled data
    """
    if cache is None:
        cache = {}

    key = (var, granularity)
    if key in cache:
        print("MEMORY CACHE HIT", key)
        return cache[key]

    # Check for direct file first (same as before)
    valid_entries = {}
    for entry in variable_file_map.get(var, []):
        gran = entry["granularity"]
        file_path = entry["file"]
        valid_entries[gran] = (file_path, os.path.exists(file_path))

    # Try direct file
    if granularity in valid_entries:
        file_path, exists = valid_entries[granularity]
        if exists:
            print(f"Loading {var}@{granularity} directly from {file_path}")
            ds = xr.open_dataset(file_path)
            print(f"Before standardization - dimensions: {list(ds.dims.keys())}")
            print(f"Before standardization - coordinates: {list(ds.coords.keys())}")
            ds = standardize_variables(ds, VARIABLE_ALIASES)

            print(f"After standardization - dimensions: {list(ds.dims.keys())}")
            print(f"After standardization - coordinates: {list(ds.coords.keys())}")

            # check if the extracted variable is standardised correctly:
            print(f"var: {var}")
            data_check = ds[var]
            print(f"DataArray type: {type(data_check)}")
            print(f"DataArray dimensions: {list(data_check.dims)}")  # No .keys() here
            print(f"DataArray coordinates: {list(data_check.coords.keys())}")  # .keys() is OK here
            print(f"DataArray shape: {data_check.shape}")

            # Check specific coordinates
            if 'depth' in data_check.coords:
                print(f"✓ Has 'depth' coordinate")
            else:
                print(f"✗ Missing 'depth' coordinate")
                
            if 'nav_lev' in data_check.dims:
                print(f"✗ Still has 'nav_lev' dimension")
            elif 'depth' in data_check.dims:
                print(f"✓ Has 'depth' dimension")
            
            if var in ds:
                data = ds[var]
                cache[key] = data
                return data
            else:
                print(f"Warning: Variable {var} not found in standardized dataset")
                print(f"Available variables: {list(ds.variables.keys())}")


    if not allow_resampling:
        raise ValueError(f"Cannot get {var} at {granularity}")

    # Find best resampling source
    available_grans = [gran for gran, (_, exists) in valid_entries.items() if exists]
    target_rank = GRANULARITY_ORDER.index(granularity)
    best_source_gran = next(
        (
            GRANULARITY_ORDER[rank]
            for rank in reversed(range(target_rank))
            if GRANULARITY_ORDER[rank] in available_grans
        ),
        None,
    )

    if best_source_gran:
        source_file = valid_entries[best_source_gran][0]


        # TRY DISK CACHE FIRST
        cached_resampled = load_resampled_from_cache(
            var, best_source_gran, granularity, source_file, disk_cache_dir
        )

        if cached_resampled is not None:
            cache[key] = cached_resampled
            return cached_resampled

        # Cache miss - compute and save
        print(f"Resampling {var}: {best_source_gran} → {granularity}")

        # Get source data (recursive call)
        finer_data = get_data_optimised_with_cache(
            var,
            best_source_gran,
            variable_file_map,
            cache,
            allow_resampling,
            disk_cache_dir,
            save_to_cache,
        )

        # Resample
        time_dim = find_time_dimension(finer_data)
        if time_dim is None:
            raise ValueError(
                f"No time dimension found in {var}. Dims: {finer_data.dims}"
            )

        freq_map = {"10d": "10D", "1m": "1ME", "3m": "3ME", "1y": "1YE"}
        resample_kwargs = {time_dim: freq_map[granularity]}
        resampled = finer_data.resample(**resample_kwargs).mean()

        # OPTIONALLY SAVE TO DISK CACHE
        if save_to_cache:
            # SAVE TO DISK CACHE (compute and save)
            save_resampled_to_cache(
                resampled.compute(),
                var,
                best_source_gran,
                granularity,
                source_file,
                disk_cache_dir,
            )

            # Reload as lazy for memory efficiency
            cached_data = load_resampled_from_cache(
                var, best_source_gran, granularity, source_file, disk_cache_dir
            )

            cache[key] = cached_data
            return cached_data
        else:
            print(f"  Not saving to cache (save_to_cache=False)")
            cache[key] = resampled  # Keep as lazy
            return resampled

    raise ValueError(f"Cannot get {var} at {granularity}")


def get_data_optimised(
    var, granularity, variable_file_map, cache=None, allow_resampling=True
):
    """
    Original version without disk caching (for backward compatibility)
    """
    return get_data_optimised_with_cache(
        var,
        granularity,
        variable_file_map,
        cache,
        allow_resampling,
        disk_cache_dir="./resampled_cache",
        save_to_cache=False,
    )


def run_metric_with_cache(
    metric_name,
    metric_function,
    required_vars,
    granularity,
    variable_file_map,
    cache=None,
    down_sample=True,
    disk_cache_dir="./resampled_cache",
    save_to_cache=True,
):
    """Enhanced run_metric with disk caching"""
    try:
        print(f"Running {metric_name} at {granularity}")
        inputs = [
            get_data_optimised_with_cache(
                var,
                granularity,
                variable_file_map,
                cache,
                down_sample,
                disk_cache_dir,
                save_to_cache,
            )
            for var in required_vars
        ]
        result = metric_function(*inputs)
        print(f"✓ Success: {metric_name}")
        return result
    except Exception as e:
        print(f"✗ Failed: {metric_name} - {e}")
        return None


def run_all_metrics_with_cache(
    metric_requirements,
    metric_functions,
    variable_file_map,
    granularities=None,
    down_sampling=True,
    disk_cache_dir="./resampled_cache",
    save_to_cache=True,
):
    """
    Enhanced run_all_metrics with disk caching for resampled data
    """
    if granularities is None:
        granularities = GRANULARITY_ORDER

    cache = {}  # Memory cache
    results = {}

    print(f"=== RUNNING ALL METRICS WITH DISK CACHING ===")
    print(f"Cache directory: {disk_cache_dir}")
    print(f"Save to cache: {save_to_cache}")
    print(f"Target granularities: {granularities}")

    for gran in granularities:
        print(f"\n=== GRANULARITY: {gran} ===")

        for metric_name, required_vars in metric_requirements.items():
            if metric_name in metric_functions:
                result = run_metric_with_cache(
                    metric_name,
                    metric_functions[metric_name],
                    required_vars,
                    gran,
                    variable_file_map,
                    cache,
                    down_sampling,
                    disk_cache_dir,
                    save_to_cache,
                )
                if result is not None:
                    results[(gran, metric_name)] = {
                        "result": result,
                        "granularity": gran,
                        "variables_used": required_vars,
                    }

    print(f"\n=== FINAL RESULTS ===")
    for metric_key, info in results.items():
        print(f"✓ {metric_key} computed at {info['granularity']}")

    # Show cache statistics
    if save_to_cache:
        show_cache_stats(disk_cache_dir)

    return results


def show_cache_stats(cache_dir="./resampled_cache"):
    """Show cache statistics"""
    if not os.path.exists(cache_dir):
        print("No disk cache found")
        return

    nc_files = [f for f in os.listdir(cache_dir) if f.endswith(".nc")]
    if not nc_files:
        print("Cache directory exists but is empty")
        return

    total_size = 0
    cached_items = []

    for nc_file in nc_files:
        file_path = os.path.join(cache_dir, nc_file)
        size = os.path.getsize(file_path)
        total_size += size

        # Parse filename: "temperature_10d_to_1m.nc"
        name_parts = nc_file[:-3].split("_to_")
        if len(name_parts) == 2:
            source_parts = name_parts[0].split("_")
            if len(source_parts) >= 2:
                var = "_".join(source_parts[:-1])
                source_gran = source_parts[-1]
                target_gran = name_parts[1]
                cached_items.append((var, source_gran, target_gran, size))

    print(f"\n=== DISK CACHE STATISTICS ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Total files: {len(nc_files)}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")

    if cached_items:
        print("\nCached resampled data:")
        for var, source_gran, target_gran, size in sorted(cached_items):
            print(f"  {var}: {source_gran}→{target_gran} ({size / 1024**2:.1f} MB)")
    else:
        print("No valid cached items found")


def run_metrics_intelligently_with_cache(
    metric_requirements,
    metric_functions,
    variable_file_map,
    disk_cache_dir="./resampled_cache",
    save_to_cache=True,
):
    """Your intelligent function with disk caching"""
    print("=== INTELLIGENT METRIC COMPUTATION WITH DISK CACHING ===")
    print(f"Save to cache: {save_to_cache}")

    analysis = analyze_what_is_possible_efficient(
        variable_file_map, metric_requirements
    )

    cache = {}
    results = {}

    for gran, runnable_metrics_list in analysis["runnable_metrics"].items():
        if not runnable_metrics_list:
            continue

        print(f"\n=== GRANULARITY: {gran} ===")

        for metric_name in runnable_metrics_list:
            if metric_name in metric_functions:
                required_vars = metric_requirements[metric_name]

                result = run_metric_with_cache(
                    metric_name,
                    metric_functions[metric_name],
                    required_vars,
                    gran,
                    variable_file_map,
                    cache,
                    down_sample=True,
                    disk_cache_dir=disk_cache_dir,
                    save_to_cache=save_to_cache,
                )

                if result is not None:
                    results[(gran, metric_name)] = {
                        "result": result,
                        "granularity": gran,
                        "variables_used": required_vars,
                    }

    return results, analysis


# Add this function to parallelize result computation
def compute_results_parallel(results, n_workers=1):
    """
    Compute all dask results in parallel
    """
    print(f"Computing {len(results)} results in parallel with {n_workers} workers...")

    # Extract all dask objects that need computing
    compute_tasks = []
    result_keys = []

    for key, info in results.items():
        result = info["result"]
        if hasattr(result, "compute"):
            compute_tasks.append(result)
            result_keys.append(key)

    if compute_tasks:
        # Compute all tasks in parallel
        # with dask.config.set(scheduler='threads', num_workers=n_workers):
        #     computed_results = dask.compute(*compute_tasks)

        # Update results with computed values
        for i, key in enumerate(result_keys):
            print("TO compute key ", key)
            if key[1] != "ACC_Drake_metric_2":
                continue
            breakpoint()
            results[key]["result"] = compute_tasks[i].compute()
            print(f"✓ Computed {key}")

    return results


def compute_results_parallel_fixed(results, n_workers=1):
    """
    Fixed compute function without dask dependency
    """
    print(f"Computing {len(results)} results...")

    for key, info in results.items():
        result = info["result"]
        if hasattr(result, "compute"):
            print(f"Computing {key}...")
            try:
                computed_result = result.compute()
                results[key]["result"] = computed_result
                print(f"✓ Computed {key}")
            except Exception as e:
                print(f"✗ Failed to compute {key}: {e}")
        else:
            print(f"✓ {key} already computed")

    return results


# 1. we can get the maximum granularity and run_all_metrics_with_cache
# 2. run_metrics_intelligently_with_cache
