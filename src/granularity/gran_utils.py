import os
import pandas as pd
import xarray as xr

from standardise_variables import VARIABLE_ALIASES, standardize_variables
from granularity.cache import show_cache_stats, get_cache_filename, get_file_hash

# from fix_timing import find_time_dimension, resample_to_reference_bins
from granularity.fix_timing import (
    ensure_time,
    same_time_axis,
    resample_like,
    find_time_dimension,
)
from granularity.gran_analysis import (
    analyze_metric_requirements,
)

from pathlib import Path

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]


import hashlib
import json

# USED (1) - called from get_data
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

# USED (1) - called from get_data
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


# USED (1) (mainly) and (2) - need to check how this affects (2)
def load_aligned_from_cache(var, target_gran, cache_dir="./resampled_cache"):
    import glob, xarray as xr, os
    pattern = os.path.join(cache_dir, f"{var}_ALIGNED_to_{target_gran}_*.nc")
    hits = sorted(glob.glob(pattern))
    if hits:
        print(f"Loading {var} ALIGNED→{target_gran} from cache")
        return xr.open_dataarray(hits[-1], chunks={"time": 100})
    return None

# USED (1) and (2) public interface
def get_data(
    var,
    granularity,
    variable_file_map,
    cache,
    analysis,
    allow_resampling=True,
    disk_cache_dir="./resampled_cache",
    save_to_cache=True,
):
    """
    Retrieve data for a variable at a given granularity, using memory and disk cache,
    and resampling from the closest available finer granularity if needed.
    """
    key = (var, granularity)
    if key in cache:
        print("MEMORY CACHE HIT", key)
        return cache[key]

    # NEW: aligned artifact takes precedence (skips re-binning later)
    aligned = load_aligned_from_cache(var, granularity, disk_cache_dir)
    if aligned is not None:
        cache[key] = aligned
        return aligned

    # 1. Try direct file first
    if granularity in analysis["direct_files"].get(var, []):
        file_entry = next(
            e for e in variable_file_map[var] if e["granularity"] == granularity
        )
        print(f"Loading {var}@{granularity} directly from {file_entry['file']}")
        ds = xr.open_dataset(file_entry["file"])
        ds = standardize_variables(ds, VARIABLE_ALIASES)
        if var in ds:
            data = ds[var]
            cache[key] = data
            return data
        else:
            raise ValueError(f"Variable {var} not found in {file_entry['file']}")

    # 2. Otherwise, resample from closest available finer granularity
    if allow_resampling:
        gran_idx = GRANULARITY_ORDER.index(granularity)
        finer_grans = [
            g
            for g in analysis["direct_files"].get(var, [])
            if GRANULARITY_ORDER.index(g) < gran_idx
        ]
        if not finer_grans:
            raise ValueError(
                f"No finer granularity available for {var} to resample to {granularity}"
            )
        # Pick the closest (highest index < gran_idx)
        # source_gran = max(finer_grans, key=lambda g: GRANULARITY_ORDER.index(g))

        source_gran = min(finer_grans, key=lambda g: GRANULARITY_ORDER.index(g))
        source_file = next(
            e["file"] for e in variable_file_map[var] if e["granularity"] == source_gran
        )

        # Try disk cache for resampled data
        cached_resampled = load_resampled_from_cache(
            var, source_gran, granularity, source_file, disk_cache_dir
        )
        if cached_resampled is not None:
            cache[key] = cached_resampled
            return cached_resampled

        print(f"Resampling {var}: {source_gran} → {granularity}")

        # Get source data (recursive call)
        finer_data = get_data(
            var,
            source_gran,
            variable_file_map,
            cache,
            analysis,
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

        # Use fallback_freq for resampling frequency string
        freq_map = {"10d": "10D", "1m": "1ME", "3m": "3ME", "1y": "1YE"}
        resample_kwargs = {time_dim: freq_map[granularity]}
        resampled = finer_data.resample(
            **resample_kwargs, label="right", closed="right"
        ).mean(skipna=True)

        # Optionally save to disk cache
        if save_to_cache:
            save_resampled_to_cache(
                resampled.compute(),
                var,
                source_gran,
                granularity,
                source_file,
                disk_cache_dir,
            )
            # Reload as lazy for memory efficiency
            cached_data = load_resampled_from_cache(
                var, source_gran, granularity, source_file, disk_cache_dir
            )
            cache[key] = cached_data
            return cached_data
        else:
            print(f"  Not saving to cache (save_to_cache=False)")
            cache[key] = resampled  # Keep as lazy
            return resampled

    raise ValueError(f"{var} not available at {granularity} and resampling not allowed")

# Maybe USED (1) - called from run_all_metrics_with_cache and run_metrics_intelligently_with_cache
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
            get_data(
                var,
                granularity,
                variable_file_map,
                cache,
                analysis,
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

# UNUSED
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
    Enhanced run_all_metrics with disk caching for resampled data.

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


# UNUSED and probably doesn't work - need to align variables at each granularity. 
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

    analysis = analyze_metric_requirements(variable_file_map, metric_requirements)

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
# UNUSED (we decided to preload first)
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
            results[key]["result"] = compute_tasks[i].compute()
            print(f"✓ Computed {key}")

    return results

# UNUSED 
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


# USED (1) - called from preload_and_align_all_variables
def load_all_variables_at_granularity(
    gran, variable_file_map, analysis, cache, disk_cache_dir, save_to_cache
):
    """
    Loads all variables available at a given granularity (direct or resampled).
    Returns a dict: {var: xarray.DataArray}
    """
    available_vars = [
        var for var, grans in analysis["variable_availability"].items() if gran in grans
    ]
    loaded_vars = {}
    for var in available_vars:
        try:
            data = get_data(
                var,
                gran,
                variable_file_map,
                cache,
                analysis=analysis,
                allow_resampling=True,
                disk_cache_dir=disk_cache_dir,
                save_to_cache=save_to_cache,
            )
            loaded_vars[var] = data
        except Exception as e:
            print(f"✗ Failed to load {var}@{gran}: {e}")
    return loaded_vars


# USED (1) - called from preload_and_align_all_variables
def align_all_to_reference(
    loaded_vars: dict[str, xr.DataArray],
    ref_var: str,
    fallback_freq: str | None = None,
) -> dict[str, xr.DataArray]:
    """
    For each variable, return a version whose time axis exactly matches loaded_vars[ref_var].time.
    Only resamples when needed; otherwise returns the data unchanged.
    """
    ref = ensure_time(loaded_vars[ref_var])
    aligned = {}
    for name, da in loaded_vars.items():
        da = ensure_time(da)
        if same_time_axis(da, ref):
            aligned[name] = da
        else:
            aligned[name] = resample_like(da, ref, fallback_freq=fallback_freq)
    return aligned

# USED (1)
def _time_hash(da: xr.DataArray) -> str:
    t = ensure_time(da)["time"].values
    return hashlib.sha1("".join(map(str, t)).encode()).hexdigest()[:16]

# USED (1) 
def _save_aligned_to_cache(
    data: xr.DataArray,
    var: str,
    target_gran: str,
    ref_var: str,
    cache_dir: str = "./resampled_cache",
) -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    th = _time_hash(data)
    fname = f"{var}_ALIGNED_to_{target_gran}_{th}.nc"
    fpath = os.path.join(cache_dir, fname)
    ensure_time(data).to_netcdf(fpath)

    meta = {
        "variable": var,
        "target_granularity": target_gran,
        "aligned": True,
        "reference_variable": ref_var,
        "time_hash": th,
        "calendar": ensure_time(data)["time"].attrs.get("calendar"),
        "units": ensure_time(data)["time"].attrs.get("units"),
    }
    with open(fpath.replace(".nc", "_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return fpath

# USED (1) - public interface
def preload_and_align_all_variables(
    variable_file_map,
    granularities,
    analysis,
    disk_cache_dir="./resampled_cache",
    save_to_cache=False,
    persist_aligned=False,
):
    cache = {}
    aligned_vars_by_gran = {}

    for gran in granularities:
        print(f"\n=== GRANULARITY: {gran} ===")
        loaded_vars = load_all_variables_at_granularity(
            gran, variable_file_map, analysis, cache, disk_cache_dir, save_to_cache
        )

        direct_vars = analysis["direct_from_file_by_granularity"].get(gran, [])
        ref_var = next((var for var in direct_vars if var in loaded_vars), None)
        if ref_var is None:
            print(f"No reference variable for {gran}")
            aligned_vars_by_gran[gran] = loaded_vars
            continue
        print(f"Using '{ref_var}' as reference for alignment at {gran}")

        # Get calendar and units from the reference variable's time coordinate
        # ref_data = loaded_vars[ref_var]
        # time_dim = find_time_dimension(ref_data)
        # calendar = ref_data[time_dim].attrs.get("calendar", "360_day")
        # units = ref_data[time_dim].attrs.get("units", "days since 0001-01-01")

        # Align all variables to the reference bins
        # aligned_vars = align_all_variables_to_reference_bins(
        #     loaded_vars, ref_var, calendar, units, time_dim=time_dim, fallback_freq=gran
        # )

        aligned_vars = align_all_to_reference(loaded_vars, ref_var, fallback_freq=gran)

        if persist_aligned:  # new flag you add
            for var, da in aligned_vars.items():
                _save_aligned_to_cache(da, var, gran, ref_var, cache_dir=disk_cache_dir)

        for var, data in aligned_vars.items():
            cache[(var, gran)] = data

    return cache
