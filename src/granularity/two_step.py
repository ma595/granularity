from pathlib import Path
import glob
import os
import xarray as xr

from granularity.fix_timing import ensure_time, same_time_axis
from granularity.cache import get_cache_filename, write_metadata
from granularity.gran_utils import get_data, align_all_to_reference

# USED (2): Step 2 of 2. Called by run_metrics_from_materialized
def open_materialized(var, target_gran, disk_cache_dir="./resampled_cache"):
    """
    Open a previously materialized artifact var_*_to_{target_gran}.nc.
    Raises if not found.
    """
    import glob

    pattern = os.path.join(disk_cache_dir, f"{var}_*_to_{target_gran}.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No materialized cache for {var}@{target_gran} in {disk_cache_dir}"
        )
    # Use the newest if multiple
    path = hits[-1]
    da = xr.open_dataarray(path, chunks={"time": 100})
    return da

# USED (2): Step 1 of 2. Called by two_step_resample_all_aligned
def materialize_resamples_aligned(
    variable_file_map: dict,
    analysis: dict,
    gran: str,
    disk_cache_dir: str = "./resampled_cache",
    prefer_ref_vars: tuple[str, ...] = (
        "tn",
        "sshn",
        "thetao",
    ),  # tweak to your favorites
):
    """
    Materialize *aligned* artifacts for all variables available at `gran`.
    Every written file will have the exact same time axis.
    """
    Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)

    # Variables we plan to materialize at this granularity
    vars_at_gran = analysis["granularity_to_variables"].get(gran, [])
    if not vars_at_gran:
        print(f"[align materialize] No variables at {gran}")
        return []

    # Choose a reference variable:
    direct_vars = analysis["direct_from_file_by_granularity"].get(gran, [])
    ref_var = next((v for v in prefer_ref_vars if v in direct_vars), None)
    if ref_var is None:
        ref_var = direct_vars[0] if direct_vars else vars_at_gran[0]
    print(f"[align materialize] Using '{ref_var}' as reference for {gran}")

    # Load all variables lazily (no write yet)
    cache = {}
    loaded = {}
    for v in vars_at_gran:
        try:
            da = get_data(
                var=v,
                granularity=gran,
                variable_file_map=variable_file_map,
                cache=cache,
                analysis=analysis,
                allow_resampling=True,
                disk_cache_dir=disk_cache_dir,
                save_to_cache=False,  # <-- IMPORTANT: don't write yet
            )
            loaded[v] = da
        except Exception as e:
            print(f"[align materialize] skip {v}@{gran}: {e}")

    if ref_var not in loaded:
        print(
            f"[align materialize] Reference '{ref_var}' failed to load; aborting for {gran}"
        )
        return []

    # Align everyone to the reference axis (uses your resample_to_reference_bins under the hood)
    aligned = align_all_to_reference(loaded, ref_var=ref_var, fallback_freq=gran)

    # Persist aligned results into the cache
    written = []
    ref = ensure_time(aligned[ref_var])
    ref_time = ref["time"].values
    cal = ref["time"].attrs.get("calendar", "360_day")
    units = ref["time"].attrs.get("units", "days since 0001-01-01")

    for v, da in aligned.items():
        da = ensure_time(da)
        # sanity: enforce equality
        if not same_time_axis(da, ref):
            print(f"[align materialize] {v}@{gran} still misaligned; skipping write")
            continue

        cache_file = get_cache_filename(v, "aligned", gran, cache_dir=disk_cache_dir)
        da.to_netcdf(cache_file)

        meta = {
            "variable": v,
            "source_granularity": "aligned",
            "target_granularity": gran,
            "calendar": cal,
            "units": units,
            "reference_variable": ref_var,
            "time_len": int(da.sizes["time"]),
            "note": "Aligned to reference axis during materialization",
        }
        write_metadata(cache_file, meta)
        written.append(cache_file)
        print(f"[align materialize] wrote {os.path.basename(cache_file)}")

    return written

# NOT USED 
def materialize_resamples(
    variable_file_map,
    analysis,
    targets=("10d", "1m", "3m", "1y"),
    variables=None,
    disk_cache_dir="./resampled_cache",
):
    """
    Step 1: Precompute & store resampled intermediates for (variables x targets).
    Uses your disk cache format (*.nc + *_metadata.json).

    Returns a list of created or existing cache files.
    """
    if variables is None:
        # All variables known to the map (excluding mesh_mask etc., if any)
        variables = [k for k in variable_file_map.keys() if k != "mesh_mask"]

    Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)
    created_or_existing = []

    # Warm a small memory cache to avoid re-opens
    mem_cache = {}

    for var in variables:
        # What direct granularities exist for this var?
        direct_grans = [
            e["granularity"]
            for e in variable_file_map.get(var, [])
            if os.path.exists(e["file"])
        ]
        if not direct_grans:
            print(f"[materialize] Skip {var}: no direct source files")
            continue

        for tgt in targets:
            try:
                # This call *resamples if needed* and, with save_to_cache=True, writes the artifact.
                _ = get_data(
                    var=var,
                    granularity=tgt,
                    variable_file_map=variable_file_map,
                    cache=mem_cache,
                    analysis=analysis,
                    allow_resampling=True,
                    disk_cache_dir=disk_cache_dir,
                    save_to_cache=True,
                )
                # find what we just wrote (var_*_to_tgt.nc)
                pattern = os.path.join(disk_cache_dir, f"{var}_*_to_{tgt}.nc")
                hits = sorted(glob.glob(pattern))
                created_or_existing.extend(hits)
                if hits:
                    print(
                        f"[materialize] {var}@{tgt}: ready ({os.path.basename(hits[-1])})"
                    )
            except Exception as e:
                print(f"[materialize] {var}@{tgt}: not materialized ({e})")

    return created_or_existing


# USED (2) : Step 2 of 2. Called by two_step_metrics_all
def run_metrics_from_materialized(
    metric_requirements,
    metric_functions,
    target_gran,
    disk_cache_dir="./resampled_cache",
):
    """
    Step 2: Compute metrics using only materialized data.
    No on-the-fly resampling is allowed.
    """
    results = {}

    for metric_name, required_vars in metric_requirements.items():
        if metric_name not in metric_functions:
            continue

        try:
            # Load all inputs from materialized artifacts
            inputs = [
                open_materialized(v, target_gran, disk_cache_dir) for v in required_vars
            ]

            # Align on time (inner join) before computing
            aligned = xr.align(*inputs, join="inner")

            out = metric_functions[metric_name](*aligned)
            results[(target_gran, metric_name)] = {
                "result": out,
                "granularity": target_gran,
                "variables_used": required_vars,
            }
            print(f"✓ {metric_name}@{target_gran}")
        except Exception as e:
            print(f"✗ {metric_name}@{target_gran}: {e}")

    return results


# USED (2): Step 1 of 2.
def two_step_resample_all_aligned(
    variable_file_map,
    analysis,
    targets=None,
    disk_cache_dir="./resampled_cache",
):
    if targets is None:
        targets = analysis["available_granularities"]
    for gran in targets:
        print(f"\n[Two-step aligned] Materializing aligned artifacts at {gran}")
        materialize_resamples_aligned(
            variable_file_map=variable_file_map,
            analysis=analysis,
            gran=gran,
            disk_cache_dir=disk_cache_dir,
        )


# USED (2): Step 2 of 2.
def two_step_metrics_all(
    metric_requirements,
    metric_functions,
    targets,
    disk_cache_dir="./resampled_cache",
):
    """
    Run metrics for each granularity exclusively from materialized artifacts.
    """
    all_results = {}
    for gran in targets:
        print(f"\n[Two-step] Metrics at {gran}")
        res = run_metrics_from_materialized(
            metric_requirements=metric_requirements,
            metric_functions=metric_functions,
            target_gran=gran,
            disk_cache_dir=disk_cache_dir,
        )
        all_results.update(res)

    # We currently get data in this shape
    # ('10d', 'check_density') -> {result, granularity, variables_used}
    # we want
    # ['10d'] -> { 'check_density': result, 'temperature_500m_30NS_metric : result, ...}

    results_by_granularity = {}
    for (gran, func), result_meta in all_results.items():
        if gran not in results_by_granularity:
            results_by_granularity[gran] = {}
        results_by_granularity[gran][func] = result_meta["result"]

    return results_by_granularity
