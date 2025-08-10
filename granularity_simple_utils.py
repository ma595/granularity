import os
from pathlib import Path
import pandas as pd
import xarray as xr

from standardise_variables import VARIABLE_ALIASES, standardize_variables

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]

def get_maximum_granularity_with_all(variable_file_map, metric_requirements, GRANULARITY_ORDER):
    """
    Find the maximum granularity that is available in the variable_file_map.
    """
    all_grans = get_available_granularities(variable_file_map)
    all_vars = set()
    for fn, vars_list in metric_requirements.items():
        # print(vars_list)
        for var in vars_list:
            all_vars.add(var)
    
    cache = {}

    for gran in all_grans:
        for var in all_vars:
            if (var, gran) in cache:
                continue

            # this finds if var at that granularity exists in variable_file_map
            entry = variable_file_map.get(var, [])
            for e in entry:
                # print(e)
                if e["granularity"] == gran:
                    file_path = e['file']
                    if os.path.exists(file_path):
                        cache[(var, gran)] = True
                        break

            # Now resample backwards
            grans = GRANULARITY_ORDER[:GRANULARITY_ORDER.index(gran)]
            # print(grans)

            for finer_gran in reversed(grans):
                if (var, finer_gran) in cache:
                    # print("Can be resampled")
                    cache[(var, gran)] = True

    gran_fn_map = {} #  "gran" : [fn names]

    for gran in all_grans:
        for fn, vars in metric_requirements.items():
            # print(fn, vars)
            if gran not in gran_fn_map:
                gran_fn_map[gran] = []

            if all((v,gran) in cache for v in vars ):
                gran_fn_map[gran].append(fn)

    counts = [len(gran_fn_map[gran]) for gran in gran_fn_map]
    max_index = counts.index(max(counts))
    return all_grans[max_index]

def find_time_dimension(data):
    """
    Find the time dimension in the dataset
    Different climate models use different names for time dimensions
    """
    possible_time_dims = ['time', 'time_counter', 't', 'T', 'Time']
    
    for dim in data.dims:
        if dim in possible_time_dims:
            return dim
    
    # If no exact match, look for anything with 'time' in the name
    for dim in data.dims:
        if 'time' in dim.lower():
            return dim
    
    return None

def get_data_optimised(var, granularity, variable_file_map, cache=None, allow_resampling=True):
    """
    Fully optimized version with consistent lazy loading and chunking
    """
    if cache is None:
        cache = {}
    
    key = (var, granularity)
    if key in cache:
        print("CACHE HIT", key)
        return cache[key]
    
    # Single-pass file validation
    valid_entries = {}
    for entry in variable_file_map.get(var, []):
        gran = entry["granularity"]
        file_path = entry["file"]
        valid_entries[gran] = (file_path, os.path.exists(file_path))
    
    # Try direct file first
    if granularity in valid_entries:
        file_path, exists = valid_entries[granularity]
        if exists:
            print(f"Loading {var}@{granularity} directly from {file_path}")
            
            # Open with chunking for better performance
            # ds = xr.open_dataset(file_path, chunks={'time': 100, 'depth': 20})
            ds = xr.open_dataset(file_path)
            # breakpoint()
            
            # Optimized variable lookup
            actual_var = var if var in ds else next(
                (alias for alias in VARIABLE_ALIASES.get(var, []) if alias in ds), None
            )
            
            if actual_var:
                data = ds[actual_var]  # Keep lazy!
                cache[key] = data
                return data
            else:
                print(f"Variable '{var}' not found. Available: {list(ds.data_vars.keys())}")
    
    if not allow_resampling:
        raise ValueError(f"Cannot get {var} at {granularity} - no direct file and resampling disabled")
    
    # Find best resampling source
    available_grans = [gran for gran, (_, exists) in valid_entries.items() if exists]
    
    if not available_grans:
        raise ValueError(f"No valid files found for {var}")
    
    print(f"Available granularities for {var}: {available_grans}")
    
    # Direct lookup for best source
    target_rank = GRANULARITY_ORDER.index(granularity)
    best_source_gran = next(
        (GRANULARITY_ORDER[rank] for rank in reversed(range(target_rank))
         if GRANULARITY_ORDER[rank] in available_grans), None
    )
    
    if best_source_gran:
        print(f"Downsampling {var}: {best_source_gran} → {granularity}")
        
        # Recursive call (already optimized with cache check)
        finer_data = get_data_optimised(var, best_source_gran, variable_file_map, cache, allow_resampling)
        
        # Find time dimension
        time_dim = find_time_dimension(finer_data)
        if time_dim is None:
            raise ValueError(f"No time dimension found in {var}. Dims: {finer_data.dims}")
        
        print(f"  Using time dimension: '{time_dim}'")
        
        # Resample (keep lazy!)
        freq_map = {"10d": "10D", "1m": "1ME", "3m": "3ME", "1y": "1YE"}
        resampled = finer_data.resample(**{time_dim: freq_map[granularity]}).mean()
        
        # Don't force loading here - let the metric function decide when to load
        cache[key] = resampled
        return resampled
    
    raise ValueError(f"Cannot get {var} at {granularity} - no finer data available")


def run_metric(metric_name, metric_function, required_vars, granularity, variable_file_map, cache=None, down_sample=True):
    """
    Simple function: run one metric at one granularity.
    """
    try:
        print(f"Running {metric_name} at {granularity}")
        inputs = [get_data_optimised(var, granularity, variable_file_map, cache, down_sample) for var in required_vars]
        result = metric_function(*inputs)
        print(f"✓ Success: {metric_name}")
        return result
    except Exception as e:
        print(f"✗ Failed: {metric_name} - {e}")
        return None

def run_all_metrics(metric_requirements, metric_functions, variable_file_map, granularities=None, down_sample=True):
    """
    Simple function: try to run all metrics at all available granularities.
    """
    if granularities is None:
        granularities = GRANULARITY_ORDER
    
    cache = {}
    results = {}
    
    for gran in granularities:
        print(f"\n=== GRANULARITY: {gran} ===")
        
        for metric_name, required_vars in metric_requirements.items():
            # if metric_name not in results:  # Only if not already computed
            if metric_name in metric_functions:
                result = run_metric(
                    metric_name, 
                    metric_functions[metric_name], 
                    required_vars, 
                    gran, 
                    variable_file_map, 
                    cache,
                    down_sample
                )
                if result is not None:
                    results[(gran, metric_name)] = {'result': result, 'granularity': gran}
    
    print(f"\n=== FINAL RESULTS ===")
    for metric_name, info in results.items():
        print(f"✓ {metric_name} computed at {info['granularity']}")
    
    return results

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
        result = info['result']
        if hasattr(result, 'compute'):
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
            results[key]['result'] = compute_tasks[i].compute()
            print(f"✓ Computed {key}")
    
    return results


def get_available_granularities(variable_file_map):
    all_grans = set()
    for entries in variable_file_map.values():
        for entry in entries:
            all_grans.add(entry["granularity"])
    return sorted(all_grans, key=GRANULARITY_ORDER.index)

def analyze_what_is_possible_efficient(variable_file_map, metric_requirements):
    """
    Efficient analysis without loading data - just check files and resampling logic
    """
    available_grans = get_available_granularities(variable_file_map)
    
    # Get all variables needed
    all_vars = set()
    for vars_list in metric_requirements.values():
        all_vars.update(vars_list)
    
    variable_availability = {}
    
    for var in all_vars:
        variable_availability[var] = []
        
        # Check direct file availability (no data loading)
        direct_available = []
        for entry in variable_file_map.get(var, []):
            if os.path.exists(entry["file"]):
                direct_available.append(entry["granularity"])
        
        print(f"{var}: direct files at {direct_available}")
        
        # For each target granularity, check if achievable
        for target_gran in available_grans:
            # Direct file available?
            if target_gran in direct_available:
                variable_availability[var].append(target_gran)
                print(f"{var}: achievable at {target_gran} (direct)")
                continue
            
            # Can we resample from a finer granularity?
            target_rank = GRANULARITY_ORDER.index(target_gran)
            can_resample = any(
                GRANULARITY_ORDER.index(direct_gran) < target_rank 
                for direct_gran in direct_available
            )
            
            if can_resample:
                variable_availability[var].append(target_gran)
                print(f"{var}: achievable at {target_gran} (via resampling)")
            else:
                print(f"{var}: NOT achievable at {target_gran}")
    
    # Determine runnable metrics per granularity
    runnable_metrics = {}
    for gran in available_grans:
        runnable_metrics[gran] = []
        
        for metric_name, required_vars in metric_requirements.items():
            if all(gran in variable_availability.get(var, []) for var in required_vars):
                runnable_metrics[gran].append(metric_name)
    
    return {
        'variable_availability': variable_availability,
        'runnable_metrics': runnable_metrics,
        'available_granularities': available_grans
    }

def run_metrics_intelligently_fixed(metric_requirements, metric_functions, variable_file_map):
    """
    Fixed intelligent version with efficient analysis
    """
    print("=== INTELLIGENT METRIC COMPUTATION ===")
    
    # Step 1: Efficient analysis (no data loading)
    analysis = analyze_what_is_possible_efficient(variable_file_map, metric_requirements)
    
    print(f"\n=== ANALYSIS RESULTS ===")
    for gran, metrics in analysis['runnable_metrics'].items():
        if metrics:
            print(f"At {gran}: can run {metrics}")
    
    cache = {}
    results = {}
    
    # Step 2: Run metrics only at their optimal granularities
    for gran, runnable_metrics_list in analysis['runnable_metrics'].items():
        if not runnable_metrics_list:
            continue
            
        print(f"\n=== GRANULARITY: {gran} ===")
        print(f"Can run: {runnable_metrics_list}")
        
        for metric_name in runnable_metrics_list:
            if metric_name in metric_functions:
                required_vars = metric_requirements[metric_name]
                
                result = run_metric(
                    metric_name, 
                    metric_functions[metric_name], 
                    required_vars, 
                    gran, 
                    variable_file_map, 
                    cache,
                    down_sample=True
                )
                
                if result is not None:
                    results[(gran, metric_name)] = {
                        'result': result, 
                        'granularity': gran,
                        'variables_used': required_vars
                    }
    
    return results, analysis

def compute_results_parallel_fixed(results, n_workers=1):
    """
    Fixed compute function without dask dependency
    """
    print(f"Computing {len(results)} results...")
    
    for key, info in results.items():
        result = info['result']
        if hasattr(result, 'compute'):
            print(f"Computing {key}...")
            try:
                computed_result = result.compute()
                results[key]['result'] = computed_result
                print(f"✓ Computed {key}")
            except Exception as e:
                print(f"✗ Failed to compute {key}: {e}")
        else:
            print(f"✓ {key} already computed")
    
    return results