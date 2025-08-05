import os
import xarray as xr
from standardise_variables import VARIABLE_ALIASES, standardize_variables

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]

def get_maximum_granularity_with_all(variable_file_map, metric_requirements):
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

def get_available_granularities(variable_file_map):
    all_grans = set()
    for entries in variable_file_map.values():
        for entry in entries:
            all_grans.add(entry["granularity"])
    return sorted(all_grans, key=GRANULARITY_ORDER.index)


def find_time_dimension(data):
    """Find the time dimension in the dataset"""
    possible_time_dims = ['time', 'time_counter', 't', 'T']
    for dim in data.dims:
        if dim in possible_time_dims:
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
            ds = xr.open_dataset(file_path, chunks={'time': 100, 'depth': 20})
            
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
        freq_map = {"10d": "10D", "1m": "1M", "3m": "3M", "1y": "1Y"}
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


import yaml

# Load your data map
with open("DINO_map.yaml", "r") as file:
    variable_file_map = yaml.safe_load(file)["variable_file_map"]

# Define what you want to compute
metric_requirements = {
    "check_density": ["temperature", "salinity"],
    "temperature_500m_30NS_metric": ["temperature"],
    "ACC_Drake_metric_2": ["velocity_u", "ssh"],
}

# Add dummy metric functions for testing
def dummy_check_density(temperature, salinity):
    """Dummy density calculation"""
    return temperature * 0.1 + salinity * 0.05

def dummy_temperature_500m_30NS_metric(temperature):
    """Dummy temperature metric"""
    return temperature.mean()

def dummy_ACC_Drake_metric_2(velocity_u, ssh):
    """Dummy ACC Drake metric"""
    return velocity_u * ssh

metric_functions = {
    "check_density": dummy_check_density,
    "temperature_500m_30NS_metric": dummy_temperature_500m_30NS_metric,
    "ACC_Drake_metric_2": dummy_ACC_Drake_metric_2,
}

# Run everything!
results = run_all_metrics(metric_requirements, metric_functions, variable_file_map, granularities=None, down_sample=True)

print(results.keys())

# get maximum granularities
gran_max = get_maximum_granularity_with_all(variable_file_map, metric_requirements)
print(gran_max)

gran_available = get_available_granularities(variable_file_map)

print(gran_available)