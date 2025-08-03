import os
import xarray as xr
from standardise_variables import VARIABLE_ALIASES, standardize_variables

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]

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

def get_data(var, granularity, variable_file_map, cache=None):
    """
    Simple function: get data for a variable at a granularity.
    Either load directly or resample from ANY available data.
    """
    if cache is None:
        cache = {}
    
    key = (var, granularity)
    if key in cache:
        return cache[key]
    
    # Try direct file first (with file existence check)
    for entry in variable_file_map.get(var, []):
        if entry["granularity"] == granularity:
            file_path = entry["file"]
            if os.path.exists(file_path):
                print(f"Loading {var}@{granularity} directly from {file_path}")
                # Try to find the variable by checking aliases manually
                actual_var = None
                ds = xr.open_dataset(file_path)
                if var in ds:
                    actual_var = var
                else:
                    aliases = VARIABLE_ALIASES.get(var, [])
                    for alias in aliases:
                        if alias in ds:
                            actual_var = alias
                            break
                
                if actual_var is None:
                    print(f"⚠️  Variable '{var}' not found. Available: {list(ds.data_vars.keys())}")
                    continue

                data = ds[actual_var]
                cache[key] = data
                return data
            else:
                print(f"⚠️  Direct file missing: {file_path}")
    
    # Direct file doesn't exist or granularity not available
    # Find ANY available data to resample from
    print(f"Direct {var}@{granularity} not available, looking for alternatives...")
    
    available_grans = []
    for entry in variable_file_map.get(var, []):
        if os.path.exists(entry["file"]):
            available_grans.append(entry["granularity"])
    
    if not available_grans:
        raise ValueError(f"No valid files found for {var}")
    
    print(f"Available granularities for {var}: {available_grans}")
    
    # Try to resample from finer granularities first
    target_rank = GRANULARITY_ORDER.index(granularity)
    
    # Look for finer granularities
    for finer_rank in reversed(range(target_rank)):
        finer_gran = GRANULARITY_ORDER[finer_rank]
        if finer_gran in available_grans:
            print(f"Resampling {var}: {finer_gran} → {granularity}")
            finer_data = get_data(var, finer_gran, variable_file_map, cache)
            
            # Find the time dimension
            time_dim = find_time_dimension(finer_data)
            if time_dim is None:
                print(f"⚠️  No time dimension found in {var}. Dims: {finer_data.dims}")
                continue
            
            print(f"  Using time dimension: '{time_dim}'")
            
            freq_map = {"10d": "10D", "1m": "1M", "3m": "3M", "1y": "1Y"}
            
            # Use the correct time dimension for resampling
            resample_kwargs = {time_dim: freq_map[granularity]}
            resampled = finer_data.resample(**resample_kwargs).mean()
            
            cache[key] = resampled
            return resampled
    
    # If no finer granularities, try coarser ones (less ideal but better than nothing)
    for coarser_rank in range(target_rank + 1, len(GRANULARITY_ORDER)):
        coarser_gran = GRANULARITY_ORDER[coarser_rank]
        if coarser_gran in available_grans:
            print(f"⚠️  Using coarser data: {var}@{coarser_gran} for {granularity}")
            coarser_data = get_data(var, coarser_gran, variable_file_map, cache)
            # Note: This doesn't actually increase temporal resolution, just returns the coarser data
            cache[key] = coarser_data
            return coarser_data
    
    raise ValueError(f"Cannot get {var} at {granularity} - no alternative sources")

def run_metric(metric_name, metric_function, required_vars, granularity, variable_file_map, cache=None):
    """
    Simple function: run one metric at one granularity.
    """
    try:
        print(f"Running {metric_name} at {granularity}")
        inputs = [get_data(var, granularity, variable_file_map, cache) for var in required_vars]
        result = metric_function(*inputs)
        print(f"✓ Success: {metric_name}")
        return result
    except Exception as e:
        print(f"✗ Failed: {metric_name} - {e}")
        return None

def run_all_metrics(metric_requirements, metric_functions, variable_file_map, granularities=None):
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
            if metric_name not in results:  # Only if not already computed
                if metric_name in metric_functions:
                    result = run_metric(
                        metric_name, 
                        metric_functions[metric_name], 
                        required_vars, 
                        gran, 
                        variable_file_map, 
                        cache
                    )
                    if result is not None:
                        results[metric_name] = {'result': result, 'granularity': gran}
    
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
results = run_all_metrics(metric_requirements, metric_functions, variable_file_map, granularities=["1y"])

print(results)