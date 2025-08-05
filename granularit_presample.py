# this was generated from Claude inspired by granularity_simple.py

import os
import xarray as xr
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from standardise_variables import VARIABLE_ALIASES, standardize_variables

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]

def create_all_resampled_files(variable_file_map, output_dir="resampled_data", target_granularities=None):
    """
    Pre-compute all possible resampled files
    """
    if target_granularities is None:
        target_granularities = ["1m", "3m", "1y"]  # Don't resample to 10d (it's usually the source)
    
    os.makedirs(output_dir, exist_ok=True)
    created_files = {}
    
    print("=== PRE-COMPUTING ALL RESAMPLED FILES ===")
    
    for var, entries in variable_file_map.items():
        print(f"\n--- Processing {var} ---")
        
        # Find all available source granularities
        available_sources = []
        for entry in entries:
            gran = entry["granularity"]
            file_path = entry["file"]
            if os.path.exists(file_path):
                available_sources.append((gran, file_path))
        
        if not available_sources:
            print(f"  No files found for {var}")
            continue
        
        # Sort by fineness (10d first)
        available_sources.sort(key=lambda x: GRANULARITY_ORDER.index(x[0]))
        
        created_files[var] = []
        
        # For each target granularity
        for target_gran in target_granularities:
            target_rank = GRANULARITY_ORDER.index(target_gran)
            
            # Find best source (finest available that's finer than target)
            best_source = None
            for source_gran, source_file in available_sources:
                source_rank = GRANULARITY_ORDER.index(source_gran)
                if source_rank < target_rank:  # Source is finer than target
                    best_source = (source_gran, source_file)
                    break
            
            if not best_source:
                print(f"  ✗ {target_gran}: No finer source available")
                continue
            
            source_gran, source_file = best_source
            output_file = f"{output_dir}/{var}_{target_gran}.nc"
            
            if os.path.exists(output_file):
                print(f"  ✓ {target_gran}: Already exists")
                created_files[var].append({"granularity": target_gran, "file": output_file})
                continue
            
            try:
                print(f"  🔄 {target_gran}: Resampling from {source_gran}...")
                
                # Load source data
                ds = xr.open_dataset(source_file, chunks={'time': 100})
                
                # Find actual variable name
                actual_var = var if var in ds else next(
                    (alias for alias in VARIABLE_ALIASES.get(var, []) if alias in ds), None
                )
                
                if not actual_var:
                    print(f"    ✗ Variable {var} not found in {source_file}")
                    continue
                
                source_data = ds[actual_var]
                
                # Find time dimension
                time_dim = find_time_dimension(source_data)
                if not time_dim:
                    print(f"    ✗ No time dimension found")
                    continue
                
                # Resample
                freq_map = {"1m": "1M", "3m": "3M", "1y": "1Y"}
                resampled = source_data.resample(**{time_dim: freq_map[target_gran]}).mean()
                
                # Save
                resampled_ds = resampled.to_dataset(name=actual_var)
                resampled_ds.to_netcdf(
                    output_file,
                    encoding={actual_var: {'zlib': True, 'complevel': 4}}
                )
                
                created_files[var].append({"granularity": target_gran, "file": output_file})
                print(f"    ✓ Saved to {output_file}")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
    
    return created_files

def find_time_dimension(data):
    """Find the time dimension in the dataset"""
    possible_time_dims = ['time', 'time_counter', 't', 'T']
    for dim in data.dims:
        if dim in possible_time_dims:
            return dim
    return None

def update_variable_file_map_with_resampled(original_map, created_files):
    """
    Create new variable file map including both original and resampled files
    """
    updated_map = {}
    
    for var, original_entries in original_map.items():
        updated_map[var] = original_entries.copy()  # Keep original files
        
        # Add resampled files
        if var in created_files:
            updated_map[var].extend(created_files[var])
    
    return updated_map

def get_available_granularities(variable_file_map):
    """Same as your original function"""
    all_grans = set()
    for entries in variable_file_map.values():
        for entry in entries:
            all_grans.add(entry["granularity"])
    return sorted(all_grans, key=GRANULARITY_ORDER.index)

def get_maximum_granularity_with_all(variable_file_map, metric_requirements):
    """
    Same logic as your original but simplified since all files exist
    """
    all_grans = get_available_granularities(variable_file_map)
    all_vars = set()
    for fn, vars_list in metric_requirements.items():
        for var in vars_list:
            all_vars.add(var)
    
    # Check which granularities have all required variables
    gran_fn_map = {}
    
    for gran in all_grans:
        gran_fn_map[gran] = []
        
        for fn, vars_list in metric_requirements.items():
            # Check if all variables exist at this granularity
            all_vars_available = True
            for var in vars_list:
                var_exists = any(
                    entry["granularity"] == gran and os.path.exists(entry["file"])
                    for entry in variable_file_map.get(var, [])
                )
                if not var_exists:
                    all_vars_available = False
                    break
            
            if all_vars_available:
                gran_fn_map[gran].append(fn)
    
    # Find granularity with most metrics
    counts = [len(gran_fn_map[gran]) for gran in all_grans]
    max_index = counts.index(max(counts))
    return all_grans[max_index]

def get_data_simple(var, granularity, variable_file_map, cache=None):
    """
    Simplified data loading - just direct file access since all files exist
    """
    if cache is None:
        cache = {}
    
    key = (var, granularity)
    if key in cache:
        return cache[key]
    
    # Find direct file (should always exist after pre-processing)
    for entry in variable_file_map.get(var, []):
        if entry["granularity"] == granularity and os.path.exists(entry["file"]):
            print(f"Loading {var}@{granularity} from {entry['file']}")
            
            ds = xr.open_dataset(entry["file"], chunks={'time': 50})
            
            # Handle variable aliases
            actual_var = var if var in ds else next(
                (alias for alias in VARIABLE_ALIASES.get(var, []) if alias in ds), None
            )
            
            if actual_var:
                data = ds[actual_var]
                cache[key] = data
                return data
    
    raise ValueError(f"No file found for {var}@{granularity}")

def run_metric_simple(metric_name, metric_function, required_vars, granularity, variable_file_map, cache=None):
    """
    Simplified metric runner - no resampling logic needed
    """
    try:
        print(f"Running {metric_name} at {granularity}")
        inputs = [get_data_simple(var, granularity, variable_file_map, cache) for var in required_vars]
        
        # Load data only when computing
        loaded_inputs = [inp.load() if hasattr(inp, 'load') else inp for inp in inputs]
        result = metric_function(*loaded_inputs)
        print(f"✓ Success: {metric_name}")
        return result
    except Exception as e:
        print(f"✗ Failed: {metric_name} - {e}")
        return None

def run_all_metrics_simple(metric_requirements, metric_functions, variable_file_map, granularities=None):
    """
    Simplified version of your run_all_metrics - same interface, much faster
    """
    if granularities is None:
        granularities = GRANULARITY_ORDER
    
    cache = {}
    results = {}
    
    for gran in granularities:
        print(f"\n=== GRANULARITY: {gran} ===")
        
        for metric_name, required_vars in metric_requirements.items():
            if metric_name in metric_functions:
                result = run_metric_simple(
                    metric_name, 
                    metric_functions[metric_name], 
                    required_vars, 
                    gran, 
                    variable_file_map, 
                    cache
                )
                if result is not None:
                    results[(gran, metric_name)] = {'result': result, 'granularity': gran}
    
    print(f"\n=== FINAL RESULTS ===")
    for metric_name, info in results.items():
        print(f"✓ {metric_name} computed at {info['granularity']}")
    
    return results

def setup_precomputed_environment(original_yaml_file="DINO_map.yaml", output_dir="resampled_data"):
    """
    One-time setup: create all resampled files and updated YAML
    """
    print("=== SETTING UP PRE-COMPUTED ENVIRONMENT ===")
    
    # Load original map
    with open(original_yaml_file, "r") as file:
        variable_file_map = yaml.safe_load(file)["variable_file_map"]
    
    # Create all resampled files
    created_files = create_all_resampled_files(variable_file_map, output_dir)
    
    # Update map
    updated_map = update_variable_file_map_with_resampled(variable_file_map, created_files)
    
    # Save updated map
    updated_yaml_file = f"{original_yaml_file.split('.')[0]}_with_resampled.yaml"
    with open(updated_yaml_file, "w") as f:
        yaml.dump({"variable_file_map": updated_map}, f)
    
    print(f"\n✓ Setup complete!")
    print(f"✓ Resampled files saved to: {output_dir}")
    print(f"✓ Updated map saved to: {updated_yaml_file}")
    
    return updated_map, updated_yaml_file

# ============================================================================
# USAGE EXAMPLE - Same interface as your original code!
# ============================================================================

if __name__ == "__main__":
    # Step 1: One-time setup (run this once)
    print("Step 1: Creating pre-computed files...")
    updated_map, updated_yaml = setup_precomputed_environment("DINO_map.yaml")
    
    # Step 2: Use exactly like your original code but much faster!
    metric_requirements = {
        "check_density": ["temperature", "salinity"],
        "temperature_500m_30NS_metric": ["temperature"],
        "ACC_Drake_metric_2": ["velocity_u", "ssh"],
    }
    
    # Your dummy metric functions (same as before)
    def dummy_check_density(temperature, salinity):
        return temperature * 0.1 + salinity * 0.05
    
    def dummy_temperature_500m_30NS_metric(temperature):
        return temperature.mean()
    
    def dummy_ACC_Drake_metric_2(velocity_u, ssh):
        return velocity_u * ssh
    
    metric_functions = {
        "check_density": dummy_check_density,
        "temperature_500m_30NS_metric": dummy_temperature_500m_30NS_metric,
        "ACC_Drake_metric_2": dummy_ACC_Drake_metric_2,
    }
    
    print("\nStep 2: Running metrics (fast!)...")
    
    # Same interface as your original code!
    results = run_all_metrics_simple(metric_requirements, metric_functions, updated_map, granularities=None)
    
    print(f"\nResults keys: {list(results.keys())}")
    
    # Same functions as your original code!
    gran_max = get_maximum_granularity_with_all(updated_map, metric_requirements)
    print(f"Maximum granularity: {gran_max}")
    
    gran_available = get_available_granularities(updated_map)
    print(f"Available granularities: {gran_available}")
    
    # Run at optimal granularity only (super fast!)
    print(f"\nRunning at optimal granularity {gran_max} only...")
    optimal_results = run_all_metrics_simple(
        metric_requirements, 
        metric_functions, 
        updated_map, 
        granularities=[gran_max]
    )
    
    print(f"Optimal results: {list(optimal_results.keys())}")