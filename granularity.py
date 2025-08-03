import os
import yaml
import xarray as xr

GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]


def get_var_granularities(var, variable_file_map):
    # breakpoint()
    return {entry["granularity"] for entry in variable_file_map.get(var, [])}


def get_highest_supported_granularity(vars_required, variable_file_map):
    var_grans = [get_var_granularities(var, variable_file_map) for var in vars_required]
    if not all(var_grans):
        return None  # Some variable has no files at all

    common = set.intersection(*var_grans)
    if not common:
        return None  # No shared granularity

    return min(common, key=get_granularity_rank)  # Finest shared


def get_runnable_metrics_at_max_frequency(metric_requirements, variable_file_map):
    """
    Given a dictionary of metric requirements and a variable file map,
    return a dictionary of runnable metrics at the highest supported granularity.
    """
    runnable = {}
    # breakpoint()
    for metric, vars_required in metric_requirements.items():
        gran = get_highest_supported_granularity(
            vars_required, variable_file_map
        )  # vars_required = ""
        if gran:
            runnable[metric] = {"vars": vars_required, "granularity": gran}
    return runnable


def get_granularity_rank(gran):
    return GRANULARITY_ORDER.index(gran)




def get_resample_plan(
    var, target_gran, variable_file_map, resample_cache, mode="lazy", cache_dir=None
):
    freq_map = {"10d": "10D", "1m": "1M", "3m": "3M", "1y": "1Y"}
    key = (var, target_gran)
    if key in resample_cache:
        print(f"[CACHE HIT] {key}")
        return resample_cache[key]

    # 1. Try direct file from variable_file_map
    for entry in variable_file_map.get(var, []):
        if entry["granularity"] == target_gran:
            print(f"[FOUND DIRECT] {var}@{target_gran}")
            resample_cache[key] = lambda entry=entry: xr.open_dataset(entry["file"])[var]
            return resample_cache[key]

    # 2. Find the closest available finer granularity (OPTIMIZED!)
    target_rank = GRANULARITY_ORDER.index(target_gran)
    
    # Get available granularities for this variable
    available_grans = {entry["granularity"] for entry in variable_file_map.get(var, [])}
    
    # Find the finest available granularity that's finer than target
    best_source_gran = None
    for finer_rank in reversed(range(target_rank)):
        candidate_gran = GRANULARITY_ORDER[finer_rank]
        if candidate_gran in available_grans:
            best_source_gran = candidate_gran
            break  # Take the first (finest) match
    
    if best_source_gran is None:
        print(f"[FAILED] No finer granularity available for {var}@{target_gran}")
        resample_cache[key] = None
        return None

    # 3. Get or create plan for the best source (only one recursive call!)
    source_key = (var, best_source_gran)
    if source_key in resample_cache:
        source_plan = resample_cache[source_key]
    else:
        source_plan = get_resample_plan(
            var, best_source_gran, variable_file_map, resample_cache, mode=mode, cache_dir=cache_dir
        )
        resample_cache[source_key] = source_plan

    if source_plan is not None:
        print(f"[RESAMPLE] {var}@{best_source_gran} → {target_gran}")

        def lazy_loader(source_plan=source_plan, gran=target_gran):
            try:
                data = source_plan()
                print(f"[DEBUG] Data loaded, type: {type(data)}")
                
                freq = freq_map[gran]
                print(f"[DEBUG] Using frequency: {freq}")
                result = data.resample(time=freq).mean()
                print(f"[DEBUG] Resample successful")
                return result
                
            except Exception as e:
                print(f"[ERROR] Resample failed for {var} to {target_gran}: {e}")
                raise

        if mode == "eager":
            path = f"{cache_dir}/{var}_{target_gran}.zarr"
            if not os.path.exists(path):
                print(f"[WRITE] {path}")
                lazy_loader().to_zarr(path)
            resample_cache[key] = lambda path=path: xr.open_zarr(path)[var]
        else:
            resample_cache[key] = lazy_loader

        return resample_cache[key]

    print(f"[FAILED] No resample plan for {var}@{target_gran}")
    resample_cache[key] = None
    return None

# required - returns the resample_cache
# it loops over all metric_vars i.e. T, S, SSH U, V
# and computes all the available granularities for each metric.
# can't this be simplified - GRANULARITY_ORDER doesn't need to be looped over?
# well it picks up the data that is already there - as a lambda to open.
# computes the resample cache as a side effect
# def build_resample_cache_for_all_metrics(metric_requirements, variable_file_map):
#     resample_cache = {}
#     # Get all variables mentioned across all metrics
#     all_vars = set(var for vars in metric_requirements.values() for var in vars)

#     for var in all_vars:
#         for gran in GRANULARITY_ORDER:
#             _ = get_resample_plan(var, , variable_file_map, resample_cache)

#     return resample_cache

def build_resample_cache_for_granularity(metric_requirements, variable_file_map, desired_gran):
    resample_cache = {}
    # Get all variables mentioned across all metrics
    all_vars = set(var for vars in metric_requirements.values() for var in vars)

    for var in all_vars:
        _ = get_resample_plan(var, desired_gran, variable_file_map, resample_cache)

    return resample_cache

def get_minimum_supported_granularity(metric_requirements, resample_cache):
    supported_per_metric = []

    for metric, vars in metric_requirements.items():
        metric_supported = set()
        for gran in GRANULARITY_ORDER:
            if all(
                (var, gran) in resample_cache
                and resample_cache[(var, gran)] is not None
                for var in vars
            ):
                metric_supported.add(gran)
        if not metric_supported:
            print(f"No supported granularity for metric '{metric}'")
            return None
        supported_per_metric.append(metric_supported)

    shared = set.intersection(*supported_per_metric)
    if not shared:
        return None
    return min(shared, key=GRANULARITY_ORDER.index)


def get_available_granularities(variable_file_map):
    all_grans = set()
    for entries in variable_file_map.values():
        for entry in entries:
            all_grans.add(entry["granularity"])
    return sorted(all_grans, key=GRANULARITY_ORDER.index)


def check_variable_group_available(granularity, required_vars, variable_file_map):
    return all(
        any(
            entry["granularity"] == granularity
            for entry in variable_file_map.get(var, [])
        )
        for var in required_vars
    )


def evaluate_resample_plan(var, granularity, data_cache, resample_cache):
    key = (var, granularity)

    # 1. If we've already computed this, return it
    if key in data_cache:
        print(f"[CACHE HIT - data] {key}")
        return data_cache[key]

    # 2. If we don't even have a plan, fail
    if key not in resample_cache or resample_cache[key] is None:
        raise ValueError(f"[MISSING PLAN] No resample plan for {var}@{granularity}")

    # 3. Run the lazy plan and standardize the result
    print(f"[EVALUATING] {key}")
    raw_data = resample_cache[key]()
    
    # Standardize if it's a dataset (from direct file load)
    if hasattr(raw_data, 'variables'):
        print(f"[STANDARDIZING] Dataset variables: {list(raw_data.variables.keys())}")
        standardized_ds = standardize_variables(raw_data, VARIABLE_ALIASES)
        result = standardized_ds[var] if var in standardized_ds else raw_data[var]
    else:
        # Already a DataArray (from resampling chain)
        result = raw_data
    
    data_cache[key] = result
    return result

# def run_metrics_over_granularities(
#     granularity_list,
#     metric_requirements,
#     metric_functions,
#     data_cache,
#     resample_cache,  # unified view
# ):
#     for gran in granularity_list:
#         ran_metrics = []
#         skipped_metrics = []

#         for metric_name, required_vars in metric_requirements.items():
#             # Check if all plans exist
#             if all(
#                 (var, gran) in resample_cache
#                 and resample_cache[(var, gran)] is not None
#                 for var in required_vars
#             ):
#                 try:
#                     inputs = [
#                         evaluate_resample_plan(var, gran, data_cache, resample_cache)
#                         for var in required_vars
#                     ]
#                     print(f"Running {metric_name} at {gran}...")
#                     # metric_fn = metric_functions[metric_name]
#                     # metric_fn(*inputs)
#                     ran_metrics.append(metric_name)
#                 except Exception as e:
#                     print(f"Failed: {metric_name} at {gran}: {e}")
#             else:
#                 skipped_metrics.append(metric_name)

#         print(f"\nSummary for {gran}:")
#         print(f"  Ran: {ran_metrics}")
#         print(f"  Skipped: {skipped_metrics}\n")


def _compute_metrics_at_granularity(
    metric_requirements, 
    metric_functions, 
    granularity, 
    data_cache, 
    resample_cache
):
    """
    Helper function: compute all possible metrics at a specific granularity.
    Returns (results, ran_metrics, failed_metrics)
    """
    results = {}
    ran_metrics = []
    failed_metrics = []
    
    for metric_name, required_vars in metric_requirements.items():
        try:
            print(f"  Computing {metric_name}...")
            
            # Get the data
            inputs = [
                evaluate_resample_plan(var, granularity, data_cache, resample_cache)
                for var in required_vars
            ]
            
            # Run the metric function
            if metric_name in metric_functions:
                result = metric_functions[metric_name](*inputs)
                results[metric_name] = {
                    'result': result,
                    'granularity': granularity,
                    'variables': required_vars
                }
                ran_metrics.append(metric_name)
            else:
                print(f"    ! No function defined for {metric_name}")
                failed_metrics.append(metric_name)
                
        except Exception as e:
            print(f"    ✗ Failed {metric_name}: {e}")
            failed_metrics.append(metric_name)
    
    return results, ran_metrics, failed_metrics

# def _check_all_metrics_can_run(metric_requirements, granularity, variable_file_map, resample_cache):
#     """
#     Helper function: check if ALL metrics can run at a specific granularity.
#     Returns (can_run, missing_metrics)
#     """
#     missing_metrics = []
    
#     for metric_name, required_vars in metric_requirements.items():
#         metric_can_run = all(
#             get_resample_plan(var, granularity, variable_file_map, resample_cache) is not None
#             for var in required_vars
#         )
        
#         if not metric_can_run:
#             missing_metrics.append(metric_name)
    
#     return len(missing_metrics) == 0, missing_metrics

# def compute_all_metrics_at_minimum_granularity(
#     metric_requirements, 
#     metric_functions, 
#     variable_file_map
# ):
#     """
#     Find the finest granularity where ALL metrics can run, then compute them there.
#     """
#     resample_cache = {}
#     data_cache = {}
    
#     print("Finding minimum granularity where all metrics can run...")
    
#     # Try each granularity from finest to coarsest
#     for gran in GRANULARITY_ORDER:
#         print(f"\n[CHECKING] Can all metrics run at {gran}?")
        
#         all_can_run, missing_metrics = _check_all_metrics_can_run(
#             metric_requirements, gran, variable_file_map, resample_cache
#         )
        
#         if all_can_run:
#             print(f"✓ ALL metrics can run at {gran}! Computing...")
            
#             results, ran_metrics, failed_metrics = _compute_metrics_at_granularity(
#                 metric_requirements, metric_functions, gran, data_cache, resample_cache
#             )
            
#             print(f"\n{'='*50}")
#             print(f"COMPUTED ALL METRICS AT {gran}")
#             print(f"{'='*50}")
#             print(f"Successful: {ran_metrics}")
#             print(f"Failed: {failed_metrics}")
            
#             return results, gran, data_cache, resample_cache
        
#         else:
#             print(f"✗ Cannot run all metrics at {gran}")
#             print(f"  Missing: {missing_metrics}")
    
#     print("\n✗ No granularity supports all metrics!")
#     return {}, None, data_cache, resample_cache

def compute_metrics_with_available_data(
    metric_requirements, 
    metric_functions, 
    variable_file_map, 
    preferred_granularities=None
):
    """
    Compute metrics where we have data, resample when needed.
    """
    if preferred_granularities is None:
        preferred_granularities = get_available_granularities(variable_file_map)
    
    resample_cache = {}
    data_cache = {}
    all_results = {}
    
    for gran in preferred_granularities:
        print(f"\n{'='*50}")
        print(f"TRYING GRANULARITY: {gran}")
        print(f"{'='*50}")
        
        # Filter to only metrics we haven't computed yet
        remaining_metrics = {
            name: vars_req for name, vars_req in metric_requirements.items()
            if name not in all_results
        }
        
        if not remaining_metrics:
            print("All metrics already computed!")
            break
        
        # Check which metrics can run at this granularity
        runnable_metrics = {}
        for metric_name, required_vars in remaining_metrics.items():
            plans_available = all(
                get_resample_plan(var, gran, variable_file_map, resample_cache) is not None
                for var in required_vars
            )
            if plans_available:
                runnable_metrics[metric_name] = required_vars
        
        if runnable_metrics:
            results, ran_metrics, failed_metrics = _compute_metrics_at_granularity(
                runnable_metrics, metric_functions, gran, data_cache, resample_cache
            )
            
            all_results.update(results)
            
            print(f"\nResults for {gran}:")
            print(f"  Computed: {ran_metrics}")
            print(f"  Failed: {failed_metrics}")
        else:
            print(f"  - No remaining metrics can run at {gran}")
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for metric_name, info in all_results.items():
        print(f"✓ {metric_name} → {info['granularity']} | vars: {info['variables']}")
    
    return all_results, data_cache, resample_cache

