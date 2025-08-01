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
            resample_cache[key] = lambda: xr.open_dataset(
                entry["file"], chunks={"time": 10}
            )[var]
            return resample_cache[key]

    # 2. Try to build from next-finer granularity
    target_rank = GRANULARITY_ORDER.index(target_gran)
    for finer_rank in reversed(range(target_rank)):
        finer_gran = GRANULARITY_ORDER[finer_rank]
        finer_key = (var, finer_gran)

        # RECURSIVE CACHE CHECK
        if finer_key in resample_cache:
            finer_plan = resample_cache[finer_key]
        else:
            finer_plan = get_resample_plan(
                var,
                finer_gran,
                variable_file_map,
                resample_cache,
                mode=mode,
                cache_dir=cache_dir,
            )
            resample_cache[finer_key] = finer_plan

        if finer_plan is not None:
            print(f"[RESAMPLE] {var}@{finer_gran} → {target_gran}")

            def lazy_loader(finer_plan=finer_plan, gran=target_gran):
                print(f"[LAZY EVAL] Resampling {var} to {target_gran}")

                return finer_plan().resample(time=freq_map[gran]).mean()

            if mode == "eager":
                path = f"{cache_dir}/{var}_{target_gran}.zarr"
                if not os.path.exists(path):
                    print(f"[WRITE] {path}")
                    lazy_loader().to_zarr(path)
                resample_cache[key] = lambda: xr.open_zarr(path, chunks={"time": 10})[
                    var
                ]
            else:
                resample_cache[key] = lazy_loader

            return resample_cache[key]

    # No available path
    print(f"[FAILED] No resample plan for {var}@{target_gran}")
    resample_cache[key] = None
    return None


# required - returns the resample_cache
# it loops over all metric_vars i.e. T, S, SSH U, V
# and computes all the available granularities for each metric.
# can't this be simplified - GRANULARITY_ORDER doesn't need to be looped over?
# well it picks up the data that is already there - as a lambda to open.
# computes the resample cache as a side effect
def build_resample_cache_for_all_metrics(metric_requirements, variable_file_map):
    resample_cache = {}
    # Get all variables mentioned across all metrics
    all_vars = set(var for vars in metric_requirements.values() for var in vars)

    for var in all_vars:
        for gran in GRANULARITY_ORDER:
            _ = get_resample_plan(var, gran, variable_file_map, resample_cache)

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

    # 3. Run the lazy plan and cache the result
    print(f"[EVALUATING] {key}")
    result = resample_cache[key]()
    data_cache[key] = result
    return result


def run_metrics_over_granularities(
    granularity_list,
    metric_requirements,
    metric_functions,
    data_cache,
    resample_cache,  # unified view
):
    for gran in granularity_list:
        ran_metrics = []
        skipped_metrics = []

        for metric_name, required_vars in metric_requirements.items():
            # Check if all plans exist
            if all(
                (var, gran) in resample_cache
                and resample_cache[(var, gran)] is not None
                for var in required_vars
            ):
                try:
                    inputs = [
                        evaluate_resample_plan(var, gran, data_cache, resample_cache)
                        for var in required_vars
                    ]
                    print(f"Running {metric_name} at {gran}...")
                    metric_fn = metric_functions[metric_name]
                    metric_fn(*inputs)
                    ran_metrics.append(metric_name)
                except Exception as e:
                    print(f"Failed: {metric_name} at {gran}: {e}")
            else:
                skipped_metrics.append(metric_name)

        print(f"\nSummary for {gran}:")
        print(f"  Ran: {ran_metrics}")
        print(f"  Skipped: {skipped_metrics}\n")
