import os


def get_var_granularities(var, variable_file_map):
    # breakpoint()
    return {entry["granularity"] for entry in variable_file_map.get(var, [])}


GRANULARITY_ORDER = ["10d", "1m", "3m", "1y"]


def get_granularity_rank(gran):
    return GRANULARITY_ORDER.index(gran)


def get_highest_supported_granularity(vars_required, variable_file_map):
    var_grans = [get_var_granularities(var, variable_file_map) for var in vars_required]
    # breakpoint()
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


metric_requirements = {
    "check_density": ["Temperature", "Salinity"],
    "temperature_500m_30NS_metric": ["Temperature"],
    "ACC_Drake_metric_2": ["Velocity_U", "SSH"],
}
# load yaml file with variable file map
import yaml

with open("DINO_map.yaml", "r") as file:
    variable_file_map = yaml.safe_load(file)
    variable_file_map = variable_file_map["variable_file_map"]
# print(variable_file_map)

# loop over granularities and print runnable metrics

# we should provide a selected granularities:

selected_gran = ["10d", "1m", "1y"]

# difficulty - we want to compute the available metrics at a given granularity
# but we also want to compute the minimum granularity of a set metrics.
# but we can always downsample (never upsample)
#

# if this isn't provided - we provide metrics of all provided granularities (files)

# if we don't provide the selected granularity
# 1. get the range of granularities i.e. 10d, 1m, 3m
# 2. for each granularity compute the metrics that we can.
# 3. for the next granularity - identify if we have a higher granularity input if so, downsample.


# preprocessing step:
# 1. Automatically compute range of available granularities:
# 2. If selected gran hasn't been provided i.e [1m, 3m, 6m].
# 2b for each granularity in granularity range, check if we can get the temperature, salinity, SSH, U, V.
# Resample from higher frequency data if it doesn't already exist. Store everything or compute on the fly lazily? Dask?
# some files can be 7GB or more - particularly if sampled at high frequency.

# Compute metrics step
# 1. Loop over granularity range:
# 2. if


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


# this just picks up the range of granularities i.e. 10d, 1m, 3m
# we could use this upper limit to just run at a single granularity. otherwise run across all?
# if ["all", "min"]

# want to get the minimum for which we can compute all metrics - assuming that resampling is possible.


gran_range = get_available_granularities(variable_file_map)
# this checks if for a given metric function it can be computed at the required granularity
test_metric_check_density_1m = check_variable_group_available(
    "3m",
    required_vars=metric_requirements["check_density"],
    variable_file_map=variable_file_map,
)


from functools import lru_cache
import xarray as xr

resample_cache = {}  # (var, granularity) -> lazy loader
freq_map = {"10d": "10D", "1m": "1M", "3m": "3M", "1y": "1Y"}


def get_resample_plan(var, target_gran, variable_file_map, mode="lazy", cache_dir=None):
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
                var, finer_gran, variable_file_map, mode=mode, cache_dir=cache_dir
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


# def get_resample_plan(var, target_gran, variable_file_map, mode="lazy", cache_dir=None):
#     key = (var, target_gran)
#     if key in resample_cache:
#         return resample_cache[key]

#     # 1. Try direct file from variable_file_map
#     for entry in variable_file_map.get(var, []):
#         if entry["granularity"] == target_gran:
#             resample_cache[key] = lambda: xr.open_dataset(
#                 entry["file"], chunks={"time": 10}
#             )[var]
#             return resample_cache[key]

#     # 2. Try to build from next-finer granularity
#     target_rank = GRANULARITY_ORDER.index(target_gran)
#     for finer_rank in reversed(range(target_rank)):
#         finer_gran = GRANULARITY_ORDER[finer_rank]
#         finer_plan = get_resample_plan(
#             var, finer_gran, variable_file_map, mode=mode, cache_dir=cache_dir
#         )
#         if finer_plan is not None:

#             def lazy_loader(finer_plan=finer_plan, gran=target_gran):
#                 return finer_plan().resample(time=freq_map[gran]).mean()

#             if mode == "eager":
#                 path = f"{cache_dir}/{var}_{target_gran}.zarr"
#                 if not os.path.exists(path):
#                     lazy_loader().to_zarr(path)
#                 resample_cache[key] = lambda: xr.open_zarr(path, chunks={"time": 10})[
#                     var
#                 ]
#             else:
#                 resample_cache[key] = lazy_loader
#             return resample_cache[key]

#     # No source available
#     resample_cache[key] = None
#     return None


def get_supported_granularities_for_metric_with_resampling(
    metric_vars, variable_file_map
):
    possible_grans = set(GRANULARITY_ORDER)

    for var in metric_vars:
        var_grans = set()
        for g in GRANULARITY_ORDER:
            if get_resample_plan(var, g, variable_file_map) is not None:
                var_grans.add(g)
        possible_grans = possible_grans.intersection(var_grans)

    print(possible_grans)
    return possible_grans


def get_minimum_global_metric_granularity(metric_requirements, variable_file_map):
    supported_per_metric = []

    for metric, required_vars in metric_requirements.items():

        print(metric, required_vars)
        supported = get_supported_granularities_for_metric_with_resampling(
            required_vars, variable_file_map
        )
        if not supported:
            print(f"⚠️ No supported granularity for metric '{metric}'")
            return None  # or raise exception
        supported_per_metric.append(supported)

    shared = set.intersection(*supported_per_metric)
    if not shared:
        return None  # no global granularity possible

    return min(shared, key=GRANULARITY_ORDER.index)


print(get_minimum_global_metric_granularity(metric_requirements, variable_file_map))

print(resample_cache)
# TODO: check if I'm using black or ruff to format.
# Now down-sample

lazy_loader = resample_cache[("SSH", "1y")]
import inspect

print(inspect.signature(lazy_loader))

print(resample_cache[("Temperature", "3m")])


# runnable = {}
# for gran in GRANULARITY_ORDER:
#     print(f"Granularity: {gran}")

#     for metric_func_name, var_requirements in metric_requirements.items():

#         break
#         # runnable[metric_func_name] =

# runnable = get_runnable_metrics_at_max_frequency(metric_requirements, variable_file_map)
# runnable = {k: v for k, v in runnable.items() if v["granularity"] == i}
# if runnable:
#     print(f"Runnable metrics at {i}:")


# runnable = get_runnable_metrics_at_max_frequency(metric_requirements, variable_file_map)
# for metric, info in runnable.items():
#     print(f"✓ {metric} → granularity: {info['granularity']} | vars: {info['vars']}")


def evaluate_resample_plan(var, granularity, data_cache):
    key = (var, granularity)

    # ✅ 1. If we've already computed this, return it
    if key in data_cache:
        print(f"[CACHE HIT - data] {key}")
        return data_cache[key]

    # ✅ 2. If we don't even have a plan, fail
    if key not in resample_cache or resample_cache[key] is None:
        raise ValueError(f"[MISSING PLAN] No resample plan for {var}@{granularity}")

    # ✅ 3. Run the lazy plan and cache the result
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
                        evaluate_resample_plan(var, gran, data_cache)
                        for var in required_vars
                    ]
                    print(f"✅ Running {metric_name} at {gran}...")
                    metric_fn = metric_functions[metric_name]
                    metric_fn(*inputs)
                    ran_metrics.append(metric_name)
                except Exception as e:
                    print(f"❌ Failed: {metric_name} at {gran}: {e}")
            else:
                skipped_metrics.append(metric_name)

        print(f"\n🔍 Summary for {gran}:")
        print(f"  ✅ Ran: {ran_metrics}")
        print(f"  ❌ Skipped: {skipped_metrics}\n")


# TODO:
## check that the functions are not repeats of each other.
