import os
import dask
from collections import defaultdict
import warnings
import xarray as xr

from standardise_variables import standardize_variables, VARIABLE_ALIASES
from granularity.gran_utils import (
    preload_and_align_all_variables,
    run_all_metrics_with_cache,
    run_metrics_intelligently_with_cache,
)

from granularity.gran_analysis import (
    analyze_metric_requirements,
    show_availability_summary,
    get_maximum_granularity_with_all,
)

from granularity.two_step import two_step_resample_all_aligned, two_step_metrics_all


from metrics_io import write_metrics_to_csv


from metrics_real import (
    ACC_Drake_metric,
    NASTG_BSF_max,
    check_density,
    temperature_500m_30NS_metric,
)

# Configure xarray for climate model data
xr.set_options(enable_cftimeindex=True)
warnings.filterwarnings(
    "ignore", message="Unable to decode time axis into full numpy.datetime64"
)

import yaml


def group_by_granularity(results):
    """{(gran, metric): xr.DataArray} -> {gran: {metric: xr.DataArray}}"""
    out = defaultdict(dict)
    for (gran, metric), da in results.items():
        out[gran][metric] = da
    return dict(out)


def select_gran(results, gran):
    """View: all metrics at one granularity (stays flat)."""
    return {metric: v for (g, metric), v in results.items() if g == gran}


def two_step(analysis, variable_file_map, metric_requirements, metric_functions):
    # A) Analyze what’s possible (you already do this)
    analysis = analyze_metric_requirements(variable_file_map, metric_requirements)

    targets = ["10d", "1m", "3m"]
    # B) STEP 1 — materialize (pick granularities you care about)
    # two_step_resample_all_aligned(
    #     variable_file_map,
    #     analysis,
    #     targets=targets,
    #     disk_cache_dir="./resampled_cache_materialised",
    # )

    # C) STEP 2 — compute metrics from only the materialized data
    results_by_granularity = two_step_metrics_all(
        metric_requirements,
        metric_functions,
        targets=targets,
        disk_cache_dir="./resampled_cache_materialised",
    )

    for gran in results_by_granularity:
        write_metrics_to_csv(
            results_by_granularity[gran], f"outputs/metrics_{gran}.csv"
        )

    # (Optional) compute lazy results now
    # results = compute_results_parallel_fixed(results)


def run_on_the_fly(analysis, variable_file_map, metric_requirements, metric_functions):
    grans = analysis["available_granularities"]

    cache = preload_and_align_all_variables(
        variable_file_map, grans, analysis, save_to_cache=False
    )

    # print(cache.keys())
    # now run metrics

    print(analysis)
    runnable_functions_gran = analysis["runnable_metrics"]

    results_by_granularity = {}

    print(runnable_functions_gran)

    for gran, func_list in runnable_functions_gran.items():
        print(f"\n=== Running metrics at granularity: {gran} ===")
        for func_name in func_list:
            print(f"Running metric: {func_name}")
            if func_name in metric_functions:
                # Get the required variables for this metric
                required_vars = metric_requirements[func_name]
                # Fetch the aligned data from cache
                inputs = [cache[(var, gran)] for var in required_vars]
                # Call the metric function with the inputs
                result = metric_functions[func_name](*inputs)
                # print(f"Result for {func_name}: {result}")
                if gran not in results_by_granularity:
                    results_by_granularity[gran] = {}
                results_by_granularity[gran][func_name] = result
            else:
                print(
                    f"Metric function '{func_name}' not found in metric_functions dictionary."
                )

    # now write results to file
    # print(analysis)
    # print(show_availability_summary(variable_file_map, metric_requirements))
    # print(get_maximum_granularity_with_all(variable_file_map, metric_requirements))

    for gran in results_by_granularity:
        write_metrics_to_csv(
            results_by_granularity[gran], f"outputs/metrics_{gran}.csv"
        )


def main():
    # Load your data map
    with open("config/DINO_map.yaml", "r") as file:
        variable_file_map = yaml.safe_load(file)["variable_file_map"]

    # Load mesh_mask once at the beginning
    print("Loading mesh_mask...")
    mesh_mask_file = variable_file_map["mesh_mask"][0]["file"]
    mesh_mask = xr.open_dataset(mesh_mask_file)

    mesh_mask_ds = standardize_variables(
        mesh_mask, VARIABLE_ALIASES
    )  # or whatever variables you need from it
    mesh_mask = mesh_mask_ds  # or extract specific variables you need
    # sanitize mesh_mask using src/standardize_variables.py
    # mesh_mask_ds = standardize_variables(mesh_mask)  # or whatever variables you need from it
    # Don't standardize mesh_mask - assume it already has correct names
    print(f"✓ Loaded mesh_mask from: {mesh_mask_file}")
    print(f"Available variables: {list(mesh_mask.variables.keys())}")

    # TODO: this all needs to be reworked:

    # Define what you want to compute
    metric_requirements = {
        "check_density": ["temperature"],
        "temperature_500m_30NS_metric": ["temperature"],
        "ACC_Drake_metric": ["velocity_u"],
        "NASTG_BSF_max": ["velocity_v", "ssh"],
    }

    # ADD THIS - Define the metric functions dictionary
    metric_functions = {
        "check_density": check_density,
        "temperature_500m_30NS_metric": lambda temp: temperature_500m_30NS_metric(
            temp, mesh_mask
        ),
        "ACC_Drake_metric": lambda vel: ACC_Drake_metric(vel, mesh_mask),
        "NASTG_BSF_max": lambda vel_v, ssh: NASTG_BSF_max(vel_v, ssh, mesh_mask),
    }
    # results, analysis = run_metrics_intelligently_with_cache(metric_requirements, metric_functions, variable_file_map)

    ## analysis

    analysis = analyze_metric_requirements(variable_file_map, metric_requirements)

    # run_on_the_fly(analysis, variable_file_map, metric_requirements, metric_functions)

    two_step(analysis, variable_file_map, metric_requirements, metric_functions)

    # options:
    ## We can get the maximum granularity where the all exist - this gives us a month

    ## We can loop over all granularities and just compute all metrics at that granularity with resampling (intelligent)

    # print(analysis)

    # results = run_all_metrics_with_cache(
    #     metric_requirements=metric_requirements,
    #     metric_functions=metric_functions,
    #     variable_file_map=variable_file_map,
    #     granularities=["1y"],
    #     save_to_cache=False
    # )
    # perhaps I want a unified list of each variable at each granularity.
    # i.e. (1y, "temperature", "velocity_u", "velocity_v", "ssh")

    # I want to get a list of all ('granularity', 'fn', 'inputs')

    # then run through them as follows:

    # # Actually compute the lazy results
    # print(f"\n=== COMPUTING RESULTS ===")
    # print(f"Number of results: {len(results)}")
    # for key, info in results.items():
    #     result = info["result"]
    #     if hasattr(result, "compute"):
    #         print(f"Computing {key}...")
    #         try:
    #             computed_result = result.compute()
    #             results[key]["result"] = computed_result
    #             print(f"✓ Computed {key}: {computed_result}")
    #         except Exception as e:
    #             print(f"✗ Failed to compute {key}: {e}")
    #     else:
    #         print(f"✓ {key} already computed: {result}")

    # print(results.keys())

    # # provide a separate file for each granularity and write results to CSV
    # for granularity in results:
    #     # write_metrics_to_csv(results[granularity], f"outputs/metrics_{granularity}.csv")
    #     print(results[granularity])

    # print(results[('1m', 'temperature_500m_30NS_metric')])

    # return results

    # Run everything!
    # results = run_all_metrics(metric_requirements, metric_functions, variable_file_map, granularities=['1y'], down_sample=True)

    # print(results.keys())

    # print(results[("10d", "check_density")]['result'].compute())

    # results_out = compute_results_parallel(results, n_workers=1)

    # print(results)

    # get maximum granularities
    # gran_max = get_maximum_granularity_with_all(variable_file_map, metric_requirements)
    # print(gran_max)

    # gran_available = get_available_granularities(variable_file_map)

    # print(gran_available)


if __name__ == "__main__":
    main()
