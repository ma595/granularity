import os
import dask
import warnings
import xarray as xr

from gran_utils import (
    run_all_metrics_with_cache,
    run_metrics_intelligently_with_cache,
)
# from metrics import (
#     dummy_ACC_Drake_metric_2,
#     dummy_check_density,
#     dummy_temperature_500m_30NS_metric,
# )

from metric_real import (
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


def main():
    # Load your data map
    with open("config/DINO_simple_map.yaml", "r") as file:
        variable_file_map = yaml.safe_load(file)["variable_file_map"]

    # Load mesh_mask once at the beginning
    print("Loading mesh_mask...")
    mesh_mask_file = variable_file_map["mesh_mask"][0]["file"]
    mesh_mask = xr.open_dataset(mesh_mask_file)

    from standardise_variables import standardize_variables, VARIABLE_ALIASES
    mesh_mask_ds = standardize_variables(mesh_mask, VARIABLE_ALIASES)  # or whatever variables you need from it
    mesh_mask = mesh_mask_ds  # or extract specific variables you need
    # sanitize mesh_mask using src/standardize_variables.py
    # mesh_mask_ds = standardize_variables(mesh_mask)  # or whatever variables you need from it
    # Don't standardize mesh_mask - assume it already has correct names
    print(f"✓ Loaded mesh_mask from: {mesh_mask_file}")
    print(f"Available variables: {list(mesh_mask.variables.keys())}")

    # TODO: this all needs to be reworked:

    # Define what you want to compute
    metric_requirements = {
        "check_density": ["temperature"],
        "temperature_500m_30NS_metric": ["temperature"],
        "ACC_Drake_metric": ["velocity_u"],
        "NASTG_BSF_max": ["temperature", "ssh"],
    }

    # ADD THIS - Define the metric functions dictionary
    metric_functions = {
        "check_density": check_density,
        "temperature_500m_30NS_metric": lambda temp: temperature_500m_30NS_metric(temp, mesh_mask),
        "ACC_Drake_metric": lambda vel: ACC_Drake_metric(vel, mesh_mask),
        "NASTG_BSF_max": lambda temp, ssh: NASTG_BSF_max(temp, ssh, mesh_mask),
    }

    # results, analysis = run_metrics_intelligently_with_cache(metric_requirements, metric_functions, variable_file_map)

    results = run_all_metrics_with_cache(
        metric_requirements=metric_requirements,
        metric_functions=metric_functions,
        variable_file_map=variable_file_map,
        granularities=["1y"],
        save_to_cache=False
    )

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
