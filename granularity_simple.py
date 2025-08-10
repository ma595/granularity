import os
import dask
import warnings
import xarray as xr

from granularity_simple_utils import run_all_metrics, run_metrics_intelligently_fixed
from metrics import dummy_ACC_Drake_metric_2, dummy_check_density, dummy_temperature_500m_30NS_metric

# Configure xarray for climate model data
xr.set_options(enable_cftimeindex=True)
warnings.filterwarnings("ignore", message="Unable to decode time axis into full numpy.datetime64")

import yaml

def main():
    # Load your data map
    with open("DINO_map.yaml", "r") as file:
        variable_file_map = yaml.safe_load(file)["variable_file_map"]

    # Define what you want to compute
    metric_requirements = {
        "check_density": ["temperature", "salinity"],
        "temperature_500m_30NS_metric": ["temperature"],
        "ACC_Drake_metric_2": ["velocity_u", "ssh"],
    }

    # ADD THIS - Define the metric functions dictionary
    metric_functions = {
        "check_density": dummy_check_density,
        "temperature_500m_30NS_metric": dummy_temperature_500m_30NS_metric,
        "ACC_Drake_metric_2": dummy_ACC_Drake_metric_2,
    }

    results, analysis = run_metrics_intelligently_fixed(metric_requirements, metric_functions, variable_file_map)

        # Print what was computed

    # Actually compute the lazy results
    print(f"\n=== COMPUTING RESULTS ===")
    print(f"Number of results: {len(results)}")
    for key, info in results.items():
        result = info['result']
        if hasattr(result, 'compute'):
            print(f"Computing {key}...")
            try:
                computed_result = result.compute()
                results[key]['result'] = computed_result
                print(f"✓ Computed {key}: {computed_result}")
            except Exception as e:
                print(f"✗ Failed to compute {key}: {e}")
        else:
            print(f"✓ {key} already computed: {result}")
    
    return results 
    

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

