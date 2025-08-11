import yaml
from granularity import (
    get_available_granularities,
    get_highest_supported_granularity,
    check_variable_group_available,
    get_minimum_supported_granularity,
    build_resample_cache_for_all_metrics,
    get_runnable_metrics_at_max_frequency,
    run_metrics_over_granularities,
)

metric_requirements = {
    "check_density": ["temperature", "salinity"],
    "temperature_500m_30NS_metric": ["temperature"],
    "ACC_Drake_metric_2": ["velocity_u", "ssh"],
}
# load yaml file with variable file map

with open("DINO_map.yaml", "r") as file:
    variable_file_map = yaml.safe_load(file)
    variable_file_map = variable_file_map["variable_file_map"]
# print(variable_file_map)

# loop over granularities and print runnable metrics

# we should provide a selected granularities:

selected_gran = ["10d", "1m", "1y"]

gran_range = get_available_granularities(variable_file_map)

assert gran_range == ["10d", "1m", "3m"]

# gets the highest supported granularity for a given set of inputs
# this is designed to work when we run items().
finest = get_highest_supported_granularity(
    vars_required=["temperature", "salinity"], variable_file_map=variable_file_map
)

assert finest == "10d"

runnable = get_runnable_metrics_at_max_frequency(
    metric_requirements=metric_requirements, variable_file_map=variable_file_map
)

# now print
for metric, info in runnable.items():
    print(f"✓ {metric} → granularity: {info['granularity']} | vars: {info['vars']}")

from metrics import (
    dummy_check_density,
    dummy_temperature_500m_30NS_metric,
    dummy_ACC_Drake_metric_2,
)

metric_functions = {
    "check_density": dummy_check_density,
    "temperature_500m_30NS_metric": dummy_temperature_500m_30NS_metric,
    "ACC_Drake_metric_2": dummy_ACC_Drake_metric_2,
}

# this checks if for a given metric function it can be computed at the required granularity
# not really used - can delete
test_metric_check_density_1m = check_variable_group_available(
    "3m",
    required_vars=metric_requirements["check_density"],
    variable_file_map=variable_file_map,
)

# print(resample_cache)
resample_cache = build_resample_cache_for_all_metrics(
    metric_requirements, variable_file_map
)

print(resample_cache)

data_cache = {}
computed_metrics = run_metrics_over_granularities(
    granularity_list=["10d", "1m", "3m"],
    metric_requirements=metric_requirements,
    metric_functions=metric_functions,
    data_cache=data_cache,
    resample_cache=resample_cache,
)

print("Computed metrics:")
for (gran, metric_name), result in computed_metrics.items():
    print(f"  ✓ {metric_name}@{gran} = {result}")

# print(resample_cache[("Temperature", "3m")])

# TODO:
# check that the functions are not repeats of each other.
# check evaluations work on real data
# how to account for comparisons?
