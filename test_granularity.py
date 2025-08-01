import yaml
from granularity import (
    get_available_granularities,
    get_highest_supported_granularity,
    check_variable_group_available,
    get_minimum_supported_granularity,
    build_resample_cache_for_all_metrics,
    get_runnable_metrics_at_max_frequency,
)

metric_requirements = {
    "check_density": ["Temperature", "Salinity"],
    "temperature_500m_30NS_metric": ["Temperature"],
    "ACC_Drake_metric_2": ["Velocity_U", "SSH"],
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
    vars_required=["Temperature", "Salinity"], variable_file_map=variable_file_map
)

assert finest == "10d"

runnable = get_runnable_metrics_at_max_frequency(
    metric_requirements=metric_requirements, variable_file_map=variable_file_map
)

# now print
for metric, info in runnable.items():
    print(f"✓ {metric} → granularity: {info['granularity']} | vars: {info['vars']}")


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

# print(resample_cache)
# lazy_loader = resample_cache[("SSH", "1y")]
# import inspect

# print(inspect.signature(lazy_loader))

# print(resample_cache[("Temperature", "3m")])

# TODO:
# check that the functions are not repeats of each other.
# check evaluations work on real data
# how to account for comparisons?
