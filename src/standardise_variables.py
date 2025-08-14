"""Lookup table for mapping standard variables with different variants."""

# VARIABLE_ALIASES = {
#     "temperature": [
#         "toce",
#         "toce_inst",
#         "tn",
#         "temperature",
#     ],  # Example: can be called 'toce' or 'tn' in various datasets
#     "velocity_u": [
#         "un",
#         "u",
#         "uoce",
#         "uoce_inst",
#         "zonal_velocity",
#     ],  # Zonal velocity can be 'un', 'u', or 'zonal_velocity'
#     "velocity_v": [
#         "vn",
#         "v",
#         "voce",
#         "voce_inst",
#         "meridional_velocity",
#     ],  # Meridional velocity
#     "depth": [
#         "depth",
#         "nav_lev",
#         "deptht",
#         "depthu",
#         "depthv",
#     ],  # Depth can be 'depth', 'nav_lev', or 'deptht'
#     "latitude": ["nav_lat", "y"],  # Latitude can be 'nav_lat' or 'y'
#     "longitude": ["nav_lon", "x"],  # Longitude can be 'nav_lon' or 'x'
#     "ssh": ["sshn", "ssh"],  # Sea surface height could be 'sshn' or 'ssh'
#     "time": [
#         "time_counter",
#         "time",
#     ],  # Time variable may be 'time_counter' or 'time'
#     "salinity": ["so", "sn", "soce", "soce_inst", "salinity"],  # Example for salinity variable
#     "density": ["rhop", "density"],  # Density might be named 'rhop' or 'density'
#     # Add more mappings if necessary depending on your datasets
# }

VARIABLE_ALIASES = {
    "temperature": ["toce", "toce_inst", "tn", "temperature"],
    "velocity_u": ["un", "u", "uoce", "uoce_inst", "zonal_velocity"],
    "velocity_v": ["vn", "v", "voce", "voce_inst", "meridional_velocity"],
    "depth": ["depth", "nav_lev", "deptht", "depthu", "depthv"],
    "nav_lat": ["nav_lat", "y", "latitude"],    # Standardize TO nav_lat
    "nav_lon": ["nav_lon", "x", "longitude"],   # Standardize TO nav_lon
    "ssh": ["sshn", "ssh"],
    "time": ["time_counter", "time"],
    "salinity": ["so", "sn", "soce", "soce_inst", "salinity"],
    "density": ["rhop", "density"],
}


def standardize_variables(dataset, variable_dict):
    """
    Standardizes variable, coordinate, and dimension names across datasets.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing variables, coordinates, and dimensions.
    variable_dict : dict
        A dictionary mapping standardized names to possible alternatives.

    Returns
    -------
    xarray.Dataset
        The dataset with standardized names.
    """
    rename_map = {}


    for standard_name, aliases in variable_dict.items():
        # Skip if standard name already exists
        if (standard_name in dataset.variables or 
            standard_name in dataset.coords or 
            standard_name in dataset.dims):
            continue
            
        for alias in aliases:
            if alias in dataset.variables:
                rename_map[alias] = standard_name
                break
            elif alias in dataset.coords:
                rename_map[alias] = standard_name
                break
            elif alias in dataset.dims:
                rename_map[alias] = standard_name
                break

    print(f"DEBUG: Rename map = {rename_map}")  # Add this to see what's happening
    
    if rename_map:
        dataset = dataset.rename(rename_map)

    # Explicitly handle renaming of 'x' and 'y' dimensions to 'nav_lon' and 'nav_lat'
    if "x" in dataset.dims:
        dataset = dataset.rename({"x": "nav_lon"})
    if "y" in dataset.dims:
        dataset = dataset.rename({"y": "nav_lat"})

    return dataset
