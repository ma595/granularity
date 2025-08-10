# Climate Data Granularity Analysis

A Python framework for analyzing and processing climate model data at different temporal granularities. This project enables efficient computation of climate metrics across multiple time resolutions with intelligent caching and resampling capabilities.

## 🌊 Overview

This tool is designed for climate scientists working with NEMO ocean model data and similar NetCDF datasets. It automatically determines which metrics can be computed at different temporal granularities (10-day, monthly, 3-monthly, yearly) and provides intelligent resampling between resolutions.

## ✨ Key Features

- **Multi-Granularity Processing**: Handle data at 10d, 1m, 3m, and 1y temporal resolutions
- **Intelligent Analysis**: Automatically determine which metrics can run at each granularity
- **Smart Resampling**: Downsample from finer to coarser resolutions when needed
- **Disk Caching**: Persistent NetCDF caching with hash validation for resampled data
- **Memory Optimization**: Lazy loading with dask to handle large datasets efficiently
- **Flexible Metrics**: Extensible framework for custom climate metrics

## 📁 Project Structure

```
granularity/
├── README.md                           # This file
├── granularity_simple.py              # Main execution script
├── granularity_simple_utils.py        # Core utilities (data loading, caching, metrics)
├── granularity_simple_analysis.py     # Analysis functions (refactored from utils)
├── metrics.py                          # Climate metric definitions
├── standardise_variables.py           # Variable name standardization
├── precomputed_data/                   # Pre-processed NetCDF files
├── resampled_cache/                    # Cached resampled data
└── tests/                              # Test files
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install xarray pandas dask netcdf4
```

### Basic Usage

```python
from granularity_simple_utils import run_metrics_intelligently_with_cache
from metrics import dummy_ACC_Drake_metric_2, dummy_check_density

# Define your metrics and required variables
metric_requirements = {
    "ACC_Drake_metric": ["velocity_u", "ssh"],
    "density_check": ["temperature", "salinity"],
}

metric_functions = {
    "ACC_Drake_metric": dummy_ACC_Drake_metric_2,
    "density_check": dummy_check_density,
}

# Define your data files
variable_file_map = {
    "velocity_u": [
        {"granularity": "10d", "file": "data/velocity_u_10d.nc"},
        {"granularity": "1m", "file": "data/velocity_u_1m.nc"},
    ],
    "ssh": [
        {"granularity": "1m", "file": "data/ssh_1m.nc"},
        {"granularity": "3m", "file": "data/ssh_3m.nc"},
    ],
    # ... more variables
}

# Run intelligent analysis
results, analysis = run_metrics_intelligently_with_cache(
    metric_requirements, 
    metric_functions, 
    variable_file_map,
    disk_cache_dir="./cache",
    save_to_cache=True
)
```

## 🧠 Intelligent Features

### Automatic Granularity Analysis

The system automatically determines:
- Which variables are available at each granularity
- Which metrics can run at each temporal resolution
- Optimal granularities for each metric
- When resampling is needed and feasible

```python
from granularity_simple_analysis import show_availability_summary

# Get comprehensive analysis
analysis = show_availability_summary(variable_file_map, metric_requirements)
```

### Smart Caching System

- **Memory Cache**: In-memory storage for frequently accessed data
- **Disk Cache**: Persistent NetCDF files with metadata and hash validation
- **Cache Invalidation**: Automatic detection of source file changes

## 📊 Supported Granularities

| Granularity | Description | Pandas Frequency |
|-------------|-------------|------------------|
| `10d`       | 10-day      | `10D`           |
| `1m`        | Monthly     | `1ME`           |
| `3m`        | 3-monthly   | `3ME`           |
| `1y`        | Yearly      | `1YE`           |

## 🔧 Core Functions

### Data Loading
- `get_data_optimised_with_cache()`: Enhanced data loading with caching
- `get_data_optimised()`: Basic data loading (backward compatibility)

### Metric Execution
- `run_metrics_intelligently_with_cache()`: Smart metric execution
- `run_all_metrics_with_cache()`: Brute-force all metrics at all granularities
- `run_metric_with_cache()`: Single metric execution

### Analysis Functions
- `analyze_variable_availability()`: Core availability analysis
- `get_maximum_granularity_with_all()`: Find best granularity for most metrics
- `get_optimal_granularities_for_metrics()`: Find finest granularity per metric

## 💾 Caching System

### Memory Cache
```python
cache = {}  # Automatically managed during execution
```

### Disk Cache
```python
# Cached files: variable_source-gran_to_target-gran.nc
# Example: temperature_10d_to_1m.nc
# With metadata: temperature_10d_to_1m_metadata.json
```

### Cache Statistics
```python
from granularity_simple_utils import show_cache_stats
show_cache_stats("./resampled_cache")
```

## 🧪 Example: Climate Metrics

```python
def compute_drake_passage_transport(velocity_u, ssh):
    """Compute transport through Drake Passage"""
    # Your climate science logic here
    return (velocity_u * ssh).sum(dim=['x', 'y'])

def analyze_temperature_anomaly(temperature):
    """Analyze temperature anomalies"""
    climatology = temperature.groupby('time.month').mean()
    return temperature.groupby('time.month') - climatology

# Register metrics
metric_requirements = {
    "drake_transport": ["velocity_u", "ssh"],
    "temp_anomaly": ["temperature"],
}

metric_functions = {
    "drake_transport": compute_drake_passage_transport,
    "temp_anomaly": analyze_temperature_anomaly,
}
```

## 🔄 Resampling Logic

The system follows this resampling strategy:
1. **Direct file**: Use exact granularity if available
2. **Downsample**: Resample from finer to coarser resolution
3. **Cache**: Save resampled data for future use
4. **Validate**: Check source file changes via hash

```
10d → 1m → 3m → 1y
 ↓     ↓     ↓
Can resample to any coarser granularity →
```

## ⚠️ Important Notes

- **Memory Management**: Uses dask for lazy loading to handle large datasets
- **File Validation**: Cache automatically invalidates when source files change
- **Variable Aliases**: Supports different variable naming conventions
- **Error Handling**: Graceful degradation when data/metrics unavailable

## 🛠️ Development

### Running Tests
```bash
python test_refactoring.py
python test_granularity.py
```

### Adding New Metrics
1. Define your metric function in `metrics.py`
2. Add to `metric_requirements` and `metric_functions`
3. Ensure required variables are in your `variable_file_map`

### Variable Standardization
The system handles different variable naming conventions via `standardise_variables.py`:
```python
VARIABLE_ALIASES = {
    "temperature": ["temp", "T", "potential_temperature"],
    "salinity": ["sal", "S", "practical_salinity"],
    # ... more aliases
}
```

## 📈 Performance Tips

1. **Use disk caching** for expensive resampling operations
2. **Lazy evaluation** - results are computed only when needed
3. **Chunking** - Configure dask chunks for your data size
4. **Memory monitoring** - Use `dask.distributed` for large datasets

## 🤝 Contributing

This is a research tool for climate data analysis. Feel free to:
- Add new metrics for your research
- Improve caching strategies  
- Extend to new temporal granularities
- Optimize memory usage

## 📄 Legacy Algorithm Notes

### Original Design Rules
- Compute metrics at minimum possible granularity
- Always downsample (never upsample)
- If granularity not provided, use all available granularities

### Original Algorithm
1. **Preprocessing**: Determine available granularities, check variable availability
2. **Resampling**: Downsample from higher frequency when needed
3. **Evaluation**: Loop over granularities and compute feasible metrics

### Options
```python
options = ["all", "min"]  # Historical: run all granularities vs minimum
```

---

*Built for analyzing NEMO ocean model data and similar climate datasets at multiple temporal resolutions.*
