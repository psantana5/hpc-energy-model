# Public HPC Datasets Integration

This directory contains scripts to download, process, and integrate public HPC datasets into your energy modeling project.

## Overview

The scripts download workload traces from the Parallel Workloads Archive (PWA) and other sources, clean the data, and prepare it for integration with your existing HPC energy modeling pipeline.

## Available Datasets

1. **ETH Euler Cluster** - Production workload traces from ETH Zurich
2. **HPC2N Seth** - Historical traces from Swedish National Infrastructure
3. **SDSC SP2** - IBM SP2 cluster traces from San Diego Supercomputer Center

## Quick Start

### 1. Download and Process All Datasets

```bash
# Run the complete setup (recommended)
./scripts/setup_public_data.sh
```

### 2. Manual Step-by-Step Process

```bash
# Download and process datasets
python3 scripts/download_public_datasets.py --create-integration

# Integrate datasets
python3 scripts/integrate_datasets.py

# Update configuration
python3 scripts/update_config_for_public_data.py
```

## Script Details

### `download_public_datasets.py`

**Purpose**: Downloads, parses, and cleans public HPC datasets

**Features**:
- Downloads SWF (Standard Workload Format) files
- Parses job traces with timing, resource usage, and metadata
- Estimates energy consumption using simplified models
- Generates synthetic temperature data
- Creates cleaned CSV files for each dataset

**Usage**:
```bash
python3 scripts/download_public_datasets.py [options]

Options:
  --data-dir DIR          Directory to store datasets (default: ./data/public_datasets)
  --create-integration    Create integration script
```

**Output Files**:
- `{dataset}_cleaned.csv` - Individual cleaned datasets
- `dataset_summary.md` - Summary report with statistics

### `integrate_datasets.py`

**Purpose**: Combines all cleaned datasets into unified files

**Output Files**:
- `combined_public_data.csv` - All datasets combined
- `sample_public_data.csv` - Sample subset for testing (1000 jobs)

### `update_config_for_public_data.py`

**Purpose**: Updates your modeling configuration to use public datasets

**Features**:
- Modifies `modeling_config.yaml` to include public data sources
- Creates backup of original configuration
- Generates data loader patch for integration

### `setup_public_data.sh`

**Purpose**: One-command setup script

**Features**:
- Installs required Python packages
- Runs all processing scripts in sequence
- Provides setup completion summary

## Data Schema

The cleaned datasets use a standardized schema:

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | string | Unique job identifier |
| `submit_time` | datetime | Job submission timestamp |
| `start_time` | datetime | Job start timestamp |
| `end_time` | datetime | Job completion timestamp |
| `duration_seconds` | int | Job runtime in seconds |
| `num_nodes` | int | Number of allocated nodes |
| `cpu_hours` | float | Total CPU hours consumed |
| `memory_gb` | float | Memory usage in GB |
| `queue` | string | Queue/partition name |
| `user_id` | string | User identifier |
| `status` | string | Job status (COMPLETED) |
| `estimated_power_w` | float | Estimated power consumption (W) |
| `estimated_energy_wh` | float | Estimated energy consumption (Wh) |
| `avg_temp_c` | float | Average temperature (°C) |
| `max_temp_c` | float | Maximum temperature (°C) |
| `dataset_source` | string | Source dataset identifier |

## Energy Estimation Model

The scripts use a simplified energy estimation model:

- **Base Power**: 200W per node
- **CPU Utilization**: Random factor (0.6-0.9)
- **Power = Nodes × Base Power × CPU Utilization**
- **Energy = Power × Duration**

> **Note**: These are estimates for demonstration. For production use, calibrate with actual measurements.

## Integration with Existing Pipeline

### 1. Configuration Updates

The `update_config_for_public_data.py` script adds this to your `modeling_config.yaml`:

```yaml
data_sources:
  public_datasets:
    enabled: true
    path: '../data/public_datasets/combined_public_data.csv'
    sample_path: '../data/public_datasets/sample_public_data.csv'
    weight: 0.3  # 30% public data, 70% internal data
    validation_split: 0.2

data_loading:
  combine_sources: true
  normalize_features: true
  feature_columns:
    - duration_seconds
    - num_nodes
    - cpu_hours
    - memory_gb
    - estimated_power_w
    - avg_temp_c
    - max_temp_c
```

### 2. Data Loader Modifications

Add the methods from `scripts/data_loader_patch.py` to your `DataLoader` class:

- `load_public_datasets()` - Load public dataset files
- `combine_datasets()` - Merge internal and public data with weighting

### 3. Model Training

Update your training pipeline to:

1. Load both internal and public datasets
2. Combine them with appropriate weighting
3. Normalize features across combined dataset
4. Train models on the enhanced dataset

## Validation and Quality

### Dataset Statistics

After processing, check `data/public_datasets/dataset_summary.md` for:

- Number of jobs per dataset
- Total CPU hours and estimated energy
- Date ranges and job characteristics
- Average job duration and node usage

### Quality Checks

```python
# Load and inspect the data
import pandas as pd

df = pd.read_csv('data/public_datasets/combined_public_data.csv')

# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Validate energy estimates
print(f"Energy range: {df['estimated_energy_wh'].min():.1f} - {df['estimated_energy_wh'].max():.1f} Wh")
```

## Troubleshooting

### Common Issues

1. **Download Failures**
   - Check internet connection
   - Verify URLs are accessible
   - Some datasets may be temporarily unavailable

2. **Parsing Errors**
   - SWF format variations between datasets
   - Check log output for specific parsing issues

3. **Memory Issues**
   - Large datasets may require significant RAM
   - Process datasets individually if needed

4. **Integration Problems**
   - Ensure column names match between datasets
   - Check data types and formats
   - Verify file paths in configuration

### Debug Mode

```bash
# Run with verbose logging
python3 scripts/download_public_datasets.py --data-dir ./debug_data 2>&1 | tee download.log
```

## Extending the Scripts

### Adding New Datasets

1. Add dataset configuration to `HPCDatasetDownloader.datasets`:

```python
"new_dataset": {
    "url": "https://example.com/dataset.swf.gz",
    "description": "Description of the dataset",
    "format": "swf"
}
```

2. Implement custom parsing if needed
3. Update cleaning and transformation logic

### Custom Energy Models

Modify the `clean_and_transform()` method to use:

- Hardware-specific power models
- Workload-dependent efficiency curves
- Temperature-dependent power scaling
- Real measurement data for calibration

## References

- [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/)
- [Standard Workload Format](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)
- [HPC Energy Efficiency Research](https://www.top500.org/green500/)

## Support

For issues or questions:

1. Check the generated log files
2. Review the dataset summary report
3. Validate your configuration files
4. Test with sample data first