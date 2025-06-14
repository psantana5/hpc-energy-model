#!/bin/bash

# Public HPC Dataset Setup Script
# Downloads, processes, and integrates public HPC datasets

set -e  # Exit on any error

echo "=== HPC Public Dataset Setup ==="
echo "This script will download and process public HPC datasets for energy modeling."
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if required packages are available
echo "Checking required Python packages..."
python3 -c "import requests, pandas, numpy; print('All required packages are available')" || {
    echo "Some packages are missing. Please install them manually:"
    echo "pip3 install --user requests pandas numpy"
    echo "or use a virtual environment"
    exit 1
}

# Create data directory
echo "Creating data directory..."
mkdir -p data/public_datasets

# Make scripts executable
chmod +x download_public_datasets.py

# Download and process datasets
echo "Downloading and processing public datasets..."
python3 download_public_datasets.py --create-integration

# Run integration
if [ -f "integrate_datasets.py" ]; then
    echo "Integrating datasets..."
    python3 integrate_datasets.py
fi

echo
echo "=== Setup Complete ==="
echo "Public datasets have been downloaded and processed."
echo "Check data/public_datasets/ for the following files:"
echo "  - *_cleaned.csv (individual datasets)"
echo "  - combined_public_data.csv (all datasets combined)"
echo "  - sample_public_data.csv (sample for testing)"
echo "  - dataset_summary.md (summary report)"
echo
echo "To use in your modeling pipeline:"
echo "  1. Update modeling_config.yaml to include public data paths"
echo "  2. Modify data_loader.py to load the new datasets"
echo "  3. Run your modeling pipeline with the enhanced data"