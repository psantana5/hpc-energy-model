#!/usr/bin/env python3
"""
Integrate public datasets with existing HPC energy modeling project.
"""

import pandas as pd
import sys
from pathlib import Path

def integrate_datasets():
    """Integrate public datasets with project data."""
    data_dir = Path("./data/public_datasets")
    
    # Find all cleaned CSV files
    csv_files = list(data_dir.glob("*_cleaned.csv"))
    
    if not csv_files:
        print("No cleaned datasets found. Run download_public_datasets.py first.")
        return
    
    # Combine all datasets
    combined_df = pd.DataFrame()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f"Added {len(df)} jobs from {csv_file.name}")
    
    # Save combined dataset
    output_file = data_dir / "combined_public_data.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined dataset saved: {output_file}")
    print(f"Total jobs: {len(combined_df):,}")
    print(f"Total CPU hours: {combined_df['cpu_hours'].sum():,.1f}")
    print(f"Total estimated energy: {combined_df['estimated_energy_wh'].sum():,.1f} Wh")
    
    # Create sample for testing
    sample_df = combined_df.sample(n=min(1000, len(combined_df)), random_state=42)
    sample_file = data_dir / "sample_public_data.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"Sample dataset (1000 jobs) saved: {sample_file}")

if __name__ == "__main__":
    integrate_datasets()
