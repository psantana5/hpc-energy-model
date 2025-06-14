#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO)

from modeling.main import HPCModelingPipeline

pipeline = HPCModelingPipeline('modeling/modeling_config.yaml')

print("Loading historical data...")
data = pipeline.load_historical_data()

print("\n=== DATA SUMMARY ===")
print(f"Job metrics shape: {data['job_metrics'].shape if data['job_metrics'] is not None else 'None'}")
print(f"Node metrics shape: {data['node_metrics'].shape if data['node_metrics'] is not None else 'None'}")
print(f"Energy predictions shape: {data['energy_predictions'].shape if data['energy_predictions'] is not None else 'None'}")

if data['job_metrics'] is not None:
    print(f"\nJob metrics columns: {list(data['job_metrics'].columns)}")
    print(f"Job metrics sample:\n{data['job_metrics'].head()}")

if data['node_metrics'] is not None:
    print(f"\nNode metrics columns: {list(data['node_metrics'].columns)}")
    print(f"Node metrics sample:\n{data['node_metrics'].head()}")

if data['energy_predictions'] is not None:
    print(f"\nEnergy predictions columns: {list(data['energy_predictions'].columns)}")
    print(f"Energy predictions sample:\n{data['energy_predictions'].head()}")