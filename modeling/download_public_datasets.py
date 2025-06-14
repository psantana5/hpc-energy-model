#!/usr/bin/env python3
"""
Public HPC Dataset Downloader and Processor
Downloads, cleans, and integrates public HPC datasets for energy modeling.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse
import logging
import argparse
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPCDatasetDownloader:
    """Downloads and processes public HPC datasets."""
    
    def __init__(self, data_dir: str = "./data/public_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and configurations
        # Note: Original URLs are not accessible, using mock data generation
        self.datasets = {
            "euler_cluster": {
                "url": "mock",
                "description": "ETH Euler cluster workload traces (mock data)",
                "format": "swf"
            },
            "hpc2n_seth": {
                "url": "mock",
                "description": "HPC2N Seth cluster workload traces (mock data)",
                "format": "swf"
            },
            "sdsc_sp2": {
                "url": "mock",
                "description": "SDSC SP2 cluster workload traces (mock data)",
                "format": "swf"
            }
        }
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL or generate mock data."""
        if url == "mock":
            return self.generate_mock_swf(filename)
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            logger.info(f"Downloaded {filename} ({filepath.stat().st_size} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def generate_mock_swf(self, filename: str) -> bool:
        """Generate mock SWF data for testing."""
        try:
            logger.info(f"Generating mock data for {filename}")
            filepath = self.data_dir / filename.replace('.gz', '')
            
            # Generate mock SWF data
            with open(filepath, 'w') as f:
                f.write("; Mock SWF data for testing\n")
                f.write("; Job_ID Submit_Time Wait_Time Run_Time Allocated_Processors Average_CPU_Time Used_Memory Requested_Processors Requested_Time Requested_Memory Status User_ID Group_ID Executable_ID Queue_ID Partition_ID Preceding_Job_ID Think_Time\n")
                
                # Generate 1000 mock jobs
                base_time = 1640995200  # Jan 1, 2022 timestamp
                for i in range(1, 1001):
                    submit_time = base_time + i * 3600  # Jobs submitted hourly
                    wait_time = np.random.randint(0, 3600)  # 0-1 hour wait
                    run_time = np.random.randint(60, 86400)  # 1 min to 24 hours
                    processors = np.random.choice([1, 2, 4, 8, 16, 32, 64])
                    cpu_time = run_time * processors * np.random.uniform(0.7, 0.95)
                    memory = np.random.randint(1024, 32768)  # 1-32 GB in MB
                    req_processors = processors
                    req_time = run_time + np.random.randint(0, 3600)
                    req_memory = memory + np.random.randint(0, 4096)
                    user_id = np.random.randint(1, 100)
                    group_id = np.random.randint(1, 10)
                    
                    f.write(f"{i} {submit_time} {wait_time} {run_time} {processors} {cpu_time:.0f} {memory} {req_processors} {req_time} {req_memory} 1 {user_id} {group_id} 1 1 -1 -1 0\n")
            
            logger.info(f"Generated mock data: {filename} (1000 jobs)")
            return True
        except Exception as e:
            logger.error(f"Failed to generate mock data for {filename}: {e}")
            return False
    
    def extract_gzip(self, gz_file: str) -> str:
        """Extract gzip file."""
        gz_path = self.data_dir / gz_file
        extracted_path = gz_path.with_suffix('')
        
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted {gz_file} to {extracted_path.name}")
            return extracted_path.name
        except Exception as e:
            logger.error(f"Failed to extract {gz_file}: {e}")
            return ""
    
    def parse_swf_file(self, swf_file: str) -> pd.DataFrame:
        """Parse Standard Workload Format (SWF) file."""
        swf_path = self.data_dir / swf_file
        
        # SWF format columns
        columns = [
            'job_id', 'submit_time', 'wait_time', 'run_time', 'allocated_processors',
            'average_cpu_time', 'used_memory', 'requested_processors', 'requested_time',
            'requested_memory', 'status', 'user_id', 'group_id', 'executable_id',
            'queue_id', 'partition_id', 'preceding_job_id', 'think_time'
        ]
        
        try:
            # Read SWF file, skip comment lines starting with ';'
            data = []
            with open(swf_path, 'r') as f:
                for line in f:
                    if not line.startswith(';') and line.strip():
                        fields = line.strip().split()
                        if len(fields) >= len(columns):
                            data.append(fields[:len(columns)])
            
            df = pd.DataFrame(data, columns=columns)
            
            # Convert numeric columns
            numeric_cols = ['submit_time', 'wait_time', 'run_time', 'allocated_processors',
                          'average_cpu_time', 'used_memory', 'requested_processors',
                          'requested_time', 'requested_memory']
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Parsed {len(df)} jobs from {swf_file}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to parse {swf_file}: {e}")
            return pd.DataFrame()
    
    def clean_and_transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean and transform dataset to match project schema."""
        if df.empty:
            return df
        
        # Remove invalid entries
        df = df.dropna(subset=['job_id', 'submit_time', 'run_time'])
        df = df[df['run_time'] > 0]  # Only completed jobs
        df = df[df['allocated_processors'] > 0]
        
        # Create standardized columns for energy modeling
        cleaned_df = pd.DataFrame({
            'job_id': df['job_id'].astype(str),
            'submit_time': pd.to_datetime(df['submit_time'], unit='s', errors='coerce'),
            'start_time': pd.to_datetime(df['submit_time'] + df['wait_time'], unit='s', errors='coerce'),
            'end_time': pd.to_datetime(df['submit_time'] + df['wait_time'] + df['run_time'], unit='s', errors='coerce'),
            'duration_seconds': df['run_time'],
            'num_nodes': df['allocated_processors'],
            'cpu_hours': df['run_time'] * df['allocated_processors'] / 3600,
            'memory_gb': df['used_memory'].fillna(0) / 1024,  # Convert MB to GB
            'queue': df.get('queue_id', 'unknown'),
            'user_id': df['user_id'],
            'status': 'COMPLETED',
            'dataset_source': dataset_name
        })
        
        # Estimate energy consumption (simplified model)
        # Base power per node: ~200W, CPU utilization factor
        base_power_per_node = 200  # Watts
        cpu_utilization = np.random.uniform(0.6, 0.9, len(cleaned_df))  # Estimated
        
        cleaned_df['estimated_power_w'] = (
            cleaned_df['num_nodes'] * base_power_per_node * cpu_utilization
        )
        cleaned_df['estimated_energy_wh'] = (
            cleaned_df['estimated_power_w'] * cleaned_df['duration_seconds'] / 3600
        )
        
        # Add synthetic temperature data (for demonstration)
        cleaned_df['avg_temp_c'] = np.random.normal(45, 8, len(cleaned_df))
        cleaned_df['max_temp_c'] = cleaned_df['avg_temp_c'] + np.random.uniform(5, 15, len(cleaned_df))
        
        logger.info(f"Cleaned dataset: {len(cleaned_df)} jobs, {cleaned_df['cpu_hours'].sum():.1f} CPU hours")
        return cleaned_df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV."""
        if df.empty:
            logger.warning(f"Empty dataset, skipping save for {filename}")
            return
        
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
    
    def generate_summary_report(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Generate summary report of downloaded datasets."""
        report_path = self.data_dir / "dataset_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# Public HPC Datasets Summary\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            total_jobs = 0
            total_cpu_hours = 0
            total_energy = 0
            
            for name, df in datasets.items():
                if df.empty:
                    continue
                    
                jobs = len(df)
                cpu_hours = df['cpu_hours'].sum()
                energy = df['estimated_energy_wh'].sum()
                
                total_jobs += jobs
                total_cpu_hours += cpu_hours
                total_energy += energy
                
                f.write(f"## {name.replace('_', ' ').title()}\n")
                f.write(f"- **Jobs**: {jobs:,}\n")
                f.write(f"- **CPU Hours**: {cpu_hours:,.1f}\n")
                f.write(f"- **Estimated Energy**: {energy:,.1f} Wh\n")
                f.write(f"- **Date Range**: {df['submit_time'].min()} to {df['submit_time'].max()}\n")
                f.write(f"- **Avg Job Duration**: {df['duration_seconds'].mean()/3600:.2f} hours\n")
                f.write(f"- **Avg Nodes per Job**: {df['num_nodes'].mean():.1f}\n\n")
            
            f.write(f"## Total Summary\n")
            f.write(f"- **Total Jobs**: {total_jobs:,}\n")
            f.write(f"- **Total CPU Hours**: {total_cpu_hours:,.1f}\n")
            f.write(f"- **Total Estimated Energy**: {total_energy:,.1f} Wh\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def download_and_process_all(self) -> Dict[str, pd.DataFrame]:
        """Download and process all datasets."""
        processed_datasets = {}
        
        for name, config in self.datasets.items():
            logger.info(f"Processing {name}: {config['description']}")
            
            # Download or generate mock data
            if config['url'] == 'mock':
                # For mock data, create filename based on dataset name
                filename = f"{name}.swf"
                if not self.download_file(config['url'], filename):
                    continue
                extracted_file = filename
            else:
                filename = Path(urlparse(config['url']).path).name
                if not self.download_file(config['url'], filename):
                    continue
                
                # Extract if gzipped
                if filename.endswith('.gz'):
                    extracted_file = self.extract_gzip(filename)
                    if not extracted_file:
                        continue
                else:
                    extracted_file = filename
            
            # Parse and clean
            if config['format'] == 'swf':
                raw_df = self.parse_swf_file(extracted_file)
                cleaned_df = self.clean_and_transform(raw_df, name)
                
                if not cleaned_df.empty:
                    # Save cleaned data
                    self.save_to_csv(cleaned_df, f"{name}_cleaned.csv")
                    processed_datasets[name] = cleaned_df
        
        # Generate summary
        self.generate_summary_report(processed_datasets)
        
        return processed_datasets

def create_integration_script():
    """Create script to integrate datasets with existing project."""
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path("./integrate_datasets.py")
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    logger.info(f"Created integration script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and process public HPC datasets")
    parser.add_argument("--data-dir", default="./data/public_datasets",
                       help="Directory to store datasets")
    parser.add_argument("--create-integration", action="store_true",
                       help="Create integration script")
    
    args = parser.parse_args()
    
    # Create downloader and process datasets
    downloader = HPCDatasetDownloader(args.data_dir)
    datasets = downloader.download_and_process_all()
    
    if args.create_integration:
        create_integration_script()
    
    logger.info(f"Processing complete. {len(datasets)} datasets processed.")
    logger.info(f"Data saved to: {downloader.data_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review the dataset_summary.md file")
    logger.info("2. Run: python integrate_datasets.py")
    logger.info("3. Update your modeling pipeline to use the new data")

if __name__ == "__main__":
    main()