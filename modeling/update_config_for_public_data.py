#!/usr/bin/env python3
"""
Update modeling configuration to include public datasets.
"""

import yaml
import sys
from pathlib import Path

def update_modeling_config():
    """Update modeling_config.yaml to include public datasets."""
    config_path = Path("modeling_config.yaml")
    
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return False
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add public dataset configuration
    if 'data_sources' not in config:
        config['data_sources'] = {}
    
    config['data_sources']['public_datasets'] = {
        'enabled': True,
        'path': './data/public_datasets/combined_public_data.csv',
        'sample_path': './data/public_datasets/sample_public_data.csv',
        'weight': 0.3,  # Weight for combining with existing data
        'validation_split': 0.2
    }
    
    # Update data loading configuration
    if 'data_loading' not in config:
        config['data_loading'] = {}
    
    config['data_loading']['combine_sources'] = True
    config['data_loading']['normalize_features'] = True
    config['data_loading']['feature_columns'] = [
        'duration_seconds', 'num_nodes', 'cpu_hours', 'memory_gb',
        'estimated_power_w', 'avg_temp_c', 'max_temp_c'
    ]
    
    # Backup original config
    backup_path = config_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Backup created: {backup_path}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated {config_path} with public dataset configuration")
    return True

def create_data_loader_patch():
    """Create a patch for data_loader.py to handle public datasets."""
    patch_content = '''
# Add this method to your DataLoader class in modeling/core/data_loader.py

def load_public_datasets(self, config: dict) -> pd.DataFrame:
    """Load and combine public HPC datasets."""
    public_config = config.get('data_sources', {}).get('public_datasets', {})
    
    if not public_config.get('enabled', False):
        return pd.DataFrame()
    
    data_path = Path(public_config['path'])
    if not data_path.exists():
        self.logger.warning(f"Public dataset not found: {data_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_path)
        
        # Convert datetime columns
        datetime_cols = ['submit_time', 'start_time', 'end_time']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Apply validation split
        if public_config.get('validation_split', 0) > 0:
            split_ratio = public_config['validation_split']
            train_size = int(len(df) * (1 - split_ratio))
            df = df.iloc[:train_size]  # Use first part for training
        
        self.logger.info(f"Loaded {len(df)} records from public datasets")
        return df
        
    except Exception as e:
        self.logger.error(f"Failed to load public datasets: {e}")
        return pd.DataFrame()

def combine_datasets(self, internal_df: pd.DataFrame, public_df: pd.DataFrame, 
                    weight: float = 0.3) -> pd.DataFrame:
    """Combine internal and public datasets with weighting."""
    if public_df.empty:
        return internal_df
    
    if internal_df.empty:
        return public_df
    
    # Sample public data based on weight
    public_sample_size = int(len(internal_df) * weight)
    if len(public_df) > public_sample_size:
        public_df = public_df.sample(n=public_sample_size, random_state=42)
    
    # Align columns
    common_cols = list(set(internal_df.columns) & set(public_df.columns))
    internal_aligned = internal_df[common_cols]
    public_aligned = public_df[common_cols]
    
    # Add source identifier
    internal_aligned = internal_aligned.copy()
    public_aligned = public_aligned.copy()
    internal_aligned['data_source'] = 'internal'
    public_aligned['data_source'] = 'public'
    
    # Combine
    combined_df = pd.concat([internal_aligned, public_aligned], ignore_index=True)
    
    self.logger.info(f"Combined datasets: {len(internal_aligned)} internal + {len(public_aligned)} public = {len(combined_df)} total")
    return combined_df
'''
    
    patch_path = Path("data_loader_patch.py")
    with open(patch_path, 'w') as f:
        f.write(patch_content)
    
    print(f"Data loader patch created: {patch_path}")
    print("Copy the methods from this file to your DataLoader class.")

def main():
    print("=== Updating Configuration for Public Datasets ===")
    
    # Update config
    if update_modeling_config():
        print("✓ Configuration updated")
    else:
        print("✗ Failed to update configuration")
        return
    
    # Create patch
    create_data_loader_patch()
    print("✓ Data loader patch created")
    
    print("\n=== Next Steps ===")
    print("1. Review the updated modeling_config.yaml")
    print("2. Apply the data loader patch to modeling/core/data_loader.py")
    print("3. Run the public dataset download: ./scripts/setup_public_data.sh")
    print("4. Test your modeling pipeline with the new data")

if __name__ == "__main__":
    main()