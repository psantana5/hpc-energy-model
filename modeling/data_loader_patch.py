
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
