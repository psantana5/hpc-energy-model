# Historical Data Loader for HPC Energy Modeling

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass

from ..utils.config import ModelingConfig, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about loaded dataset"""
    start_time: datetime
    end_time: datetime
    total_jobs: int
    total_nodes: int
    data_points: int
    missing_data_percentage: float
    data_quality_score: float

class HistoricalDataLoader:
    """
    Loads and preprocesses historical data from the HPC energy monitoring system.
    
    This class connects to the TimescaleDB database and extracts job metrics,
    node metrics, and thermal data for use in modeling and simulation.
    """
    
    def __init__(self, config: ModelingConfig):
        self.config = config
        self.db_config = config.database
        self.connection = None
        self.dataset_info = None
        
        # Cached data
        self._job_metrics = None
        self._node_metrics = None
        self._thermal_metrics = None
        self._energy_predictions = None
        
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password
            )
            logger.info("Successfully connected to TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def load_job_metrics(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        job_types: Optional[List[str]] = None,
                        min_duration: Optional[int] = None) -> pd.DataFrame:
        """
        Load job metrics from the database
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            job_types: Filter by job types
            min_duration: Minimum job duration in seconds
            
        Returns:
            DataFrame with job metrics
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError("Cannot connect to database")
        
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.historical_data_days)
        
        # Build query
        query = f"""
        SELECT 
            time,
            job_id,
            job_type,
            node_id,
            user_id,
            partition,
            duration_seconds,
            cpu_cores,
            memory_mb,
            cpu_usage,
            memory_usage,
            io_read_mbps,
            io_write_mbps,
            network_rx_mbps,
            network_tx_mbps,
            avg_cpu_temp,
            peak_cpu_temp,
            avg_gpu_temp,
            peak_gpu_temp,
            thermal_throttling,
            avg_power_watts,
            peak_power_watts,
            estimated_energy_wh,
            job_state,
            exit_code,
            workload_pattern,
            prediction_accuracy
        FROM {self.db_config.schema}.job_metrics
        WHERE time >= %s AND time <= %s
        """
        
        params = [start_date, end_date]
        
        # Add filters
        if job_types:
            query += " AND job_type = ANY(%s)"
            params.append(job_types)
        
        if min_duration:
            query += " AND duration_seconds >= %s"
            params.append(min_duration)
        
        query += " ORDER BY time ASC"
        
        try:
            df = pd.read_sql_query(query, self.connection, params=params)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            logger.info(f"Loaded {len(df)} job metric records")
            self._job_metrics = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading job metrics: {e}")
            raise
    
    def load_node_metrics(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         node_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load node metrics from the database
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            node_ids: Filter by specific node IDs
            
        Returns:
            DataFrame with node metrics
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError("Cannot connect to database")
        
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.historical_data_days)
        
        query = f"""
        SELECT 
            time,
            node_id,
            cpu_usage,
            memory_usage,
            disk_usage,
            load_avg_1m,
            load_avg_5m,
            load_avg_15m,
            cpu_temp,
            gpu_temp,
            system_temp,
            fan_speed_rpm,
            power_consumption_watts,
            energy_consumed_wh,
            network_rx_bytes,
            network_tx_bytes,
            disk_read_bytes,
            disk_write_bytes
        FROM {self.db_config.schema}.node_metrics
        WHERE time >= %s AND time <= %s
        """
        
        params = [start_date, end_date]
        
        if node_ids:
            query += " AND node_id = ANY(%s)"
            params.append(node_ids)
        
        query += " ORDER BY time ASC"
        
        try:
            df = pd.read_sql_query(query, self.connection, params=params)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            logger.info(f"Loaded {len(df)} node metric records")
            self._node_metrics = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading node metrics: {e}")
            raise
    
    def load_energy_predictions(self,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load energy predictions from the database
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            
        Returns:
            DataFrame with energy predictions
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError("Cannot connect to database")
        
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.historical_data_days)
        
        query = f"""
        SELECT 
            time,
            job_id,
            predicted_energy_wh,
            actual_energy_wh,
            prediction_error,
            model_version,
            confidence_score,
            features_used
        FROM {self.db_config.schema}.energy_predictions
        WHERE time >= %s AND time <= %s
        ORDER BY time ASC
        """
        
        try:
            df = pd.read_sql_query(query, self.connection, params=[start_date, end_date])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            logger.info(f"Loaded {len(df)} energy prediction records")
            self._energy_predictions = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading energy predictions: {e}")
            raise
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the loaded dataset
        
        Returns:
            DatasetInfo object with dataset statistics
        """
        if self._job_metrics is None:
            raise ValueError("No job metrics loaded. Call load_job_metrics() first.")
        
        df = self._job_metrics
        
        # Calculate statistics
        start_time = df.index.min()
        end_time = df.index.max()
        total_jobs = df['job_id'].nunique()
        total_nodes = df['node_id'].nunique()
        data_points = len(df)
        
        # Calculate missing data percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        # Calculate data quality score (simple heuristic)
        quality_score = max(0, 100 - missing_percentage - 
                           (df['prediction_accuracy'].isnull().sum() / len(df)) * 10)
        
        self.dataset_info = DatasetInfo(
            start_time=start_time,
            end_time=end_time,
            total_jobs=total_jobs,
            total_nodes=total_nodes,
            data_points=data_points,
            missing_data_percentage=missing_percentage,
            data_quality_score=quality_score
        )
        
        return self.dataset_info
    
    def preprocess_data(self, 
                       fill_missing: bool = True,
                       remove_outliers: bool = True,
                       normalize_features: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Preprocess loaded data for modeling
        
        Args:
            fill_missing: Whether to fill missing values
            remove_outliers: Whether to remove statistical outliers
            normalize_features: Whether to normalize numerical features
            
        Returns:
            Dictionary with preprocessed DataFrames
        """
        processed_data = {}
        
        for name, df in [('job_metrics', self._job_metrics),
                        ('node_metrics', self._node_metrics),
                        ('energy_predictions', self._energy_predictions)]:
            
            if df is None:
                continue
                
            processed_df = df.copy()
            
            # Fill missing values
            if fill_missing:
                # Forward fill for time series data
                processed_df = processed_df.fillna(method='ffill')
                # Backward fill for remaining NaNs
                processed_df = processed_df.fillna(method='bfill')
                # Fill remaining with median for numerical columns
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
                    processed_df[numeric_cols].median()
                )
            
            # Remove outliers using IQR method
            if remove_outliers:
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_df = processed_df[
                        (processed_df[col] >= lower_bound) & 
                        (processed_df[col] <= upper_bound)
                    ]
            
            # Normalize features
            if normalize_features:
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_cols] = (
                    processed_df[numeric_cols] - processed_df[numeric_cols].mean()
                ) / processed_df[numeric_cols].std()
            
            processed_data[name] = processed_df
            logger.info(f"Preprocessed {name}: {len(processed_df)} records")
        
        return processed_data
    
    def export_to_files(self, output_dir: str, format: str = 'parquet'):
        """
        Export loaded data to files
        
        Args:
            output_dir: Output directory path
            format: Export format ('parquet', 'csv', 'json')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        datasets = {
            'job_metrics': self._job_metrics,
            'node_metrics': self._node_metrics,
            'energy_predictions': self._energy_predictions
        }
        
        for name, df in datasets.items():
            if df is None:
                continue
                
            file_path = output_path / f"{name}.{format}"
            
            if format == 'parquet':
                df.to_parquet(file_path)
            elif format == 'csv':
                df.to_csv(file_path)
            elif format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {name} to {file_path}")
        
        # Export dataset info
        if self.dataset_info:
            info_path = output_path / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    'start_time': self.dataset_info.start_time.isoformat(),
                    'end_time': self.dataset_info.end_time.isoformat(),
                    'total_jobs': self.dataset_info.total_jobs,
                    'total_nodes': self.dataset_info.total_nodes,
                    'data_points': self.dataset_info.data_points,
                    'missing_data_percentage': self.dataset_info.missing_data_percentage,
                    'data_quality_score': self.dataset_info.data_quality_score
                }, f, indent=2)
            
            logger.info(f"Exported dataset info to {info_path}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()