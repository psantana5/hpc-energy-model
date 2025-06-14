#!/usr/bin/env python3
"""
Data Analysis Module for HPC Energy Model

Analyzes collected metrics from jobs, thermal sensors, and system monitoring
to identify patterns and correlations for energy consumption prediction.

Author: HPC Energy Model Project
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class HPCEnergyAnalyzer:
    """
    Comprehensive analyzer for HPC energy consumption data
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "analysis_output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # DataFrames for different data types
        self.job_metrics = None
        self.thermal_metrics = None
        self.system_metrics = None
        self.combined_data = None
        
        # Analysis results
        self.correlation_matrix = None
        self.energy_model = None
        self.feature_importance = None
        self.clusters = None
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_timescale_data(self, connection_params: Dict[str, str]) -> bool:
        """
        Load data from TimescaleDB
        """
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(**connection_params)
            
            # Load job metrics
            job_query = """
            SELECT 
                time,
                job_id,
                job_state,
                job_type,
                cpu_usage,
                memory_usage,
                duration_seconds,
                estimated_energy_wh,
                node_id
            FROM job_metrics 
            WHERE time >= NOW() - INTERVAL '7 days'
            ORDER BY time DESC;
            """
            
            self.job_metrics = pd.read_sql(job_query, conn)
            
            # Load thermal metrics
            thermal_query = """
            SELECT 
                time,
                node_id,
                cpu_temp_avg,
                cpu_temp_max,
                gpu_temp_avg,
                gpu_temp_max,
                system_temp,
                fan_speed_avg,
                thermal_throttling
            FROM thermal_metrics 
            WHERE time >= NOW() - INTERVAL '7 days'
            ORDER BY time DESC;
            """
            
            self.thermal_metrics = pd.read_sql(thermal_query, conn)
            
            # Load system metrics
            system_query = """
            SELECT 
                time,
                node_id,
                cpu_usage_percent,
                memory_usage_percent,
                disk_io_read_mbps,
                disk_io_write_mbps,
                network_io_mbps,
                power_consumption_w
            FROM system_metrics 
            WHERE time >= NOW() - INTERVAL '7 days'
            ORDER BY time DESC;
            """
            
            self.system_metrics = pd.read_sql(system_query, conn)
            
            conn.close()
            
            print(f"Loaded {len(self.job_metrics)} job records")
            print(f"Loaded {len(self.thermal_metrics)} thermal records")
            print(f"Loaded {len(self.system_metrics)} system records")
            
            return True
            
        except Exception as e:
            print(f"Error loading data from TimescaleDB: {e}")
            return False
    
    def load_json_data(self, pattern: str = "*.json") -> bool:
        """
        Load data from JSON files (benchmark results)
        """
        try:
            json_files = list(self.data_dir.glob(pattern))
            
            if not json_files:
                print(f"No JSON files found matching pattern: {pattern}")
                return False
            
            job_data = []
            thermal_data = []
            system_data = []
            
            for file_path in json_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract job-level metrics
                if 'benchmark_info' in data and 'performance_metrics' in data:
                    job_record = {
                        'timestamp': data['benchmark_info'].get('start_time'),
                        'job_id': file_path.stem,
                        'job_type': data['benchmark_info'].get('workload_type', 'unknown'),
                        'duration_seconds': data['benchmark_info'].get('duration_actual', 0),
                        'cpu_operations': data['performance_metrics'].get('total_cpu_operations', 0),
                        'io_operations': data['performance_metrics'].get('total_io_operations', 0),
                        'throughput_mbps': data['performance_metrics'].get('total_throughput_mbps', 0),
                        'estimated_energy_wh': self._estimate_energy_consumption(data)
                    }
                    job_data.append(job_record)
                
                # Extract system metrics history
                if 'metrics_history' in data:
                    for metric in data['metrics_history']:
                        if 'error' not in metric:
                            system_record = {
                                'timestamp': metric.get('timestamp'),
                                'cpu_percent': metric.get('cpu_percent', 0),
                                'memory_percent': metric.get('memory_percent', 0),
                                'disk_read_mbps': metric.get('disk_read_bytes', 0) / (1024*1024),
                                'disk_write_mbps': metric.get('disk_write_bytes', 0) / (1024*1024),
                                'job_id': file_path.stem
                            }
                            system_data.append(system_record)
            
            # Convert to DataFrames
            if job_data:
                self.job_metrics = pd.DataFrame(job_data)
                self.job_metrics['timestamp'] = pd.to_datetime(self.job_metrics['timestamp'])
            
            if system_data:
                self.system_metrics = pd.DataFrame(system_data)
                self.system_metrics['timestamp'] = pd.to_datetime(self.system_metrics['timestamp'])
            
            print(f"Loaded {len(job_data)} job records from JSON files")
            print(f"Loaded {len(system_data)} system metric records from JSON files")
            
            return True
            
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return False
    
    def _estimate_energy_consumption(self, benchmark_data: Dict) -> float:
        """
        Estimate energy consumption based on benchmark metrics
        """
        try:
            # Base power consumption estimates (watts)
            base_power = 50  # Idle system power
            cpu_power_per_percent = 2.0  # Additional watts per CPU percent
            io_power_per_mbps = 0.5  # Additional watts per MB/s I/O
            
            duration = benchmark_data['benchmark_info'].get('duration_actual', 0)
            
            if 'system_metrics' in benchmark_data:
                avg_cpu = benchmark_data['system_metrics'].get('avg_cpu_percent', 0)
                io_rate = benchmark_data['system_metrics'].get('disk_read_rate_mbps', 0) + \
                         benchmark_data['system_metrics'].get('disk_write_rate_mbps', 0)
            else:
                avg_cpu = 50  # Assume moderate CPU usage
                io_rate = 10  # Assume moderate I/O
            
            # Calculate average power consumption
            avg_power = base_power + (avg_cpu * cpu_power_per_percent) + (io_rate * io_power_per_mbps)
            
            # Convert to watt-hours
            energy_wh = (avg_power * duration) / 3600
            
            return energy_wh
            
        except Exception:
            return 0.0
    
    def combine_datasets(self) -> bool:
        """
        Combine job, thermal, and system metrics into a unified dataset
        """
        try:
            if self.job_metrics is None:
                print("No job metrics available for combination")
                return False
            
            # Start with job metrics as base
            combined = self.job_metrics.copy()
            
            # Add system metrics if available
            if self.system_metrics is not None:
                # Aggregate system metrics by job_id
                system_agg = self.system_metrics.groupby('job_id').agg({
                    'cpu_percent': ['mean', 'max', 'std'],
                    'memory_percent': ['mean', 'max', 'std'],
                    'disk_read_mbps': ['mean', 'sum'],
                    'disk_write_mbps': ['mean', 'sum']
                }).round(2)
                
                # Flatten column names
                system_agg.columns = ['_'.join(col).strip() for col in system_agg.columns]
                system_agg = system_agg.reset_index()
                
                # Merge with job metrics
                combined = combined.merge(system_agg, on='job_id', how='left')
            
            # Add thermal metrics if available
            if self.thermal_metrics is not None and 'node_id' in combined.columns:
                # Aggregate thermal metrics by node and time window
                thermal_agg = self.thermal_metrics.groupby('node_id').agg({
                    'cpu_temp_avg': ['mean', 'max'],
                    'cpu_temp_max': ['mean', 'max'],
                    'system_temp': ['mean', 'max'],
                    'fan_speed_avg': 'mean',
                    'thermal_throttling': 'sum'
                }).round(2)
                
                thermal_agg.columns = ['_'.join(col).strip() for col in thermal_agg.columns]
                thermal_agg = thermal_agg.reset_index()
                
                # Merge with combined data
                combined = combined.merge(thermal_agg, on='node_id', how='left')
            
            # Add derived features
            combined['cpu_efficiency'] = combined.get('cpu_operations', 0) / \
                                       (combined.get('cpu_percent_mean', 1) * combined.get('duration_seconds', 1))
            
            combined['io_efficiency'] = combined.get('io_operations', 0) / \
                                      (combined.get('disk_read_mbps_mean', 1) + combined.get('disk_write_mbps_mean', 1) + 1)
            
            combined['energy_per_operation'] = combined.get('estimated_energy_wh', 0) / \
                                             (combined.get('cpu_operations', 1) + combined.get('io_operations', 1))
            
            # Handle missing values
            combined = combined.fillna(0)
            
            self.combined_data = combined
            
            print(f"Combined dataset created with {len(combined)} records and {len(combined.columns)} features")
            return True
            
        except Exception as e:
            print(f"Error combining datasets: {e}")
            return False
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between features and energy consumption
        """
        if self.combined_data is None:
            print("No combined data available for correlation analysis")
            return {}
        
        # Select numeric columns
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        self.correlation_matrix = self.combined_data[numeric_cols].corr()
        
        # Find strongest correlations with energy consumption
        if 'estimated_energy_wh' in self.correlation_matrix.columns:
            energy_corr = self.correlation_matrix['estimated_energy_wh'].abs().sort_values(ascending=False)
            energy_corr = energy_corr[energy_corr.index != 'estimated_energy_wh']
        else:
            energy_corr = pd.Series()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot energy correlations
        if not energy_corr.empty:
            plt.figure(figsize=(10, 6))
            energy_corr.head(10).plot(kind='barh')
            plt.title('Top 10 Features Correlated with Energy Consumption')
            plt.xlabel('Absolute Correlation Coefficient')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'energy_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'energy_correlations': energy_corr.to_dict() if not energy_corr.empty else {}
        }
    
    def build_energy_model(self, target_col: str = 'estimated_energy_wh') -> Dict[str, Any]:
        """
        Build predictive models for energy consumption
        """
        if self.combined_data is None or target_col not in self.combined_data.columns:
            print(f"Cannot build model: missing data or target column '{target_col}'")
            return {}
        
        # Prepare features
        feature_cols = [
            'duration_seconds', 'cpu_operations', 'io_operations', 'throughput_mbps',
            'cpu_percent_mean', 'cpu_percent_max', 'memory_percent_mean', 'memory_percent_max',
            'disk_read_mbps_mean', 'disk_write_mbps_mean', 'cpu_efficiency', 'io_efficiency'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in self.combined_data.columns]
        
        if len(available_features) < 3:
            print("Insufficient features for model building")
            return {}
        
        # Prepare data
        X = self.combined_data[available_features].fillna(0)
        y = self.combined_data[target_col]
        
        # Remove rows with zero or negative energy values
        valid_mask = y > 0
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print("Insufficient valid samples for model building")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model
            }
            
            # Feature importance for Random Forest
            if name == 'Random Forest':
                importance = pd.DataFrame({
                    'feature': available_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance = importance
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                importance.head(10).plot(x='feature', y='importance', kind='barh')
                plt.title('Feature Importance for Energy Prediction')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.energy_model = results[best_model_name]['model']
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        for name, result in results.items():
            if name == 'Linear Regression':
                y_pred = result['model'].predict(X_test_scaled)
            else:
                y_pred = result['model'].predict(X_test)
            
            plt.scatter(y_test, y_pred, alpha=0.6, label=f"{name} (R² = {result['r2']:.3f})")
        
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Energy Consumption (Wh)')
        plt.ylabel('Predicted Energy Consumption (Wh)')
        plt.title('Energy Consumption Prediction Results')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'model_results': {k: {metric: v for metric, v in result.items() if metric != 'model'} 
                            for k, result in results.items()},
            'best_model': best_model_name,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else [],
            'features_used': available_features
        }
    
    def cluster_workloads(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster workloads based on resource usage patterns
        """
        if self.combined_data is None:
            print("No data available for clustering")
            return {}
        
        # Select clustering features
        cluster_features = [
            'duration_seconds', 'cpu_operations', 'io_operations',
            'cpu_percent_mean', 'memory_percent_mean', 'estimated_energy_wh'
        ]
        
        available_features = [col for col in cluster_features if col in self.combined_data.columns]
        
        if len(available_features) < 3:
            print("Insufficient features for clustering")
            return {}
        
        # Prepare data
        X = self.combined_data[available_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.combined_data['cluster'] = clusters
        self.clusters = clusters
        
        # Analyze clusters
        cluster_summary = self.combined_data.groupby('cluster')[available_features].agg(['mean', 'std']).round(2)
        
        # Plot clusters (2D projection)
        if len(available_features) >= 2:
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Energy vs Duration
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(self.combined_data['duration_seconds'], 
                                self.combined_data['estimated_energy_wh'], 
                                c=clusters, cmap='viridis', alpha=0.6)
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Energy Consumption (Wh)')
            plt.title('Workload Clusters: Energy vs Duration')
            plt.colorbar(scatter)
            
            # Plot 2: CPU vs I/O operations
            if 'cpu_operations' in available_features and 'io_operations' in available_features:
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(self.combined_data['cpu_operations'], 
                                    self.combined_data['io_operations'], 
                                    c=clusters, cmap='viridis', alpha=0.6)
                plt.xlabel('CPU Operations')
                plt.ylabel('I/O Operations')
                plt.title('Workload Clusters: CPU vs I/O')
                plt.colorbar(scatter)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'workload_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_summary': cluster_summary.to_dict(),
            'features_used': available_features
        }
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights and recommendations based on analysis
        """
        insights = {
            'data_summary': {},
            'energy_patterns': {},
            'optimization_recommendations': [],
            'anomalies': []
        }
        
        if self.combined_data is not None:
            # Data summary
            insights['data_summary'] = {
                'total_jobs': len(self.combined_data),
                'avg_energy_consumption': self.combined_data.get('estimated_energy_wh', pd.Series([0])).mean(),
                'avg_duration': self.combined_data.get('duration_seconds', pd.Series([0])).mean(),
                'job_types': self.combined_data.get('job_type', pd.Series(['unknown'])).value_counts().to_dict()
            }
            
            # Energy patterns
            if 'estimated_energy_wh' in self.combined_data.columns:
                energy_stats = self.combined_data['estimated_energy_wh'].describe()
                insights['energy_patterns'] = {
                    'min_energy': energy_stats['min'],
                    'max_energy': energy_stats['max'],
                    'median_energy': energy_stats['50%'],
                    'energy_std': energy_stats['std']
                }
                
                # Find high energy consumption jobs
                high_energy_threshold = energy_stats['75%'] + 1.5 * (energy_stats['75%'] - energy_stats['25%'])
                high_energy_jobs = self.combined_data[self.combined_data['estimated_energy_wh'] > high_energy_threshold]
                
                if len(high_energy_jobs) > 0:
                    insights['anomalies'].append({
                        'type': 'high_energy_consumption',
                        'count': len(high_energy_jobs),
                        'threshold': high_energy_threshold,
                        'avg_energy': high_energy_jobs['estimated_energy_wh'].mean()
                    })
            
            # Optimization recommendations
            if self.feature_importance is not None:
                top_features = self.feature_importance.head(3)['feature'].tolist()
                insights['optimization_recommendations'].append({
                    'type': 'feature_optimization',
                    'message': f"Focus on optimizing: {', '.join(top_features)}",
                    'features': top_features
                })
            
            if 'cpu_efficiency' in self.combined_data.columns:
                low_efficiency = self.combined_data[self.combined_data['cpu_efficiency'] < 
                                                  self.combined_data['cpu_efficiency'].quantile(0.25)]
                if len(low_efficiency) > 0:
                    insights['optimization_recommendations'].append({
                        'type': 'cpu_efficiency',
                        'message': f"{len(low_efficiency)} jobs show low CPU efficiency",
                        'avg_efficiency': low_efficiency['cpu_efficiency'].mean()
                    })
        
        return insights
    
    def save_analysis_report(self, filename: str = None) -> str:
        """
        Save comprehensive analysis report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hpc_energy_analysis_{timestamp}.json"
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'job_metrics_count': len(self.job_metrics) if self.job_metrics is not None else 0,
                'thermal_metrics_count': len(self.thermal_metrics) if self.thermal_metrics is not None else 0,
                'system_metrics_count': len(self.system_metrics) if self.system_metrics is not None else 0,
                'combined_data_count': len(self.combined_data) if self.combined_data is not None else 0
            },
            'correlations': self.analyze_correlations(),
            'energy_model': self.build_energy_model(),
            'clustering': self.cluster_workloads(),
            'insights': self.generate_insights()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Analysis report saved to: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """
        Print analysis summary
        """
        print("\n" + "="*60)
        print("HPC ENERGY ANALYSIS SUMMARY")
        print("="*60)
        
        if self.combined_data is not None:
            print(f"Total Jobs Analyzed: {len(self.combined_data)}")
            
            if 'estimated_energy_wh' in self.combined_data.columns:
                energy_stats = self.combined_data['estimated_energy_wh'].describe()
                print(f"\nEnergy Consumption Statistics:")
                print(f"  Average: {energy_stats['mean']:.2f} Wh")
                print(f"  Median: {energy_stats['50%']:.2f} Wh")
                print(f"  Range: {energy_stats['min']:.2f} - {energy_stats['max']:.2f} Wh")
                print(f"  Std Dev: {energy_stats['std']:.2f} Wh")
            
            if 'job_type' in self.combined_data.columns:
                job_types = self.combined_data['job_type'].value_counts()
                print(f"\nJob Type Distribution:")
                for job_type, count in job_types.items():
                    print(f"  {job_type}: {count} jobs")
            
            if self.feature_importance is not None:
                print(f"\nTop Energy Predictors:")
                for _, row in self.feature_importance.head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
        
        print(f"\nOutput files saved to: {self.output_dir}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='HPC Energy Data Analysis')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory for analysis outputs')
    parser.add_argument('--data-source', choices=['json', 'timescale'], default='json',
                       help='Data source type')
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='TimescaleDB host')
    parser.add_argument('--db-port', type=int, default=5432,
                       help='TimescaleDB port')
    parser.add_argument('--db-name', type=str, default='hpc_energy',
                       help='TimescaleDB database name')
    parser.add_argument('--db-user', type=str, default='postgres',
                       help='TimescaleDB username')
    parser.add_argument('--db-password', type=str, default='password',
                       help='TimescaleDB password')
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of clusters for workload analysis')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HPCEnergyAnalyzer(args.data_dir, args.output_dir)
    
    # Load data
    if args.data_source == 'timescale':
        connection_params = {
            'host': args.db_host,
            'port': args.db_port,
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password
        }
        success = analyzer.load_timescale_data(connection_params)
    else:
        success = analyzer.load_json_data()
    
    if not success:
        print("Failed to load data. Exiting.")
        return 1
    
    # Combine datasets
    if not analyzer.combine_datasets():
        print("Failed to combine datasets. Exiting.")
        return 1
    
    # Run analysis
    try:
        print("Running correlation analysis...")
        analyzer.analyze_correlations()
        
        print("Building energy prediction models...")
        analyzer.build_energy_model()
        
        print(f"Clustering workloads into {args.clusters} groups...")
        analyzer.cluster_workloads(args.clusters)
        
        print("Generating insights...")
        analyzer.generate_insights()
        
        # Save report
        report_path = analyzer.save_analysis_report()
        
        # Print summary
        analyzer.print_summary()
        
        print(f"\nAnalysis completed successfully!")
        print(f"Report saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == '__main__':
    exit(main())