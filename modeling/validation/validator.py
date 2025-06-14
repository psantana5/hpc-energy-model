# Model Validation and Comparison Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

from ..utils.config import ModelingConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Validation metrics for model comparison"""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2: float  # R-squared
    correlation: float  # Pearson correlation
    ks_statistic: float  # Kolmogorov-Smirnov test statistic
    ks_pvalue: float  # KS test p-value
    distribution_similarity: float  # Custom distribution similarity score

@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: datetime
    model_name: str
    validation_type: str
    overall_score: float
    energy_metrics: ValidationMetrics
    thermal_metrics: ValidationMetrics
    performance_metrics: ValidationMetrics
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    plots_generated: List[str]

class ModelValidator:
    """
    Validates simulation results against real historical data
    
    This class provides comprehensive validation capabilities including:
    - Statistical comparison of energy consumption
    - Thermal behavior validation
    - Performance metrics comparison
    - Distribution analysis
    - Visual comparison plots
    """
    
    def __init__(self, config: ModelingConfig):
        self.config = config
        self.validation_results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def validate_energy_consumption(self, 
                                  real_data: pd.DataFrame, 
                                  simulated_data: pd.DataFrame,
                                  time_column: str = 'time',
                                  energy_column: str = 'energy_wh') -> ValidationMetrics:
        """
        Validate energy consumption predictions
        
        Args:
            real_data: Real energy consumption data
            simulated_data: Simulated energy consumption data
            time_column: Name of time column
            energy_column: Name of energy column
            
        Returns:
            ValidationMetrics object with energy validation results
        """
        logger.info("Validating energy consumption predictions")
        
        # Align data by time
        aligned_real, aligned_sim = self._align_time_series(
            real_data, simulated_data, time_column, energy_column
        )
        
        if len(aligned_real) == 0:
            logger.warning("No overlapping time periods found for energy validation")
            return self._empty_validation_metrics()
        
        # Calculate validation metrics
        metrics = self._calculate_validation_metrics(aligned_real, aligned_sim)
        
        logger.info(f"Energy validation completed. R² = {metrics.r2:.3f}, MAPE = {metrics.mape:.1f}%")
        return metrics
    
    def validate_thermal_behavior(self, 
                                real_data: pd.DataFrame, 
                                simulated_data: pd.DataFrame,
                                time_column: str = 'time',
                                temp_column: str = 'temperature') -> ValidationMetrics:
        """
        Validate thermal behavior predictions
        
        Args:
            real_data: Real temperature data
            simulated_data: Simulated temperature data
            time_column: Name of time column
            temp_column: Name of temperature column
            
        Returns:
            ValidationMetrics object with thermal validation results
        """
        logger.info("Validating thermal behavior predictions")
        
        # Align data by time
        aligned_real, aligned_sim = self._align_time_series(
            real_data, simulated_data, time_column, temp_column
        )
        
        if len(aligned_real) == 0:
            logger.warning("No overlapping time periods found for thermal validation")
            return self._empty_validation_metrics()
        
        # Calculate validation metrics
        metrics = self._calculate_validation_metrics(aligned_real, aligned_sim)
        
        logger.info(f"Thermal validation completed. R² = {metrics.r2:.3f}, MAPE = {metrics.mape:.1f}%")
        return metrics
    
    def validate_job_performance(self, 
                               real_jobs: pd.DataFrame, 
                               simulated_jobs: pd.DataFrame,
                               duration_column: str = 'duration') -> ValidationMetrics:
        """
        Validate job performance predictions
        
        Args:
            real_jobs: Real job performance data
            simulated_jobs: Simulated job performance data
            duration_column: Name of duration column
            
        Returns:
            ValidationMetrics object with performance validation results
        """
        logger.info("Validating job performance predictions")
        
        # Align jobs by job_id if available, otherwise by index
        if 'job_id' in real_jobs.columns and 'job_id' in simulated_jobs.columns:
            aligned_real, aligned_sim = self._align_by_job_id(
                real_jobs, simulated_jobs, duration_column
            )
        else:
            # Align by position (assuming same order)
            min_len = min(len(real_jobs), len(simulated_jobs))
            aligned_real = real_jobs[duration_column].iloc[:min_len].values
            aligned_sim = simulated_jobs[duration_column].iloc[:min_len].values
        
        if len(aligned_real) == 0:
            logger.warning("No matching jobs found for performance validation")
            return self._empty_validation_metrics()
        
        # Calculate validation metrics
        metrics = self._calculate_validation_metrics(aligned_real, aligned_sim)
        
        logger.info(f"Performance validation completed. R² = {metrics.r2:.3f}, MAPE = {metrics.mape:.1f}%")
        return metrics
    
    def comprehensive_validation(self, 
                               real_data: Dict[str, pd.DataFrame], 
                               simulated_data: Dict[str, pd.DataFrame],
                               model_name: str = "HPC_Simulation") -> ValidationReport:
        """
        Perform comprehensive validation across all metrics
        
        Args:
            real_data: Dictionary with real data DataFrames
            simulated_data: Dictionary with simulated data DataFrames
            model_name: Name of the model being validated
            
        Returns:
            ValidationReport with comprehensive results
        """
        logger.info(f"Starting comprehensive validation for {model_name}")
        
        # Initialize validation metrics
        energy_metrics = self._empty_validation_metrics()
        thermal_metrics = self._empty_validation_metrics()
        performance_metrics = self._empty_validation_metrics()
        
        # Validate energy consumption
        if 'energy' in real_data and 'energy' in simulated_data:
            energy_metrics = self.validate_energy_consumption(
                real_data['energy'], simulated_data['energy']
            )
        
        # Validate thermal behavior
        if 'thermal' in real_data and 'thermal' in simulated_data:
            thermal_metrics = self.validate_thermal_behavior(
                real_data['thermal'], simulated_data['thermal']
            )
        elif 'node_metrics' in real_data and 'node_metrics' in simulated_data:
            thermal_metrics = self.validate_thermal_behavior(
                real_data['node_metrics'], simulated_data['node_metrics'],
                temp_column='cpu_temp'
            )
        
        # Validate job performance
        if 'jobs' in real_data and 'jobs' in simulated_data:
            performance_metrics = self.validate_job_performance(
                real_data['jobs'], simulated_data['jobs']
            )
        elif 'job_metrics' in real_data and 'job_metrics' in simulated_data:
            performance_metrics = self.validate_job_performance(
                real_data['job_metrics'], simulated_data['job_metrics'],
                duration_column='duration_seconds'
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            energy_metrics, thermal_metrics, performance_metrics
        )
        
        # Perform detailed analysis
        detailed_analysis = self._detailed_analysis(
            real_data, simulated_data, energy_metrics, thermal_metrics, performance_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            energy_metrics, thermal_metrics, performance_metrics, detailed_analysis
        )
        
        # Create validation report
        report = ValidationReport(
            timestamp=datetime.now(),
            model_name=model_name,
            validation_type="comprehensive",
            overall_score=overall_score,
            energy_metrics=energy_metrics,
            thermal_metrics=thermal_metrics,
            performance_metrics=performance_metrics,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            plots_generated=[]
        )
        
        logger.info(f"Comprehensive validation completed. Overall score: {overall_score:.2f}")
        return report
    
    def generate_validation_plots(self, 
                                real_data: Dict[str, pd.DataFrame], 
                                simulated_data: Dict[str, pd.DataFrame],
                                output_dir: str) -> List[str]:
        """
        Generate validation plots comparing real vs simulated data
        
        Args:
            real_data: Dictionary with real data DataFrames
            simulated_data: Dictionary with simulated data DataFrames
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_plots = []
        
        # Energy consumption comparison
        if 'energy' in real_data and 'energy' in simulated_data:
            plot_path = self._plot_energy_comparison(
                real_data['energy'], simulated_data['energy'], output_path
            )
            generated_plots.append(plot_path)
        
        # Thermal behavior comparison
        if 'node_metrics' in real_data and 'node_metrics' in simulated_data:
            plot_path = self._plot_thermal_comparison(
                real_data['node_metrics'], simulated_data['node_metrics'], output_path
            )
            generated_plots.append(plot_path)
        
        # Job performance comparison
        if 'job_metrics' in real_data and 'job_metrics' in simulated_data:
            plot_path = self._plot_performance_comparison(
                real_data['job_metrics'], simulated_data['job_metrics'], output_path
            )
            generated_plots.append(plot_path)
        
        # Distribution comparison
        plot_path = self._plot_distribution_comparison(
            real_data, simulated_data, output_path
        )
        generated_plots.append(plot_path)
        
        logger.info(f"Generated {len(generated_plots)} validation plots")
        return generated_plots
    
    def export_validation_report(self, report: ValidationReport, output_dir: str):
        """
        Export validation report to files
        
        Args:
            report: ValidationReport to export
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export JSON report
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'model_name': report.model_name,
            'validation_type': report.validation_type,
            'overall_score': report.overall_score,
            'energy_metrics': self._metrics_to_dict(report.energy_metrics),
            'thermal_metrics': self._metrics_to_dict(report.thermal_metrics),
            'performance_metrics': self._metrics_to_dict(report.performance_metrics),
            'detailed_analysis': report.detailed_analysis,
            'recommendations': report.recommendations,
            'plots_generated': report.plots_generated
        }
        
        with open(output_path / 'validation_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Export summary CSV
        summary_data = {
            'Metric': ['Energy R²', 'Energy MAPE (%)', 'Thermal R²', 'Thermal MAPE (%)', 
                      'Performance R²', 'Performance MAPE (%)', 'Overall Score'],
            'Value': [
                report.energy_metrics.r2,
                report.energy_metrics.mape,
                report.thermal_metrics.r2,
                report.thermal_metrics.mape,
                report.performance_metrics.r2,
                report.performance_metrics.mape,
                report.overall_score
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'validation_summary.csv', index=False)
        
        logger.info(f"Validation report exported to {output_path}")
    
    def _align_time_series(self, real_data: pd.DataFrame, simulated_data: pd.DataFrame,
                          time_column: str, value_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Align time series data for comparison"""
        # Ensure time columns are datetime
        real_data = real_data.copy()
        simulated_data = simulated_data.copy()
        
        if time_column in real_data.columns:
            real_data[time_column] = pd.to_datetime(real_data[time_column])
        if time_column in simulated_data.columns:
            simulated_data[time_column] = pd.to_datetime(simulated_data[time_column])
        
        # Find overlapping time range
        if time_column in real_data.columns and time_column in simulated_data.columns:
            real_start = real_data[time_column].min()
            real_end = real_data[time_column].max()
            sim_start = simulated_data[time_column].min()
            sim_end = simulated_data[time_column].max()
            
            overlap_start = max(real_start, sim_start)
            overlap_end = min(real_end, sim_end)
            
            if overlap_start >= overlap_end:
                return np.array([]), np.array([])
            
            # Filter to overlapping period
            real_filtered = real_data[
                (real_data[time_column] >= overlap_start) & 
                (real_data[time_column] <= overlap_end)
            ]
            sim_filtered = simulated_data[
                (simulated_data[time_column] >= overlap_start) & 
                (simulated_data[time_column] <= overlap_end)
            ]
            
            # Resample to common frequency if needed
            if len(real_filtered) != len(sim_filtered):
                # Use hourly resampling as common frequency
                real_resampled = real_filtered.set_index(time_column).resample('H')[value_column].mean()
                sim_resampled = sim_filtered.set_index(time_column).resample('H')[value_column].mean()
                
                # Align indices
                common_index = real_resampled.index.intersection(sim_resampled.index)
                real_values = real_resampled.loc[common_index].values
                sim_values = sim_resampled.loc[common_index].values
            else:
                real_values = real_filtered[value_column].values
                sim_values = sim_filtered[value_column].values
        else:
            # No time alignment possible, use direct comparison
            min_len = min(len(real_data), len(simulated_data))
            real_values = real_data[value_column].iloc[:min_len].values
            sim_values = simulated_data[value_column].iloc[:min_len].values
        
        # Remove NaN values
        mask = ~(np.isnan(real_values) | np.isnan(sim_values))
        return real_values[mask], sim_values[mask]
    
    def _align_by_job_id(self, real_jobs: pd.DataFrame, simulated_jobs: pd.DataFrame,
                        value_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Align job data by job_id"""
        # Merge on job_id
        merged = pd.merge(real_jobs[['job_id', value_column]], 
                         simulated_jobs[['job_id', value_column]], 
                         on='job_id', suffixes=('_real', '_sim'))
        
        if len(merged) == 0:
            return np.array([]), np.array([])
        
        real_values = merged[f'{value_column}_real'].values
        sim_values = merged[f'{value_column}_sim'].values
        
        # Remove NaN values
        mask = ~(np.isnan(real_values) | np.isnan(sim_values))
        return real_values[mask], sim_values[mask]
    
    def _calculate_validation_metrics(self, real_values: np.ndarray, 
                                    simulated_values: np.ndarray) -> ValidationMetrics:
        """Calculate validation metrics"""
        if len(real_values) == 0 or len(simulated_values) == 0:
            return self._empty_validation_metrics()
        
        # Basic metrics
        mae = mean_absolute_error(real_values, simulated_values)
        mse = mean_squared_error(real_values, simulated_values)
        rmse = np.sqrt(mse)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((real_values - simulated_values) / np.maximum(real_values, 1e-8))) * 100
        
        # R-squared
        r2 = r2_score(real_values, simulated_values)
        
        # Correlation
        correlation, _ = stats.pearsonr(real_values, simulated_values)
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(real_values, simulated_values)
        
        # Custom distribution similarity score
        distribution_similarity = 1 - ks_statistic  # Higher is better
        
        return ValidationMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2=r2,
            correlation=correlation,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            distribution_similarity=distribution_similarity
        )
    
    def _empty_validation_metrics(self) -> ValidationMetrics:
        """Return empty validation metrics"""
        return ValidationMetrics(
            mae=np.nan, mse=np.nan, rmse=np.nan, mape=np.nan,
            r2=np.nan, correlation=np.nan, ks_statistic=np.nan,
            ks_pvalue=np.nan, distribution_similarity=np.nan
        )
    
    def _calculate_overall_score(self, energy_metrics: ValidationMetrics,
                               thermal_metrics: ValidationMetrics,
                               performance_metrics: ValidationMetrics) -> float:
        """Calculate overall validation score"""
        scores = []
        weights = []
        
        # Energy score (weight: 0.4)
        if not np.isnan(energy_metrics.r2):
            energy_score = (energy_metrics.r2 + energy_metrics.distribution_similarity) / 2
            scores.append(max(0, energy_score))
            weights.append(0.4)
        
        # Thermal score (weight: 0.3)
        if not np.isnan(thermal_metrics.r2):
            thermal_score = (thermal_metrics.r2 + thermal_metrics.distribution_similarity) / 2
            scores.append(max(0, thermal_score))
            weights.append(0.3)
        
        # Performance score (weight: 0.3)
        if not np.isnan(performance_metrics.r2):
            performance_score = (performance_metrics.r2 + performance_metrics.distribution_similarity) / 2
            scores.append(max(0, performance_score))
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return min(1.0, max(0.0, weighted_score))
    
    def _detailed_analysis(self, real_data: Dict[str, pd.DataFrame],
                         simulated_data: Dict[str, pd.DataFrame],
                         energy_metrics: ValidationMetrics,
                         thermal_metrics: ValidationMetrics,
                         performance_metrics: ValidationMetrics) -> Dict[str, Any]:
        """Perform detailed analysis"""
        analysis = {
            'data_coverage': {},
            'bias_analysis': {},
            'error_patterns': {},
            'outlier_analysis': {}
        }
        
        # Data coverage analysis
        for key in real_data.keys():
            if key in simulated_data:
                real_len = len(real_data[key])
                sim_len = len(simulated_data[key])
                coverage = min(real_len, sim_len) / max(real_len, sim_len)
                analysis['data_coverage'][key] = {
                    'real_records': real_len,
                    'simulated_records': sim_len,
                    'coverage_ratio': coverage
                }
        
        return analysis
    
    def _generate_recommendations(self, energy_metrics: ValidationMetrics,
                                thermal_metrics: ValidationMetrics,
                                performance_metrics: ValidationMetrics,
                                detailed_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Energy recommendations
        if not np.isnan(energy_metrics.r2):
            if energy_metrics.r2 < 0.7:
                recommendations.append(
                    "Energy model accuracy is below 70%. Consider improving power models or adding more features."
                )
            if energy_metrics.mape > 20:
                recommendations.append(
                    "Energy prediction error is high (>20%). Review workload characterization and power profiles."
                )
        
        # Thermal recommendations
        if not np.isnan(thermal_metrics.r2):
            if thermal_metrics.r2 < 0.6:
                recommendations.append(
                    "Thermal model needs improvement. Consider more sophisticated thermal dynamics or cooling models."
                )
        
        # Performance recommendations
        if not np.isnan(performance_metrics.r2):
            if performance_metrics.r2 < 0.8:
                recommendations.append(
                    "Job performance prediction could be improved. Review scheduling algorithms and resource contention models."
                )
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Model validation shows good agreement with real data. Consider fine-tuning for specific use cases.")
        
        return recommendations
    
    def _plot_energy_comparison(self, real_data: pd.DataFrame, simulated_data: pd.DataFrame,
                              output_path: Path) -> str:
        """Generate energy comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Energy Consumption Validation', fontsize=16)
        
        # Time series comparison
        if 'time' in real_data.columns and 'time' in simulated_data.columns:
            axes[0, 0].plot(real_data['time'], real_data.get('energy_wh', real_data.get('total_power_watts', [])), 
                           label='Real', alpha=0.7)
            axes[0, 0].plot(simulated_data['time'], simulated_data.get('energy_wh', simulated_data.get('total_power_watts', [])), 
                           label='Simulated', alpha=0.7)
            axes[0, 0].set_title('Time Series Comparison')
            axes[0, 0].legend()
        
        # Scatter plot
        real_values = real_data.get('energy_wh', real_data.get('total_power_watts', [])).values
        sim_values = simulated_data.get('energy_wh', simulated_data.get('total_power_watts', [])).values
        
        if len(real_values) > 0 and len(sim_values) > 0:
            min_len = min(len(real_values), len(sim_values))
            axes[0, 1].scatter(real_values[:min_len], sim_values[:min_len], alpha=0.5)
            axes[0, 1].plot([min(real_values), max(real_values)], 
                           [min(real_values), max(real_values)], 'r--', label='Perfect Agreement')
            axes[0, 1].set_xlabel('Real Energy')
            axes[0, 1].set_ylabel('Simulated Energy')
            axes[0, 1].set_title('Scatter Plot')
            axes[0, 1].legend()
        
        # Distribution comparison
        axes[1, 0].hist(real_values, alpha=0.5, label='Real', bins=30)
        axes[1, 0].hist(sim_values, alpha=0.5, label='Simulated', bins=30)
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        
        # Error analysis
        if len(real_values) == len(sim_values):
            errors = sim_values - real_values
            axes[1, 1].hist(errors, bins=30, alpha=0.7)
            axes[1, 1].set_title('Prediction Errors')
            axes[1, 1].set_xlabel('Error (Simulated - Real)')
        
        plt.tight_layout()
        plot_path = output_path / 'energy_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_thermal_comparison(self, real_data: pd.DataFrame, simulated_data: pd.DataFrame,
                               output_path: Path) -> str:
        """Generate thermal comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Thermal Behavior Validation', fontsize=16)
        
        # Similar structure to energy plot but for temperature data
        temp_col = 'temperature' if 'temperature' in real_data.columns else 'cpu_temp'
        
        if temp_col in real_data.columns and temp_col in simulated_data.columns:
            # Time series
            if 'time' in real_data.columns:
                axes[0, 0].plot(real_data['time'], real_data[temp_col], label='Real', alpha=0.7)
            if 'time' in simulated_data.columns:
                axes[0, 0].plot(simulated_data['time'], simulated_data[temp_col], label='Simulated', alpha=0.7)
            axes[0, 0].set_title('Temperature Time Series')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].legend()
            
            # Scatter plot
            real_temps = real_data[temp_col].values
            sim_temps = simulated_data[temp_col].values
            min_len = min(len(real_temps), len(sim_temps))
            
            axes[0, 1].scatter(real_temps[:min_len], sim_temps[:min_len], alpha=0.5)
            axes[0, 1].plot([min(real_temps), max(real_temps)], 
                           [min(real_temps), max(real_temps)], 'r--')
            axes[0, 1].set_xlabel('Real Temperature (°C)')
            axes[0, 1].set_ylabel('Simulated Temperature (°C)')
            axes[0, 1].set_title('Temperature Scatter Plot')
            
            # Distributions
            axes[1, 0].hist(real_temps, alpha=0.5, label='Real', bins=30)
            axes[1, 0].hist(sim_temps, alpha=0.5, label='Simulated', bins=30)
            axes[1, 0].set_title('Temperature Distribution')
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].legend()
            
            # Error analysis
            if len(real_temps) == len(sim_temps):
                temp_errors = sim_temps - real_temps
                axes[1, 1].hist(temp_errors, bins=30, alpha=0.7)
                axes[1, 1].set_title('Temperature Prediction Errors')
                axes[1, 1].set_xlabel('Error (°C)')
        
        plt.tight_layout()
        plot_path = output_path / 'thermal_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_performance_comparison(self, real_data: pd.DataFrame, simulated_data: pd.DataFrame,
                                   output_path: Path) -> str:
        """Generate performance comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Job Performance Validation', fontsize=16)
        
        duration_col = 'duration' if 'duration' in real_data.columns else 'duration_seconds'
        
        if duration_col in real_data.columns and duration_col in simulated_data.columns:
            real_durations = real_data[duration_col].values
            sim_durations = simulated_data[duration_col].values
            
            # Job duration comparison
            min_len = min(len(real_durations), len(sim_durations))
            job_indices = range(min_len)
            
            axes[0, 0].plot(job_indices, real_durations[:min_len], 'o-', label='Real', alpha=0.7, markersize=3)
            axes[0, 0].plot(job_indices, sim_durations[:min_len], 'o-', label='Simulated', alpha=0.7, markersize=3)
            axes[0, 0].set_title('Job Duration Comparison')
            axes[0, 0].set_xlabel('Job Index')
            axes[0, 0].set_ylabel('Duration (seconds)')
            axes[0, 0].legend()
            
            # Scatter plot
            axes[0, 1].scatter(real_durations[:min_len], sim_durations[:min_len], alpha=0.5)
            axes[0, 1].plot([min(real_durations), max(real_durations)], 
                           [min(real_durations), max(real_durations)], 'r--')
            axes[0, 1].set_xlabel('Real Duration (s)')
            axes[0, 1].set_ylabel('Simulated Duration (s)')
            axes[0, 1].set_title('Duration Scatter Plot')
            
            # Distribution comparison
            axes[1, 0].hist(real_durations, alpha=0.5, label='Real', bins=30)
            axes[1, 0].hist(sim_durations, alpha=0.5, label='Simulated', bins=30)
            axes[1, 0].set_title('Duration Distribution')
            axes[1, 0].set_xlabel('Duration (seconds)')
            axes[1, 0].legend()
            
            # Performance ratio
            if len(real_durations) == len(sim_durations):
                ratios = sim_durations / np.maximum(real_durations, 1)
                axes[1, 1].hist(ratios, bins=30, alpha=0.7)
                axes[1, 1].axvline(x=1.0, color='r', linestyle='--', label='Perfect Ratio')
                axes[1, 1].set_title('Performance Ratio (Sim/Real)')
                axes[1, 1].set_xlabel('Ratio')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = output_path / 'performance_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_distribution_comparison(self, real_data: Dict[str, pd.DataFrame],
                                    simulated_data: Dict[str, pd.DataFrame],
                                    output_path: Path) -> str:
        """Generate overall distribution comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Distribution Comparison Summary', fontsize=16)
        
        plot_idx = 0
        for key in real_data.keys():
            if key in simulated_data and plot_idx < 6:
                row = plot_idx // 3
                col = plot_idx % 3
                
                real_df = real_data[key]
                sim_df = simulated_data[key]
                
                # Find a numeric column to plot
                numeric_cols = real_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col_name = numeric_cols[0]
                    if col_name in sim_df.columns:
                        axes[row, col].hist(real_df[col_name].dropna(), alpha=0.5, label='Real', bins=20)
                        axes[row, col].hist(sim_df[col_name].dropna(), alpha=0.5, label='Simulated', bins=20)
                        axes[row, col].set_title(f'{key}: {col_name}')
                        axes[row, col].legend()
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plot_path = output_path / 'distribution_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _metrics_to_dict(self, metrics: ValidationMetrics) -> Dict[str, float]:
        """Convert ValidationMetrics to dictionary"""
        return {
            'mae': metrics.mae,
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'mape': metrics.mape,
            'r2': metrics.r2,
            'correlation': metrics.correlation,
            'ks_statistic': metrics.ks_statistic,
            'ks_pvalue': metrics.ks_pvalue,
            'distribution_similarity': metrics.distribution_similarity
        }