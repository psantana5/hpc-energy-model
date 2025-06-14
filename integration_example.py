#!/usr/bin/env python3
"""
Integration Example: Advanced Performance Benchmarking and ML for HPC Energy Model

This script demonstrates how to use the enhanced performance benchmarking
and advanced machine learning capabilities together for comprehensive
HPC energy analysis and optimization.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from performance_benchmarking import (
    AdvancedPerformanceBenchmarker,
    PerformanceMetrics
)
from advanced_ml_models import (
    AdvancedEnergyPredictor,
    ModelExperiment,
    create_ensemble_model,
    generate_model_comparison_report
)
from data_analysis import HPCDataAnalyzer
from energy_prediction_api import EnergyPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedHPCEnergyOptimizer:
    """
    Integrated HPC Energy Optimizer combining advanced performance benchmarking
    and sophisticated machine learning for comprehensive energy analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated optimizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.benchmarker = AdvancedPerformanceBenchmarker()
        self.ml_predictor = AdvancedEnergyPredictor()
        self.data_analyzer = HPCDataAnalyzer()
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Integrated HPC Energy Optimizer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'results_dir': 'results',
            'benchmark_duration': 60,  # seconds
            'ml_models': ['random_forest', 'xgboost', 'neural_network'],
            'optimization_trials': 100,
            'cross_validation_folds': 5,
            'feature_selection': True,
            'ensemble_methods': ['voting', 'stacking'],
            'performance_thresholds': {
                'cpu_utilization': 80.0,
                'memory_utilization': 85.0,
                'energy_efficiency': 0.7
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def run_comprehensive_analysis(self, workload_name: str = "comprehensive_analysis") -> Dict[str, Any]:
        """
        Run a comprehensive analysis combining performance benchmarking and ML prediction.
        
        Args:
            workload_name: Name of the workload being analyzed
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive analysis for workload: {workload_name}")
        
        results = {
            'workload_name': workload_name,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'ml_predictions': {},
            'optimization_recommendations': {},
            'energy_efficiency_score': 0.0
        }
        
        # Step 1: Performance Benchmarking
        logger.info("Step 1: Running performance benchmarks")
        performance_results = self._run_performance_benchmarks(workload_name)
        results['performance_metrics'] = performance_results
        
        # Step 2: ML-based Energy Prediction
        logger.info("Step 2: Running ML-based energy predictions")
        ml_results = self._run_ml_predictions(performance_results)
        results['ml_predictions'] = ml_results
        
        # Step 3: Generate Optimization Recommendations
        logger.info("Step 3: Generating optimization recommendations")
        optimization_results = self._generate_optimization_recommendations(
            performance_results, ml_results
        )
        results['optimization_recommendations'] = optimization_results
        
        # Step 4: Calculate Energy Efficiency Score
        logger.info("Step 4: Calculating energy efficiency score")
        efficiency_score = self._calculate_energy_efficiency_score(
            performance_results, ml_results
        )
        results['energy_efficiency_score'] = efficiency_score
        
        # Step 5: Save Results
        self._save_comprehensive_results(results)
        
        logger.info(f"Comprehensive analysis completed. Efficiency score: {efficiency_score:.3f}")
        return results
    
    def _run_performance_benchmarks(self, workload_name: str) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmarks.
        
        Args:
            workload_name: Name of the workload
            
        Returns:
            Performance benchmark results
        """
        benchmark_results = {}
        
        # CPU-intensive benchmark
        logger.info("Running CPU-intensive benchmark")
        with self.benchmarker.benchmark_context(f"{workload_name}_cpu"):
            self._simulate_cpu_intensive_workload()
        
        cpu_results = self.benchmarker.analyze_results()
        benchmark_results['cpu_intensive'] = cpu_results
        
        # Memory-intensive benchmark
        logger.info("Running memory-intensive benchmark")
        self.benchmarker.reset_metrics()
        with self.benchmarker.benchmark_context(f"{workload_name}_memory"):
            self._simulate_memory_intensive_workload()
        
        memory_results = self.benchmarker.analyze_results()
        benchmark_results['memory_intensive'] = memory_results
        
        # I/O-intensive benchmark
        logger.info("Running I/O-intensive benchmark")
        self.benchmarker.reset_metrics()
        with self.benchmarker.benchmark_context(f"{workload_name}_io"):
            self._simulate_io_intensive_workload()
        
        io_results = self.benchmarker.analyze_results()
        benchmark_results['io_intensive'] = io_results
        
        # Generate performance report
        report_path = self.results_dir / f"{workload_name}_performance_report.html"
        self.benchmarker.generate_performance_report(str(report_path))
        
        return benchmark_results
    
    def _run_ml_predictions(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ML-based energy predictions using performance data.
        
        Args:
            performance_results: Performance benchmark results
            
        Returns:
            ML prediction results
        """
        # Prepare features from performance data
        features = self._extract_features_from_performance(performance_results)
        
        # Generate synthetic training data (in real scenario, use historical data)
        X_train, y_train = self._generate_training_data(1000)
        
        # Train multiple models
        ml_results = {}
        
        for model_name in self.config['ml_models']:
            logger.info(f"Training {model_name} model")
            
            # Create and train model
            experiment = self.ml_predictor.create_experiment(
                name=f"{model_name}_energy_prediction",
                model_type=model_name,
                hyperparameter_optimization=True,
                n_trials=self.config['optimization_trials']
            )
            
            # Train model
            self.ml_predictor.train_model(
                experiment_name=experiment.name,
                X_train=X_train,
                y_train=y_train,
                cv_folds=self.config['cross_validation_folds']
            )
            
            # Make predictions
            predictions = self.ml_predictor.predict(
                experiment_name=experiment.name,
                X=features.reshape(1, -1)
            )
            
            ml_results[model_name] = {
                'prediction': predictions[0],
                'model_metrics': experiment.metrics,
                'feature_importance': self.ml_predictor.get_feature_importance(
                    experiment.name
                )
            }
        
        # Create ensemble model
        if len(self.config['ml_models']) > 1:
            logger.info("Creating ensemble model")
            ensemble_model = create_ensemble_model(
                [exp.model for exp in self.ml_predictor.experiments.values()],
                method='voting'
            )
            
            ensemble_predictions = ensemble_model.predict(features.reshape(1, -1))
            ml_results['ensemble'] = {
                'prediction': ensemble_predictions[0],
                'model_type': 'ensemble_voting'
            }
        
        return ml_results
    
    def _generate_optimization_recommendations(
        self, 
        performance_results: Dict[str, Any], 
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate optimization recommendations based on performance and ML results.
        
        Args:
            performance_results: Performance benchmark results
            ml_results: ML prediction results
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'performance_optimizations': [],
            'energy_optimizations': [],
            'scheduling_recommendations': [],
            'hardware_recommendations': []
        }
        
        # Analyze performance bottlenecks
        for workload_type, results in performance_results.items():
            if 'bottlenecks' in results:
                for bottleneck in results['bottlenecks']:
                    if bottleneck['severity'] == 'high':
                        recommendations['performance_optimizations'].append({
                            'type': bottleneck['type'],
                            'description': bottleneck['description'],
                            'workload': workload_type,
                            'priority': 'high'
                        })
        
        # Analyze energy predictions
        energy_predictions = [result['prediction'] for result in ml_results.values() if 'prediction' in result]
        if energy_predictions:
            avg_energy = sum(energy_predictions) / len(energy_predictions)
            
            if avg_energy > 1000:  # High energy consumption threshold
                recommendations['energy_optimizations'].append({
                    'type': 'high_energy_consumption',
                    'description': f'Predicted energy consumption is high ({avg_energy:.2f} kWh)',
                    'suggestions': [
                        'Consider energy-aware scheduling',
                        'Optimize workload distribution',
                        'Use dynamic voltage and frequency scaling (DVFS)'
                    ]
                })
        
        # CPU utilization recommendations
        cpu_metrics = performance_results.get('cpu_intensive', {}).get('aggregate_stats', {})
        if cpu_metrics.get('cpu_utilization_mean', 0) > self.config['performance_thresholds']['cpu_utilization']:
            recommendations['scheduling_recommendations'].append({
                'type': 'cpu_overutilization',
                'description': 'High CPU utilization detected',
                'suggestion': 'Consider load balancing or adding more compute nodes'
            })
        
        # Memory utilization recommendations
        memory_metrics = performance_results.get('memory_intensive', {}).get('aggregate_stats', {})
        if memory_metrics.get('memory_utilization_mean', 0) > self.config['performance_thresholds']['memory_utilization']:
            recommendations['hardware_recommendations'].append({
                'type': 'memory_upgrade',
                'description': 'High memory utilization detected',
                'suggestion': 'Consider increasing available memory or optimizing memory usage'
            })
        
        return recommendations
    
    def _calculate_energy_efficiency_score(self, performance_results: Dict[str, Any], ml_results: Dict[str, Any]) -> float:
        """
        Calculate an overall energy efficiency score.
        
        Args:
            performance_results: Performance benchmark results
            ml_results: ML prediction results
            
        Returns:
            Energy efficiency score (0-1, higher is better)
        """
        score_components = []
        
        # Performance efficiency component
        for workload_type, results in performance_results.items():
            if 'aggregate_stats' in results:
                stats = results['aggregate_stats']
                cpu_efficiency = min(stats.get('cpu_utilization_mean', 0) / 100.0, 1.0)
                memory_efficiency = min(stats.get('memory_utilization_mean', 0) / 100.0, 1.0)
                
                # Balanced utilization is better (not too low, not too high)
                cpu_score = 1.0 - abs(cpu_efficiency - 0.7)  # Target 70% utilization
                memory_score = 1.0 - abs(memory_efficiency - 0.7)
                
                score_components.extend([cpu_score, memory_score])
        
        # Energy prediction component
        energy_predictions = [result['prediction'] for result in ml_results.values() if 'prediction' in result]
        if energy_predictions:
            avg_energy = sum(energy_predictions) / len(energy_predictions)
            # Lower energy consumption is better (normalize to 0-1 scale)
            energy_score = max(0, 1.0 - (avg_energy / 2000.0))  # Assume 2000 kWh as high consumption
            score_components.append(energy_score)
        
        # Calculate overall score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.0
    
    def _extract_features_from_performance(self, performance_results: Dict[str, Any]) -> 'numpy.ndarray':
        """
        Extract ML features from performance benchmark results.
        
        Args:
            performance_results: Performance benchmark results
            
        Returns:
            Feature array for ML prediction
        """
        import numpy as np
        
        features = []
        
        for workload_type, results in performance_results.items():
            if 'aggregate_stats' in results:
                stats = results['aggregate_stats']
                features.extend([
                    stats.get('cpu_utilization_mean', 0),
                    stats.get('memory_utilization_mean', 0),
                    stats.get('disk_io_read_mean', 0),
                    stats.get('disk_io_write_mean', 0),
                    stats.get('network_io_sent_mean', 0),
                    stats.get('network_io_recv_mean', 0),
                    stats.get('estimated_power_mean', 0)
                ])
        
        # Pad or truncate to fixed size (21 features for 3 workload types)
        target_size = 21
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features)
    
    def _generate_training_data(self, n_samples: int) -> tuple:
        """
        Generate synthetic training data for ML models.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Tuple of (X_train, y_train)
        """
        import numpy as np
        
        # Generate synthetic features
        np.random.seed(42)
        X = np.random.rand(n_samples, 21) * 100  # 21 features, scaled 0-100
        
        # Generate synthetic energy consumption based on features
        # Simple model: energy = base + cpu_factor * cpu + memory_factor * memory + ...
        y = (
            50 +  # Base consumption
            X[:, 0] * 5 +  # CPU utilization factor
            X[:, 1] * 3 +  # Memory utilization factor
            X[:, 2] * 0.1 +  # Disk I/O factor
            X[:, 6] * 2 +  # Power factor
            np.random.normal(0, 10, n_samples)  # Noise
        )
        
        return X, y
    
    def _simulate_cpu_intensive_workload(self):
        """Simulate CPU-intensive workload."""
        import math
        end_time = time.time() + 10  # Run for 10 seconds
        while time.time() < end_time:
            # CPU-intensive calculation
            for i in range(10000):
                math.sqrt(i * math.pi)
    
    def _simulate_memory_intensive_workload(self):
        """Simulate memory-intensive workload."""
        # Allocate and use memory
        data = []
        for i in range(1000):
            data.append([j for j in range(1000)])
        
        # Process the data
        for chunk in data:
            sum(chunk)
        
        time.sleep(5)
    
    def _simulate_io_intensive_workload(self):
        """Simulate I/O-intensive workload."""
        import tempfile
        
        # Create temporary files and perform I/O operations
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(100):
                file_path = Path(temp_dir) / f"test_file_{i}.txt"
                with open(file_path, 'w') as f:
                    f.write("test data " * 1000)
                
                # Read the file back
                with open(file_path, 'r') as f:
                    f.read()
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """
        Save comprehensive analysis results to file.
        
        Args:
            results: Analysis results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive optimization report.
        
        Args:
            results: Analysis results
            
        Returns:
            HTML report content
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HPC Energy Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9f4ff; border-radius: 3px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HPC Energy Optimization Report</h1>
                <p><strong>Workload:</strong> {results['workload_name']}</p>
                <p><strong>Timestamp:</strong> {results['timestamp']}</p>
                <p><strong>Energy Efficiency Score:</strong> <span class="score">{results['energy_efficiency_score']:.3f}</span></p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics Summary</h2>
        """
        
        # Add performance metrics
        for workload_type, metrics in results['performance_metrics'].items():
            if 'aggregate_stats' in metrics:
                stats = metrics['aggregate_stats']
                html_content += f"""
                <h3>{workload_type.replace('_', ' ').title()}</h3>
                <div class="metric">CPU: {stats.get('cpu_utilization_mean', 0):.1f}%</div>
                <div class="metric">Memory: {stats.get('memory_utilization_mean', 0):.1f}%</div>
                <div class="metric">Power: {stats.get('estimated_power_mean', 0):.1f}W</div>
                """
        
        html_content += "</div><div class='section'><h2>ML Predictions</h2>"
        
        # Add ML predictions
        for model_name, prediction in results['ml_predictions'].items():
            if 'prediction' in prediction:
                html_content += f"<div class='metric'>{model_name}: {prediction['prediction']:.2f} kWh</div>"
        
        html_content += "</div><div class='section'><h2>Optimization Recommendations</h2>"
        
        # Add recommendations
        for category, recommendations in results['optimization_recommendations'].items():
            if recommendations:
                html_content += f"<h3>{category.replace('_', ' ').title()}</h3>"
                for rec in recommendations:
                    html_content += f"<div class='recommendation'><strong>{rec.get('type', 'N/A')}:</strong> {rec.get('description', 'N/A')}</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content

def main():
    """
    Main function demonstrating the integrated HPC energy optimization.
    """
    print("HPC Energy Model - Integrated Performance and ML Analysis")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = IntegratedHPCEnergyOptimizer()
    
    # Run comprehensive analysis
    results = optimizer.run_comprehensive_analysis("demo_workload")
    
    # Generate and save optimization report
    report_html = optimizer.generate_optimization_report(results)
    report_path = optimizer.results_dir / "optimization_report.html"
    
    with open(report_path, 'w') as f:
        f.write(report_html)
    
    print(f"\nAnalysis completed!")
    print(f"Energy Efficiency Score: {results['energy_efficiency_score']:.3f}")
    print(f"Detailed report saved to: {report_path}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 30)
    
    # Performance summary
    for workload_type, metrics in results['performance_metrics'].items():
        if 'aggregate_stats' in metrics:
            stats = metrics['aggregate_stats']
            print(f"{workload_type}: CPU {stats.get('cpu_utilization_mean', 0):.1f}%, "
                  f"Memory {stats.get('memory_utilization_mean', 0):.1f}%, "
                  f"Power {stats.get('estimated_power_mean', 0):.1f}W")
    
    # ML predictions summary
    print("\nEnergy Predictions:")
    for model_name, prediction in results['ml_predictions'].items():
        if 'prediction' in prediction:
            print(f"{model_name}: {prediction['prediction']:.2f} kWh")
    
    # Recommendations summary
    total_recommendations = sum(len(recs) for recs in results['optimization_recommendations'].values())
    print(f"\nTotal optimization recommendations: {total_recommendations}")

if __name__ == "__main__":
    main()