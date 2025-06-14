#!/usr/bin/env python3
# Main HPC Energy Modeling Script

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import json

# Add modeling module to path
sys.path.append(str(Path(__file__).parent))

from utils.config import ModelingConfig
from core.data_loader import HistoricalDataLoader
from core.simulation_engine import (
    HPCClusterSimulator, NodeSpec, JobSpec, 
    SimulationMetrics, ThermalModel, PowerModel
)
from validation.validator import ModelValidator, ValidationReport
from models.energy_predictor import EnergyPredictor
from models.thermal_predictor import ThermalPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modeling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HPCModelingPipeline:
    """
    Main pipeline for HPC energy modeling and simulation
    
    This class orchestrates the entire modeling workflow:
    1. Load historical data from TimescaleDB
    2. Train predictive models
    3. Generate synthetic workloads
    4. Run cluster simulations
    5. Validate results against real data
    6. Generate reports and visualizations
    """
    
    def __init__(self, config_path: str):
        self.config = ModelingConfig.from_yaml(config_path)
        self.data_loader = None
        self.simulator = None
        self.validator = None
        self.models = {}
        
        # Create output directories
        self.output_dir = Path(self.config.output.base_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'simulations').mkdir(exist_ok=True)
        (self.output_dir / 'validation').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        logger.info(f"Initialized HPC Modeling Pipeline with config: {config_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_historical_data(self) -> Dict[str, any]:
        """
        Load historical data from the database
        
        Returns:
            Dictionary with loaded datasets and metadata
        """
        logger.info("Loading historical data from TimescaleDB")
        
        self.data_loader = HistoricalDataLoader(self.config)
        
        with self.data_loader as loader:
            # Load job metrics
            job_metrics = loader.load_job_metrics(
                start_date=datetime.now() - timedelta(days=self.config.historical_data_days)
            )
            
            # Load node metrics
            node_metrics = loader.load_node_metrics(
                start_date=datetime.now() - timedelta(days=self.config.historical_data_days)
            )
            
            # Load energy predictions if available
            try:
                energy_predictions = loader.load_energy_predictions(
                    start_date=datetime.now() - timedelta(days=self.config.historical_data_days)
                )
            except Exception as e:
                logger.warning(f"Could not load energy predictions: {e}")
                energy_predictions = None
            
            # Get dataset information
            dataset_info = loader.get_dataset_info()
            
            # Preprocess data
            processed_data = loader.preprocess_data(
                fill_missing=True,
                remove_outliers=True,
                normalize_features=False
            )
            
            # Export raw data
            loader.export_to_files(
                str(self.output_dir / 'data' / 'raw'),
                format='parquet'
            )
        
        logger.info(f"Loaded data: {dataset_info.total_jobs} jobs, {dataset_info.total_nodes} nodes")
        logger.info(f"Data quality score: {dataset_info.data_quality_score:.1f}%")
        
        return {
            'job_metrics': job_metrics,
            'node_metrics': node_metrics,
            'energy_predictions': energy_predictions,
            'processed_data': processed_data,
            'dataset_info': dataset_info
        }
    
    def train_predictive_models(self, data: Dict[str, any]) -> Dict[str, any]:
        """
        Train predictive models for energy and thermal behavior
        
        Args:
            data: Historical data dictionary
            
        Returns:
            Dictionary with trained models
        """
        logger.info("Training predictive models")
        
        models = {}
        
        # Train energy predictor
        if data['job_metrics'] is not None:
            logger.info("Training energy prediction model")
            energy_predictor = EnergyPredictor(self.config)
            training_result = energy_predictor.train(data['job_metrics'])
            
            # Only save and evaluate if training was successful
            if training_result.get('status') not in ['no_data', 'no_valid_data']:
                energy_predictor.save_model(str(self.output_dir / 'models' / 'energy_predictor.pkl'))
                models['energy'] = energy_predictor
                
                # Evaluate model
                energy_metrics = energy_predictor.evaluate(data['job_metrics'])
                logger.info(f"Energy model R²: {energy_metrics.get('r2', 'N/A'):.3f}")
            else:
                logger.warning(f"Energy model training failed: {training_result.get('message', 'Unknown error')}")
        
        # Train thermal predictor
        if data['node_metrics'] is not None:
            logger.info("Training thermal prediction model")
            thermal_predictor = ThermalPredictor(self.config)
            training_result = thermal_predictor.train(data['node_metrics'])
            
            # Only save and evaluate if training was successful
            if training_result.get('status') not in ['no_data', 'no_valid_data']:
                thermal_predictor.save_model(str(self.output_dir / 'models' / 'thermal_predictor.pkl'))
                models['thermal'] = thermal_predictor
                
                # Evaluate model
                thermal_metrics = thermal_predictor.evaluate(data['node_metrics'])
                logger.info(f"Thermal model R²: {thermal_metrics.get('r2', 'N/A'):.3f}")
            else:
                logger.warning(f"Thermal model training failed: {training_result.get('message', 'Unknown error')}")
        
        self.models = models
        return models
    
    def generate_workload(self, data: Dict[str, any], 
                         num_jobs: int = None) -> List[JobSpec]:
        """
        Generate synthetic workload based on historical patterns
        
        Args:
            data: Historical data dictionary
            num_jobs: Number of jobs to generate (default from config)
            
        Returns:
            List of JobSpec objects
        """
        if num_jobs is None:
            num_jobs = self.config.simulation.num_jobs
        
        logger.info(f"Generating synthetic workload with {num_jobs} jobs")
        
        # Simple workload generation (placeholder implementation)
        jobs = []
        for i in range(num_jobs):
            job = JobSpec(
                job_id=f"synthetic_job_{i:04d}",
                job_type="cpu_intensive",
                cpu_cores=4,
                memory_mb=8192,
                duration_seconds=3600,
                arrival_time=i * 60,  # Jobs arrive every minute
                workload_pattern="compute"
            )
            jobs.append(job)
        
        # Export workload specification
        workload_data = [{
            'job_id': job.job_id,
            'job_type': job.job_type,
            'cpu_cores': job.cpu_cores,
            'memory_mb': job.memory_mb,
            'duration_seconds': job.duration_seconds,
            'arrival_time': job.arrival_time,
            'workload_pattern': job.workload_pattern
        } for job in jobs]
        
        with open(self.output_dir / 'simulations' / 'workload_spec.json', 'w') as f:
            json.dump(workload_data, f, indent=2)
        
        logger.info(f"Generated {len(jobs)} jobs for simulation")
        return jobs
    
    def setup_cluster_simulation(self, data: Dict[str, any]) -> HPCClusterSimulator:
        """
        Set up the cluster simulation environment
        
        Args:
            data: Historical data dictionary
            
        Returns:
            Configured HPCClusterSimulator
        """
        logger.info("Setting up cluster simulation")
        
        self.simulator = HPCClusterSimulator(self.config)
        
        # Create nodes based on historical data or configuration
        if data['node_metrics'] is not None:
            # Extract unique nodes from historical data
            unique_nodes = data['node_metrics']['node_id'].unique()
            
            for node_id in unique_nodes[:self.config.simulation.num_nodes]:
                node_data = data['node_metrics'][data['node_metrics']['node_id'] == node_id]
                
                # Estimate node specifications from historical data
                max_cpu = node_data['cpu_usage'].quantile(0.95) if 'cpu_usage' in node_data else 80
                max_memory = node_data['memory_usage'].quantile(0.95) if 'memory_usage' in node_data else 32000
                max_power = node_data['power_consumption_watts'].max() if 'power_consumption_watts' in node_data else 300
                
                node_spec = NodeSpec(
                    node_id=str(node_id),
                    cpu_cores=max(4, int(max_cpu / 10)),  # Estimate cores
                    memory_mb=max(8000, int(max_memory)),
                    max_power_watts=max(200, max_power),
                    idle_power_watts=max(50, max_power * 0.2),
                    thermal_capacity=1000.0,
                    cooling_efficiency=50.0
                )
                
                self.simulator.add_node(node_spec)
        else:
            # Create default cluster configuration
            for i in range(self.config.simulation.num_nodes):
                node_spec = NodeSpec(
                    node_id=f"node-{i:03d}",
                    cpu_cores=16,
                    memory_mb=32000,
                    max_power_watts=300,
                    idle_power_watts=60,
                    thermal_capacity=1000.0,
                    cooling_efficiency=50.0
                )
                
                self.simulator.add_node(node_spec)
        
        logger.info(f"Cluster simulation setup with {len(self.simulator.nodes)} nodes")
        return self.simulator
    
    def run_simulation(self, jobs: List[JobSpec]) -> Dict[str, any]:
        """
        Run the cluster simulation
        
        Args:
            jobs: List of jobs to simulate
            
        Returns:
            Simulation results dictionary
        """
        logger.info(f"Running simulation with {len(jobs)} jobs")
        
        # Submit jobs to simulator
        for job in jobs:
            self.simulator.submit_job(job)
        
        # Run simulation
        simulation_duration = self.config.simulation.simulation_duration_hours * 3600
        self.simulator.run_simulation(simulation_duration)
        
        # Export results
        self.simulator.export_results(str(self.output_dir / 'simulations'))
        
        # Get summary
        summary = self.simulator.get_summary()
        
        logger.info(f"Simulation completed. {summary['job_statistics']['total_completed']} jobs completed")
        logger.info(f"Total energy consumed: {summary['energy_summary']['total_energy_wh']:.1f} Wh")
        
        return {
            'metrics': self.simulator.metrics,
            'summary': summary,
            'completed_jobs': self.simulator.completed_jobs
        }
    
    def validate_results(self, real_data: Dict[str, any], 
                        simulation_results: Dict[str, any]) -> ValidationReport:
        """
        Validate simulation results against real data
        
        Args:
            real_data: Real historical data
            simulation_results: Simulation results
            
        Returns:
            ValidationReport with comparison results
        """
        logger.info("Validating simulation results")
        
        self.validator = ModelValidator(self.config)
        
        # Prepare data for validation
        real_validation_data = {
            'job_metrics': real_data['job_metrics'],
            'node_metrics': real_data['node_metrics']
        }
        
        # Convert simulation results to DataFrames
        import pandas as pd
        
        simulated_validation_data = {
            'job_metrics': pd.DataFrame(simulation_results['metrics'].job_metrics),
            'node_metrics': pd.DataFrame(simulation_results['metrics'].node_metrics)
        }
        
        # Add energy data if available
        if simulation_results['metrics'].energy_consumption:
            energy_predictions = real_data.get('energy_predictions')
            if energy_predictions is not None:
                # Map actual_energy_wh to energy_wh for validation
                energy_predictions = energy_predictions.copy()
                if 'actual_energy_wh' in energy_predictions.columns:
                    energy_predictions['energy_wh'] = energy_predictions['actual_energy_wh']
                real_validation_data['energy'] = energy_predictions
            else:
                real_validation_data['energy'] = None
            
            simulated_validation_data['energy'] = pd.DataFrame(
                simulation_results['metrics'].energy_consumption
            )
        
        # Perform comprehensive validation
        validation_report = self.validator.comprehensive_validation(
            real_validation_data,
            simulated_validation_data,
            model_name="HPC_Energy_Simulation"
        )
        
        # Generate validation plots
        plots = self.validator.generate_validation_plots(
            real_validation_data,
            simulated_validation_data,
            str(self.output_dir / 'validation')
        )
        
        validation_report.plots_generated = plots
        
        # Export validation report
        self.validator.export_validation_report(
            validation_report,
            str(self.output_dir / 'validation')
        )
        
        logger.info(f"Validation completed. Overall score: {validation_report.overall_score:.2f}")
        
        return validation_report
    
    def generate_final_report(self, data: Dict[str, any], 
                            simulation_results: Dict[str, any],
                            validation_report: ValidationReport):
        """
        Generate final comprehensive report
        
        Args:
            data: Historical data
            simulation_results: Simulation results
            validation_report: Validation report
        """
        logger.info("Generating final report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'pipeline_version': '1.0.0'
            },
            'data_summary': {
                'total_jobs': data['dataset_info'].total_jobs,
                'total_nodes': data['dataset_info'].total_nodes,
                'data_quality_score': data['dataset_info'].data_quality_score,
                'time_range': {
                    'start': data['dataset_info'].start_time.isoformat(),
                    'end': data['dataset_info'].end_time.isoformat()
                }
            },
            'simulation_summary': simulation_results['summary'],
            'validation_summary': {
                'overall_score': validation_report.overall_score,
                'energy_r2': validation_report.energy_metrics.r2,
                'thermal_r2': validation_report.thermal_metrics.r2,
                'performance_r2': validation_report.performance_metrics.r2,
                'recommendations': validation_report.recommendations
            },
            'model_performance': {
                'energy_model': self.models.get('energy', {}).get('metrics', {}),
                'thermal_model': self.models.get('thermal', {}).get('metrics', {})
            }
        }
        
        # Export final report
        with open(self.output_dir / 'reports' / 'final_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary markdown report
        self._generate_markdown_report(report)
        
        logger.info(f"Final report generated: {self.output_dir / 'reports' / 'final_report.json'}")
    
    def _generate_markdown_report(self, report: Dict[str, any]):
        """Generate markdown summary report"""
        markdown_content = f"""
# HPC Energy Modeling Report

**Generated:** {report['metadata']['timestamp']}

## Executive Summary

This report presents the results of High-Level Modeling (HLM) simulation for HPC cluster energy and thermal behavior.

### Key Metrics

- **Overall Validation Score:** {report['validation_summary']['overall_score']:.2f}/1.0
- **Energy Model Accuracy (R²):** {report['validation_summary']['energy_r2']:.3f}
- **Thermal Model Accuracy (R²):** {report['validation_summary']['thermal_r2']:.3f}
- **Performance Model Accuracy (R²):** {report['validation_summary']['performance_r2']:.3f}

## Data Summary

- **Historical Jobs Analyzed:** {report['data_summary']['total_jobs']:,}
- **Nodes Monitored:** {report['data_summary']['total_nodes']}
- **Data Quality Score:** {report['data_summary']['data_quality_score']:.1f}%
- **Time Period:** {report['data_summary']['time_range']['start']} to {report['data_summary']['time_range']['end']}

## Simulation Results

- **Jobs Simulated:** {report['simulation_summary']['job_statistics']['total_submitted']:,}
- **Jobs Completed:** {report['simulation_summary']['job_statistics']['total_completed']:,}
- **Completion Rate:** {(report['simulation_summary']['job_statistics']['total_completed']/report['simulation_summary']['job_statistics']['total_submitted']*100) if report['simulation_summary']['job_statistics']['total_submitted'] > 0 else 0:.1f}%
- **Total Energy Consumed:** {report['simulation_summary']['energy_summary']['total_energy_wh']:,.1f} Wh
- **Average Power:** {report['simulation_summary']['energy_summary']['average_power_w']:.1f} W

## Validation Results

The simulation results were validated against real historical data:

### Recommendations

"""
        
        for i, rec in enumerate(report['validation_summary']['recommendations'], 1):
            markdown_content += f"{i}. {rec}\n"
        
        markdown_content += """

## Files Generated

- **Raw Data:** `data/raw/`
- **Trained Models:** `models/`
- **Simulation Results:** `simulations/`
- **Validation Plots:** `validation/`
- **Final Report:** `reports/final_report.json`

## Usage

This HLM system can be used for:

1. **Energy Planning:** Predict energy consumption for different workload scenarios
2. **Thermal Management:** Simulate thermal behavior under various conditions
3. **Capacity Planning:** Evaluate cluster performance with different configurations
4. **What-if Analysis:** Test scenarios without running real workloads

---

*Generated by HPC Energy Modeling Pipeline v1.0.0*
"""
        
        with open(self.output_dir / 'reports' / 'summary_report.md', 'w') as f:
            f.write(markdown_content)
    
    def run_full_pipeline(self):
        """
        Run the complete modeling pipeline
        """
        logger.info("Starting full HPC modeling pipeline")
        
        try:
            # Step 1: Load historical data
            data = self.load_historical_data()
            
            # Step 2: Train predictive models
            models = self.train_predictive_models(data)
            
            # Step 3: Generate synthetic workload
            jobs = self.generate_workload(data)
            
            # Step 4: Set up and run simulation
            simulator = self.setup_cluster_simulation(data)
            simulation_results = self.run_simulation(jobs)
            
            # Step 5: Validate results
            validation_report = self.validate_results(data, simulation_results)
            
            # Step 6: Generate final report
            self.generate_final_report(data, simulation_results, validation_report)
            
            logger.info("HPC modeling pipeline completed successfully")
            logger.info(f"Results available in: {self.output_dir}")
            
            return {
                'success': True,
                'output_directory': str(self.output_dir),
                'validation_score': validation_report.overall_score,
                'summary': simulation_results['summary']
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HPC Energy Modeling Pipeline')
    parser.add_argument('--config', '-c', required=True, 
                       help='Path to configuration YAML file')
    parser.add_argument('--output', '-o', 
                       help='Output directory (overrides config)')
    parser.add_argument('--jobs', '-j', type=int,
                       help='Number of jobs to simulate (overrides config)')
    parser.add_argument('--duration', '-d', type=float,
                       help='Simulation duration in hours (overrides config)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = HPCModelingPipeline(args.config)
        
        # Override configuration if specified
        if args.output:
            pipeline.config.output.base_directory = args.output
            pipeline.output_dir = Path(args.output)
        
        if args.jobs:
            pipeline.config.simulation.num_jobs = args.jobs
        
        if args.duration:
            pipeline.config.simulation.simulation_duration_hours = args.duration
        
        if args.validate_only:
            logger.info("Running validation only")
            # Load existing data and results for validation
            # This would require implementing a method to load existing results
            logger.warning("Validate-only mode not fully implemented")
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline()
            
            print("\n" + "="*60)
            print("HPC ENERGY MODELING PIPELINE COMPLETED")
            print("="*60)
            print(f"Output Directory: {results['output_directory']}")
            print(f"Validation Score: {results['validation_score']:.2f}/1.0")
            print(f"Jobs Completed: {results['summary']['job_statistics']['total_completed']}")
            print(f"Energy Consumed: {results['summary']['energy_summary']['total_energy_wh']:.1f} Wh")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()