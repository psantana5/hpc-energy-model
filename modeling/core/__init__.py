# Core Module - Core Components for HPC Energy Modeling

"""
This module contains the core components for HPC energy modeling and simulation.

Components:
- data_loader: Historical data loading and preprocessing
- simulation_engine: HPC cluster simulation engine
- metrics: Performance and validation metrics
"""

from .data_loader import HistoricalDataLoader
from .simulation_engine import (
    HPCClusterSimulator,
    SimulatedNode,
    JobSpec,
    NodeSpec,
    SimulationMetrics,
    ThermalModel,
    PowerModel
)

__all__ = [
    'HistoricalDataLoader',
    'HPCClusterSimulator',
    'SimulatedNode',
    'JobSpec',
    'NodeSpec', 
    'SimulationMetrics',
    'ThermalModel',
    'PowerModel'
]

# Core module metadata
CORE_INFO = {
    'data_loader': {
        'class': HistoricalDataLoader,
        'description': 'Loads and preprocesses historical HPC data from TimescaleDB',
        'capabilities': ['job_metrics', 'node_metrics', 'energy_predictions', 'data_export']
    },
    'simulation_engine': {
        'class': HPCClusterSimulator,
        'description': 'Simulates HPC cluster behavior with thermal and energy modeling',
        'capabilities': ['job_scheduling', 'resource_management', 'thermal_modeling', 'energy_prediction']
    }
}

def get_core_info(component_name: str = None):
    """
    Get information about core components
    
    Args:
        component_name: Specific component name, or None for all components
        
    Returns:
        Component information dictionary
    """
    if component_name:
        return CORE_INFO.get(component_name, {})
    return CORE_INFO

def create_data_loader(config):
    """
    Factory function to create data loader instance
    
    Args:
        config: ModelingConfig instance
        
    Returns:
        HistoricalDataLoader instance
    """
    return HistoricalDataLoader(config)

def create_simulator(config, energy_predictor=None, thermal_predictor=None):
    """
    Factory function to create simulator instance
    
    Args:
        config: ModelingConfig instance
        energy_predictor: Optional EnergyPredictor instance
        thermal_predictor: Optional ThermalPredictor instance
        
    Returns:
        HPCClusterSimulator instance
    """
    return HPCClusterSimulator(config, energy_predictor, thermal_predictor)