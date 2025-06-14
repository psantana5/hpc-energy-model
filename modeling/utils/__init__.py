# Utils Module - Utilities for HPC Energy Modeling

"""
This module contains utility functions and classes for the HPC energy modeling system.

Components:
- config: Configuration management for modeling parameters
- data_utils: Data processing and manipulation utilities
- visualization: Plotting and visualization helpers
"""

from .config import (
    ModelingConfig,
    DatabaseConfig,
    SimulationConfig,
    ModelConfig,
    OutputConfig
)

__all__ = [
    'ModelingConfig',
    'DatabaseConfig', 
    'SimulationConfig',
    'ModelConfig',
    'OutputConfig'
]

# Version information
__version__ = '1.0.0'

# Default configuration paths
DEFAULT_CONFIG_PATHS = {
    'config_file': 'modeling_config.yaml',
    'models_dir': 'models',
    'output_dir': 'output',
    'logs_dir': 'logs'
}

def get_default_config():
    """
    Get default configuration for modeling
    
    Returns:
        ModelingConfig with default values
    """
    return ModelingConfig()

def validate_config(config: ModelingConfig) -> bool:
    """
    Validate a modeling configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises exception if invalid
    """
    return config.validate()