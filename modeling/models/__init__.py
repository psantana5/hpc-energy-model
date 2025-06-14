# Models Module - Predictive Models for HPC Energy and Thermal Behavior

"""
This module contains machine learning models for predicting energy consumption
and thermal behavior in HPC clusters.

Models:
- EnergyPredictor: Predicts energy consumption based on job characteristics
- ThermalPredictor: Predicts thermal behavior and temperature patterns
"""

from .energy_predictor import EnergyPredictor
from .thermal_predictor import ThermalPredictor

__all__ = [
    'EnergyPredictor',
    'ThermalPredictor'
]

# Model metadata
MODEL_INFO = {
    'energy_predictor': {
        'class': EnergyPredictor,
        'description': 'Predicts energy consumption for HPC jobs',
        'targets': ['estimated_energy_wh'],
        'algorithms': ['random_forest', 'gradient_boosting', 'linear', 'ridge']
    },
    'thermal_predictor': {
        'class': ThermalPredictor,
        'description': 'Predicts thermal behavior and temperature patterns',
        'targets': ['avg_cpu_temp', 'peak_cpu_temp', 'temp_variance'],
        'algorithms': ['random_forest', 'gradient_boosting', 'linear', 'ridge']
    }
}

def get_model_info(model_name: str = None):
    """
    Get information about available models
    
    Args:
        model_name: Specific model name, or None for all models
        
    Returns:
        Model information dictionary
    """
    if model_name:
        return MODEL_INFO.get(model_name, {})
    return MODEL_INFO

def create_model(model_name: str, config):
    """
    Factory function to create model instances
    
    Args:
        model_name: Name of the model to create
        config: ModelingConfig instance
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_INFO.keys())}")
    
    model_class = MODEL_INFO[model_name]['class']
    return model_class(config)