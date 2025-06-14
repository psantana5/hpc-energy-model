# Validation Module - Model Validation and Performance Assessment

"""
This module contains tools for validating simulation results and model performance
against real HPC cluster data.

Components:
- validator: Model validation and comparison tools
- metrics: Statistical and performance metrics
- reports: Validation report generation
"""

from .validator import ModelValidator

__all__ = [
    'ModelValidator'
]

# Validation metrics available
VALIDATION_METRICS = {
    'statistical': [
        'mae',  # Mean Absolute Error
        'mse',  # Mean Squared Error
        'rmse', # Root Mean Squared Error
        'mape', # Mean Absolute Percentage Error
        'r2',   # R-squared
        'correlation', # Pearson correlation
        'ks_statistic' # Kolmogorov-Smirnov test
    ],
    'energy_specific': [
        'energy_efficiency_error',
        'power_prediction_accuracy',
        'consumption_variance'
    ],
    'thermal_specific': [
        'temperature_accuracy',
        'thermal_profile_similarity',
        'peak_temperature_error'
    ]
}

# Report formats supported
REPORT_FORMATS = {
    'json': 'JSON format for programmatic access',
    'markdown': 'Markdown format for documentation',
    'html': 'HTML format for web viewing',
    'pdf': 'PDF format for formal reports'
}

def get_available_metrics(category: str = None):
    """
    Get available validation metrics
    
    Args:
        category: Metric category ('statistical', 'energy_specific', 'thermal_specific')
                 or None for all metrics
        
    Returns:
        List of available metrics
    """
    if category:
        return VALIDATION_METRICS.get(category, [])
    
    all_metrics = []
    for metrics_list in VALIDATION_METRICS.values():
        all_metrics.extend(metrics_list)
    return all_metrics

def get_supported_formats():
    """
    Get supported report formats
    
    Returns:
        Dictionary of format names and descriptions
    """
    return REPORT_FORMATS.copy()

def create_validator(config):
    """
    Factory function to create validator instance
    
    Args:
        config: ModelingConfig instance
        
    Returns:
        ModelValidator instance
    """
    return ModelValidator(config)

# Validation thresholds for different metrics
VALIDATION_THRESHOLDS = {
    'excellent': {
        'r2': 0.9,
        'mape': 5.0,
        'correlation': 0.95
    },
    'good': {
        'r2': 0.8,
        'mape': 10.0,
        'correlation': 0.85
    },
    'acceptable': {
        'r2': 0.7,
        'mape': 15.0,
        'correlation': 0.75
    },
    'poor': {
        'r2': 0.5,
        'mape': 25.0,
        'correlation': 0.6
    }
}

def assess_model_quality(metrics: dict) -> str:
    """
    Assess model quality based on validation metrics
    
    Args:
        metrics: Dictionary of validation metrics
        
    Returns:
        Quality assessment string ('excellent', 'good', 'acceptable', 'poor')
    """
    r2 = metrics.get('r2', 0)
    mape = metrics.get('mape', 100)
    correlation = metrics.get('correlation', 0)
    
    if (r2 >= VALIDATION_THRESHOLDS['excellent']['r2'] and 
        mape <= VALIDATION_THRESHOLDS['excellent']['mape'] and
        correlation >= VALIDATION_THRESHOLDS['excellent']['correlation']):
        return 'excellent'
    elif (r2 >= VALIDATION_THRESHOLDS['good']['r2'] and 
          mape <= VALIDATION_THRESHOLDS['good']['mape'] and
          correlation >= VALIDATION_THRESHOLDS['good']['correlation']):
        return 'good'
    elif (r2 >= VALIDATION_THRESHOLDS['acceptable']['r2'] and 
          mape <= VALIDATION_THRESHOLDS['acceptable']['mape'] and
          correlation >= VALIDATION_THRESHOLDS['acceptable']['correlation']):
        return 'acceptable'
    else:
        return 'poor'