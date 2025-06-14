# Configuration module for HPC Energy Modeling

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "timescaledb"
    port: int = 5432
    database: str = "hpc_energy"
    username: str = "postgres"
    password: str = "password"
    schema: str = "hpc_energy"
    connection_timeout: int = 30
    query_timeout: int = 300
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv('TIMESCALE_HOST', cls.host),
            port=int(os.getenv('TIMESCALE_PORT', cls.port)),
            database=os.getenv('TIMESCALE_DB', cls.database),
            username=os.getenv('TIMESCALE_USER', cls.username),
            password=os.getenv('TIMESCALE_PASS', cls.password)
        )

@dataclass
class SimulationConfig:
    """Simulation parameters configuration"""
    # Cluster specifications
    num_nodes: int = 10
    cores_per_node: int = 16
    memory_per_node_gb: int = 64
    
    # Time parameters
    time_step_seconds: int = 60
    simulation_duration_hours: int = 24
    warmup_time_minutes: int = 30
    cooldown_time_minutes: int = 15
    
    # Environmental conditions
    ambient_temperature_c: float = 22.0
    cooling_efficiency: float = 0.85
    humidity_percent: float = 45.0
    
    # Resource management
    scheduling_algorithm: str = "fifo"
    max_queue_size: int = 1000
    job_timeout_hours: int = 24
    
    # Thermal modeling
    thermal_time_constant: int = 300
    max_safe_temperature: float = 85.0
    throttling_temperature: float = 90.0
    
    # Power modeling
    idle_power_watts: float = 150
    max_power_watts: float = 400
    power_efficiency: float = 0.92
    
    # Legacy parameters for backward compatibility
    default_cpu_cores: int = 8
    default_memory_gb: int = 32
    default_thermal_capacity: float = 1000.0
    default_cooling_coefficient: float = 0.1
    ambient_temperature: float = 22.0
    thermal_threshold_warning: float = 75.0
    thermal_threshold_critical: float = 85.0
    
    # Workload parameters
    cpu_utilization_noise: float = 0.05
    memory_utilization_noise: float = 0.03
    io_pattern_variability: float = 0.1
    num_jobs: int = 100  # Number of synthetic jobs to generate

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # Model types to use
    energy_model_type: str = "random_forest"  # random_forest, xgboost, neural_network
    thermal_model_type: str = "physics_based"  # physics_based, ml_based, hybrid
    workload_model_type: str = "markov_chain"  # markov_chain, lstm, statistical
    
    # Training parameters
    train_test_split: float = 0.8
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Feature engineering
    feature_window_minutes: int = 10
    feature_lag_steps: int = 3
    enable_feature_selection: bool = True
    
    # Model hyperparameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    })
    
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    })

@dataclass
class OutputConfig:
    """Output and reporting configuration"""
    # Directory structure
    base_directory: str = "./modeling_output"
    models_directory: str = "models"
    reports_directory: str = "reports"
    plots_directory: str = "plots"
    data_directory: str = "data"
    logs_directory: str = "logs"
    
    # File formats
    data_format: str = "parquet"
    plot_format: str = "png"
    report_format: str = "markdown"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_rotation: bool = True
    max_log_size_mb: int = 100
    
    # Visualization
    plot_style: str = "seaborn"
    figure_size: list = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    color_palette: str = "viridis"
    
    # Report generation
    include_plots: bool = True
    include_raw_data: bool = False
    include_model_details: bool = True
    include_validation_metrics: bool = True
    
    # Data export
    export_predictions: bool = True
    export_feature_importance: bool = True
    export_validation_results: bool = True
    compression: str = "gzip"
    
    # Legacy parameters for backward compatibility
    output_directory: str = "modeling_output"
    save_intermediate_results: bool = True
    generate_plots: bool = True
    
    # Report generation
    generate_html_report: bool = True
    generate_json_summary: bool = True
    include_model_explanations: bool = True
    
    # Comparison metrics
    comparison_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'r2', 'mape', 'rmse'
    ])
    
    # Visualization options
    plot_themes: str = "seaborn"
    figure_size: tuple = (12, 8)
    dpi: int = 300

@dataclass
class ModelingConfig:
    """Main configuration class for HPC Energy Modeling"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Data parameters
    historical_data_days: int = 30
    min_job_duration_seconds: int = 60
    max_job_duration_hours: int = 48
    
    # Validation parameters
    validation_tolerance_percent: float = 10.0
    statistical_significance_level: float = 0.05
    
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            database=DatabaseConfig(**config_data.get('database', {})),
            simulation=SimulationConfig(**config_data.get('simulation', {})),
            models=ModelConfig(**config_data.get('models', {})),
            output=OutputConfig(**config_data.get('output', {}))
        )
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file (alias for from_file)"""
        return cls.from_file(config_path)
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            database=DatabaseConfig.from_env()
        )
    
    def to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_data = {
            'database': self.database.__dict__,
            'simulation': self.simulation.__dict__,
            'models': self.models.__dict__,
            'output': self.output.__dict__
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate simulation parameters
        if self.simulation.time_step_seconds <= 0:
            issues.append("time_step_seconds must be positive")
        
        if self.simulation.simulation_duration_hours <= 0:
            issues.append("simulation_duration_hours must be positive")
        
        # Validate model parameters
        if self.models.train_test_split <= 0 or self.models.train_test_split >= 1:
            issues.append("train_test_split must be between 0 and 1")
        
        # Validate output directory
        try:
            Path(self.output.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        return issues

# Default configuration instance
default_config = ModelingConfig()