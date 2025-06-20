# Thermal Behavior Prediction Model

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from ..utils.config import ModelingConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import ModelingConfig

logger = logging.getLogger(__name__)

class ThermalPredictor:
    """
    Machine learning model for predicting thermal behavior of HPC nodes
    
    This class implements multiple algorithms to predict temperature patterns
    based on workload characteristics, resource usage, and environmental factors.
    """
    
    def __init__(self, config: ModelingConfig):
        self.config = config
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = ['avg_cpu_temp']  # Single target now
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variables for training
        
        Args:
            df: DataFrame with node metrics
            
        Returns:
            Tuple of (features, targets)
        """
        logger.info("Preparing features for thermal prediction")
        logger.info(f"Input DataFrame columns: {list(df.columns)}")
        logger.info(f"Input DataFrame shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Define feature columns - use the same features as energy predictor for consistency
        numeric_features = [
            'cpu_cores', 'memory_mb', 'duration_seconds',
            'cpu_usage', 'memory_usage', 'io_read_mbps', 'io_write_mbps',
            'network_rx_mbps', 'network_tx_mbps', 'avg_cpu_temp', 'peak_cpu_temp'
        ]
        
        categorical_features = [
            'job_type', 'partition', 'workload_pattern'
        ]
        
        # Target variables - use avg_cpu_temp as primary target
        target_col = None
        possible_targets = ['avg_cpu_temp', 'peak_cpu_temp', 'cpu_temp']
        
        for col in possible_targets:
            if col in data.columns:
                target_col = col
                logger.info(f"Using thermal target column: {target_col}")
                break
        
        if target_col is None:
            logger.warning("No thermal target columns found - using dummy target")
            target_col = 'dummy_temp'
            data[target_col] = 50.0  # Default temperature
        
        # Filter available features
        available_numeric = [f for f in numeric_features if f in data.columns]
        available_categorical = [f for f in categorical_features if f in data.columns]
        
        logger.info(f"Available numeric features: {available_numeric}")
        logger.info(f"Available categorical features: {available_categorical}")
        logger.info(f"Using {len(available_numeric)} numeric and {len(available_categorical)} categorical features")
        
        # Environmental features (if available)
        environmental_features = [
            'ambient_temp', 'humidity', 'cooling_efficiency'
        ]
        available_environmental = [f for f in environmental_features if f in data.columns]
        
        # Handle missing values in numeric features
        for feature in available_numeric + available_environmental:
            if feature in data.columns:
                # Ensure numeric column is float type
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
                data[feature] = data[feature].fillna(data[feature].median())
        
        # Handle missing values in categorical features
        for feature in available_categorical:
            data[feature] = data[feature].astype(str).fillna('unknown')
        
        # Encode categorical features
        for feature in available_categorical:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                data[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(data[feature].astype(str))
            else:
                # Handle unseen categories during prediction
                unique_values = set(data[feature].astype(str).unique())
                known_values = set(self.label_encoders[feature].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to encoder
                    all_values = list(known_values) + list(new_values)
                    self.label_encoders[feature].classes_ = np.array(all_values)
                
                data[f'{feature}_encoded'] = self.label_encoders[feature].transform(data[feature].astype(str))
        
        # Create derived features
        if 'cpu_cores' in data.columns and 'duration_seconds' in data.columns:
            data['cpu_hours'] = data['cpu_cores'] * data['duration_seconds'] / 3600.0
        
        if 'memory_mb' in data.columns and 'duration_seconds' in data.columns:
            data['memory_hours'] = data['memory_mb'] * data['duration_seconds'] / 3600.0
        
        if 'cpu_usage' in data.columns and 'cpu_cores' in data.columns:
            data['effective_cpu_usage'] = data['cpu_usage'] * data['cpu_cores'] / 100.0
        
        if 'avg_cpu_temp' in data.columns and 'peak_cpu_temp' in data.columns:
            data['temp_variance'] = data['peak_cpu_temp'] - data['avg_cpu_temp']
        
        # Compile final feature list
        feature_columns = available_numeric.copy()
        feature_columns.extend([f'{f}_encoded' for f in available_categorical])
        feature_columns.extend(available_environmental)
        
        # Add derived features
        derived_features = ['cpu_hours', 'memory_hours', 'effective_cpu_usage', 'temp_variance']
        for feature in derived_features:
            if feature in data.columns:
                feature_columns.append(feature)
        
        logger.info(f"Final feature columns before filtering: {feature_columns}")
        
        # Filter out features that don't exist in data
        final_features = []
        for feature in feature_columns:
            if feature in data.columns:
                final_features.append(feature)
            else:
                logger.warning(f"Feature {feature} not found in data columns")
        
        feature_columns = final_features
        logger.info(f"Final feature columns after filtering: {feature_columns}")
        logger.info(f"Total features selected: {len(feature_columns)}")
        
        if len(feature_columns) == 0:
            logger.error("No valid features found!")
            logger.error(f"Available columns: {list(data.columns)}")
            raise ValueError("No valid features available for training")
        
        self.feature_names = feature_columns
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_col].values
        
        logger.info(f"Extracted features shape: {X.shape}")
        logger.info(f"Extracted target shape: {y.shape}")
        
        # Ensure X contains only numeric values
        try:
            X = X.astype(np.float64)
            logger.info(f"Successfully converted features to float64")
        except Exception as e:
            logger.error(f"Error converting features to float64: {e}")
            logger.error(f"Feature data types: {[type(x) for x in X.flatten()[:10]]}")
            raise
        
        # Remove rows with invalid values in features or target
        # Check for NaN/inf in features
        feature_valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        
        # Check for invalid target values
        target_valid_mask = ~(np.isnan(y) | np.isinf(y))
        
        # Combine masks
        valid_mask = feature_valid_mask & target_valid_mask
        
        logger.info(f"Valid samples before filtering: {len(valid_mask)}")
        logger.info(f"Valid samples after filtering: {valid_mask.sum()}")
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Final X shape: {X.shape}")
        logger.info(f"Final y shape: {y.shape}")
        
        if X.shape[1] == 0:
            logger.error("Features array has 0 columns after processing!")
            logger.error(f"Original feature_columns: {feature_columns}")
            logger.error(f"Data columns: {list(data.columns)}")
            raise ValueError("Features array has 0 columns")
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the thermal prediction model
        
        Args:
            df: DataFrame with node metrics
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training thermal prediction models")
        
        # Check if DataFrame is empty
        if df is None or len(df) == 0:
            logger.warning("No training data available - creating dummy model")
            return {
                'status': 'no_data',
                'message': 'No training data available',
                'models_trained': 0
            }
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if len(X) == 0:
            logger.warning("No valid training data after preprocessing - creating dummy model")
            return {
                'status': 'no_valid_data',
                'message': 'No valid training data after preprocessing',
                'models_trained': 0
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            
            try:
                # Use scaled features for linear models
                if name in ['linear', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics for single target
                target_metrics = {}
                overall_r2 = []
                
                # Since we now have a single target, handle as 1D arrays
                target_name = self.target_names[0]
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
                
                target_metrics[target_name] = {
                    'mae': mae,
                    'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2': r2,
                        'mape': mape
                }
                
                overall_r2.append(r2)
                
                # Overall model score (single target R²)
                model_scores[name] = {
                    'overall_r2': r2,  # Single target, so just use the r2 value directly
                    'target_metrics': target_metrics
                }
                
                logger.info(f"{name} - Overall R²: {r2:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
                model_scores[name] = {'overall_r2': -np.inf}
        
        # Select best model based on overall R²
        self.best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['overall_r2'])
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} (R² = {model_scores[self.best_model_name]['overall_r2']:.3f})")
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train, y_train)
        
        # Store training metrics
        self.training_metrics = {
            'best_model': self.best_model_name,
            'model_scores': model_scores,
            'feature_importance': self.feature_importance,
            'target_names': self.target_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return self.training_metrics
    
    def _calculate_feature_importance(self, X_train: np.ndarray, y_train: np.ndarray):
        """Calculate feature importance"""
        if hasattr(self.best_model, 'estimators_'):
            # For MultiOutputRegressor, get importance from first estimator
            if hasattr(self.best_model.estimators_[0], 'feature_importances_'):
                importances = self.best_model.estimators_[0].feature_importances_
                self.feature_importance = dict(zip(self.feature_names, importances))
                
                # Sort by importance
                self.feature_importance = dict(
                    sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                
                logger.info("Top 5 most important features for thermal prediction:")
                for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5]):
                    logger.info(f"  {i+1}. {feature}: {importance:.3f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict thermal behavior for new data
        
        Args:
            df: DataFrame with node characteristics
            
        Returns:
            Array of predicted thermal values
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Scale features if needed
        if self.best_model_name in ['linear', 'ridge']:
            X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.best_model.predict(X)
        
        return predictions
    
    def predict_temperature_profile(self, job_specs: Dict[str, Any], duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Predict temperature profile over time for a job
        
        Args:
            job_specs: Job specifications
            duration_minutes: Duration to simulate
            
        Returns:
            Dictionary with temperature profile
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create time series data
        time_points = np.arange(0, duration_minutes, 5)  # Every 5 minutes
        profile_data = []
        
        for t in time_points:
            # Create data point for this time
            data_point = job_specs.copy()
            data_point['duration_seconds'] = t * 60
            
            # Add time-based features
            data_point['time_progress'] = t / duration_minutes
            
            profile_data.append(data_point)
        
        # Convert to DataFrame and predict
        df = pd.DataFrame(profile_data)
        predictions = self.predict(df)
        
        # Create temperature profile
        profile = {
            'time_minutes': time_points.tolist(),
            'predictions': {}
        }
        
        for i, target_name in enumerate(self.target_names):
            profile['predictions'][target_name] = predictions[:, i].tolist()
        
        # Add summary statistics
        profile['summary'] = {}
        for i, target_name in enumerate(self.target_names):
            values = predictions[:, i]
            profile['summary'][target_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return profile
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on new data
        
        Args:
            df: DataFrame with node metrics including actual thermal data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features and targets
        X, y_true = self.prepare_features(df)
        
        if len(X) == 0:
            return {'error': 'No valid evaluation data'}
        
        # Make predictions
        if self.best_model_name in ['linear', 'ridge']:
            X = self.scaler.transform(X)
        
        y_pred = self.best_model.predict(X)
        
        # Calculate metrics for single target
        metrics = {'targets': {}, 'overall': {}}
        overall_r2 = []
        
        # Since we now have a single target, handle as 1D arrays
        target_name = self.target_names[0]
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        metrics['targets'][target_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mape': mape
            }
        
        overall_r2.append(r2)
        
        metrics['overall'] = {
            'r2': r2,  # Single target, so just use the r2 value directly
            'samples': len(y_true)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Thermal model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        
        logger.info(f"Thermal model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model
        
        Returns:
            Dictionary with model information
        """
        if self.best_model is None:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'best_model': self.best_model_name,
            'feature_count': len(self.feature_names),
            'target_count': len(self.target_names),
            'targets': self.target_names,
            'top_features': list(self.feature_importance.keys())[:5] if self.feature_importance else [],
            'training_metrics': self.training_metrics
        }
    
    def analyze_thermal_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze thermal efficiency patterns
        
        Args:
            df: DataFrame with node metrics
            
        Returns:
            Dictionary with thermal efficiency analysis
        """
        analysis = {
            'efficiency_metrics': {},
            'temperature_ranges': {},
            'correlations': {},
            'recommendations': []
        }
        
        # Calculate thermal efficiency metrics
        if 'avg_power_watts' in df.columns and 'avg_cpu_temp' in df.columns:
            df['thermal_efficiency'] = df['avg_power_watts'] / (df['avg_cpu_temp'] - 20)  # Assuming 20°C baseline
            
            analysis['efficiency_metrics'] = {
                'mean_efficiency': float(df['thermal_efficiency'].mean()),
                'std_efficiency': float(df['thermal_efficiency'].std()),
                'min_efficiency': float(df['thermal_efficiency'].min()),
                'max_efficiency': float(df['thermal_efficiency'].max())
            }
        
        # Temperature range analysis
        temp_cols = [col for col in ['avg_cpu_temp', 'peak_cpu_temp'] if col in df.columns]
        for col in temp_cols:
            analysis['temperature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'percentiles': {
                    '25': float(df[col].quantile(0.25)),
                    '50': float(df[col].quantile(0.50)),
                    '75': float(df[col].quantile(0.75)),
                    '95': float(df[col].quantile(0.95))
                }
            }
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(temp_cols) > 0 and len(numeric_cols) > 1:
            for temp_col in temp_cols:
                correlations = df[numeric_cols].corrwith(df[temp_col]).abs().sort_values(ascending=False)
                analysis['correlations'][temp_col] = correlations.head(5).to_dict()
        
        # Generate recommendations
        if 'avg_cpu_temp' in df.columns:
            avg_temp = df['avg_cpu_temp'].mean()
            if avg_temp > 80:
                analysis['recommendations'].append("High average temperatures detected. Consider improving cooling.")
            elif avg_temp > 70:
                analysis['recommendations'].append("Moderate temperatures. Monitor thermal performance.")
            else:
                analysis['recommendations'].append("Good thermal performance maintained.")
        
        if 'peak_cpu_temp' in df.columns:
            max_peak = df['peak_cpu_temp'].max()
            if max_peak > 90:
                analysis['recommendations'].append("Critical peak temperatures detected. Immediate cooling optimization needed.")
            elif max_peak > 85:
                analysis['recommendations'].append("High peak temperatures. Consider workload distribution optimization.")
        
        return analysis