# Energy Consumption Prediction Model

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

class EnergyPredictor:
    """
    Machine learning model for predicting energy consumption of HPC jobs
    
    This class implements multiple algorithms to predict energy consumption
    based on job characteristics, resource usage, and historical patterns.
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
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variables for training
        
        Args:
            df: DataFrame with job metrics
            
        Returns:
            Tuple of (features, targets)
        """
        logger.info("Preparing features for energy prediction")
        logger.info(f"Input DataFrame columns: {list(df.columns)}")
        logger.info(f"Input DataFrame shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Define feature columns
        numeric_features = [
            'cpu_cores', 'memory_mb', 'duration_seconds',
            'cpu_usage', 'memory_usage', 'io_read_mbps', 'io_write_mbps',
            'network_rx_mbps', 'network_tx_mbps', 'avg_cpu_temp', 'peak_cpu_temp'
        ]
        
        categorical_features = [
            'job_type', 'partition', 'workload_pattern'
        ]
        
        # Target variable - try multiple possible column names
        target_col = None
        possible_targets = ['estimated_energy_wh', 'energy_wh', 'avg_power_watts']
        
        for col in possible_targets:
            if col in data.columns:
                target_col = col
                logger.info(f"Using target column: {target_col}")
                break
        
        if target_col is None:
            # Try to create energy target from power and duration
            if 'avg_power_watts' in data.columns and 'duration_seconds' in data.columns:
                target_col = 'calculated_energy_wh'
                data[target_col] = data['avg_power_watts'] * data['duration_seconds'] / 3600.0
                logger.info("Created energy target from power and duration")
            elif 'estimated_power_w' in data.columns and 'duration_seconds' in data.columns:
                target_col = 'calculated_energy_wh'
                data[target_col] = data['estimated_power_w'] * data['duration_seconds'] / 3600.0
                logger.info("Created energy target from estimated power and duration")
            else:
                raise ValueError(f"No suitable energy target column found. Available columns: {list(data.columns)}")
        
        # Filter available features
        available_numeric = [f for f in numeric_features if f in data.columns]
        available_categorical = [f for f in categorical_features if f in data.columns]
        
        logger.info(f"Available numeric features: {available_numeric}")
        logger.info(f"Available categorical features: {available_categorical}")
        logger.info(f"Using {len(available_numeric)} numeric and {len(available_categorical)} categorical features")
        
        # Handle missing values in numeric features
        for feature in available_numeric:
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
        target_valid_mask = ~(np.isnan(y) | np.isinf(y) | (y <= 0))
        
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
        Train the energy prediction model
        
        Args:
            df: DataFrame with job metrics
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training energy prediction models")
        
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
        
        logger.info(f"Before train/test split - X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"After train/test split - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
        if X_train.shape[1] == 0:
            logger.error("X_train has 0 features after train/test split!")
            raise ValueError("X_train has 0 features")
        
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
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'mape': np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
                }
                
                logger.info(f"{name} - R²: {r2:.3f}, MAE: {mae:.2f}, MAPE: {model_scores[name]['mape']:.1f}%")
                
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
                model_scores[name] = {'r2': -np.inf}
        
        # Select best model based on R²
        self.best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} (R² = {model_scores[self.best_model_name]['r2']:.3f})")
        
        # Hyperparameter tuning for best model
        if self.best_model_name == 'random_forest':
            self._tune_random_forest(X_train, y_train)
        elif self.best_model_name == 'gradient_boosting':
            self._tune_gradient_boosting(X_train, y_train)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train, y_train)
        
        # Store training metrics
        self.training_metrics = {
            'best_model': self.best_model_name,
            'model_scores': model_scores,
            'feature_importance': self.feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return self.training_metrics
    
    def _tune_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tune Random Forest hyperparameters"""
        logger.info("Tuning Random Forest hyperparameters")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.models['random_forest'] = grid_search.best_estimator_
        
        if self.best_model_name == 'random_forest':
            self.best_model = self.models['random_forest']
        
        logger.info(f"Best RF parameters: {grid_search.best_params_}")
    
    def _tune_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tune Gradient Boosting hyperparameters"""
        logger.info("Tuning Gradient Boosting hyperparameters")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.models['gradient_boosting'] = grid_search.best_estimator_
        
        if self.best_model_name == 'gradient_boosting':
            self.best_model = self.models['gradient_boosting']
        
        logger.info(f"Best GB parameters: {grid_search.best_params_}")
    
    def _calculate_feature_importance(self, X_train: np.ndarray, y_train: np.ndarray):
        """Calculate feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info("Top 5 most important features:")
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5]):
                logger.info(f"  {i+1}. {feature}: {importance:.3f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict energy consumption for new jobs
        
        Args:
            df: DataFrame with job characteristics
            
        Returns:
            Array of predicted energy consumption values
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
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict energy consumption with confidence intervals
        
        Args:
            df: DataFrame with job characteristics
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        predictions = self.predict(df)
        
        # For tree-based models, use prediction variance
        if self.best_model_name in ['random_forest', 'gradient_boosting']:
            if hasattr(self.best_model, 'estimators_'):
                # Get predictions from all trees
                X, _ = self.prepare_features(df)
                tree_predictions = np.array([
                    tree.predict(X) for tree in self.best_model.estimators_
                ])
                
                # Calculate standard deviation as confidence measure
                confidence = np.std(tree_predictions, axis=0)
            else:
                # Fallback: use 10% of prediction as confidence
                confidence = predictions * 0.1
        else:
            # For linear models, use 10% of prediction as confidence
            confidence = predictions * 0.1
        
        return predictions, confidence
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on new data
        
        Args:
            df: DataFrame with job metrics including actual energy consumption
            
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
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100,
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
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
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
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        
        logger.info(f"Model loaded from {filepath}")
    
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
            'top_features': list(self.feature_importance.keys())[:5] if self.feature_importance else [],
            'training_metrics': self.training_metrics
        }
    
    def explain_prediction(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            job_data: Dictionary with job characteristics
            
        Returns:
            Dictionary with prediction explanation
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([job_data])
        
        # Make prediction
        prediction = self.predict(df)[0]
        
        # Get feature contributions (simplified)
        X, _ = self.prepare_features(df)
        
        if self.best_model_name in ['linear', 'ridge']:
            X = self.scaler.transform(X)
        
        explanation = {
            'predicted_energy_wh': prediction,
            'model_used': self.best_model_name,
            'confidence': 'medium',  # Simplified confidence
            'key_factors': []
        }
        
        # Add feature contributions for tree-based models
        if hasattr(self.best_model, 'feature_importances_') and self.feature_importance:
            top_features = list(self.feature_importance.keys())[:3]
            for feature in top_features:
                if feature in self.feature_names:
                    idx = self.feature_names.index(feature)
                    value = X[0][idx] if idx < len(X[0]) else 'N/A'
                    importance = self.feature_importance[feature]
                    
                    explanation['key_factors'].append({
                        'feature': feature,
                        'value': value,
                        'importance': importance
                    })
        
        return explanation