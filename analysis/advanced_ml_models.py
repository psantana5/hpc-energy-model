#!/usr/bin/env python3
"""
Advanced Machine Learning Models for HPC Energy Prediction

Provides sophisticated ML models, feature engineering, hyperparameter optimization,
and comprehensive model evaluation for energy consumption prediction in HPC environments.

Author: HPC Energy Model Project
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import pickle
import logging
from dataclasses import dataclass, asdict
import warnings
from collections import defaultdict
import joblib

# Machine Learning imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    TimeSeriesSplit, validation_curve, learning_curve
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE,
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    SGDRegressor, HuberRegressor, TheilSenRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    from tensorflow import keras
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    explained_variance: float
    training_time: float
    prediction_time: float
    cross_val_score_mean: float
    cross_val_score_std: float
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    importance_type: str  # 'permutation', 'tree_based', 'correlation'
    rank: int

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for HPC energy prediction
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.engineered_features = []
        
    def create_temporal_features(self, df: pd.DataFrame, 
                               timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Create temporal features from timestamp"""
        df = df.copy()
        
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Basic temporal features
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['day_of_month'] = df[timestamp_col].dt.day
            df['month'] = df[timestamp_col].dt.month
            df['quarter'] = df[timestamp_col].dt.quarter
            df['year'] = df[timestamp_col].dt.year
            
            # Cyclical encoding for temporal features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Business hours indicator
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            self.engineered_features.extend([
                'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'is_business_hours', 'is_weekend'
            ])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_cols: List[str],
                          lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features for time series"""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    lag_col = f'{col}_lag_{lag}'
                    df[lag_col] = df[col].shift(lag)
                    self.engineered_features.append(lag_col)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                              target_cols: List[str],
                              windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for window in windows:
                    # Rolling statistics
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                    df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
                    
                    # Rolling ratios
                    rolling_mean = df[col].rolling(window=window).mean()
                    df[f'{col}_ratio_to_rolling_mean_{window}'] = df[col] / rolling_mean
                    
                    self.engineered_features.extend([
                        f'{col}_rolling_mean_{window}', f'{col}_rolling_std_{window}',
                        f'{col}_rolling_min_{window}', f'{col}_rolling_max_{window}',
                        f'{col}_rolling_median_{window}', f'{col}_ratio_to_rolling_mean_{window}'
                    ])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_cols: List[str],
                                  max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Select most important features for interactions
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return df
        
        # Create pairwise interactions
        interaction_count = 0
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Multiplicative interaction
                interaction_col = f'{col1}_x_{col2}'
                df[interaction_col] = df[col1] * df[col2]
                self.engineered_features.append(interaction_col)
                
                # Ratio interaction (avoid division by zero)
                ratio_col = f'{col1}_div_{col2}'
                df[ratio_col] = df[col1] / (df[col2] + 1e-8)
                self.engineered_features.append(ratio_col)
                
                interaction_count += 2
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                 feature_cols: List[str],
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        df = df.copy()
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            return df
        
        # Select subset of features to avoid explosion
        selected_cols = available_cols[:5]  # Limit to prevent feature explosion
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[selected_cols])
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(selected_cols)
        
        # Add polynomial features (excluding original features)
        for i, name in enumerate(poly_feature_names):
            if name not in selected_cols:  # Skip original features
                df[f'poly_{name}'] = poly_features[:, i]
                self.engineered_features.append(f'poly_{name}')
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame,
                                  feature_cols: List[str]) -> pd.DataFrame:
        """Create statistical features"""
        df = df.copy()
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return df
        
        # Cross-feature statistics
        feature_matrix = df[available_cols].values
        
        # Row-wise statistics
        df['features_mean'] = np.mean(feature_matrix, axis=1)
        df['features_std'] = np.std(feature_matrix, axis=1)
        df['features_min'] = np.min(feature_matrix, axis=1)
        df['features_max'] = np.max(feature_matrix, axis=1)
        df['features_range'] = df['features_max'] - df['features_min']
        df['features_skew'] = pd.DataFrame(feature_matrix).skew(axis=1)
        df['features_kurtosis'] = pd.DataFrame(feature_matrix).kurtosis(axis=1)
        
        self.engineered_features.extend([
            'features_mean', 'features_std', 'features_min', 'features_max',
            'features_range', 'features_skew', 'features_kurtosis'
        ])
        
        return df
    
    def create_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create HPC workload-specific features"""
        df = df.copy()
        
        # CPU efficiency metrics
        if 'cpu_usage' in df.columns and 'cpu_cores' in df.columns:
            df['cpu_efficiency'] = df['cpu_usage'] / (df['cpu_cores'] * 100)
            self.engineered_features.append('cpu_efficiency')
        
        # Memory pressure indicators
        if 'memory_usage' in df.columns and 'memory_total' in df.columns:
            df['memory_pressure'] = df['memory_usage'] / df['memory_total']
            df['memory_available_ratio'] = 1 - df['memory_pressure']
            self.engineered_features.extend(['memory_pressure', 'memory_available_ratio'])
        
        # I/O intensity
        if 'disk_read' in df.columns and 'disk_write' in df.columns:
            df['io_total'] = df['disk_read'] + df['disk_write']
            df['io_read_ratio'] = df['disk_read'] / (df['io_total'] + 1e-8)
            df['io_write_ratio'] = df['disk_write'] / (df['io_total'] + 1e-8)
            self.engineered_features.extend(['io_total', 'io_read_ratio', 'io_write_ratio'])
        
        # Network activity
        if 'network_in' in df.columns and 'network_out' in df.columns:
            df['network_total'] = df['network_in'] + df['network_out']
            df['network_in_ratio'] = df['network_in'] / (df['network_total'] + 1e-8)
            df['network_out_ratio'] = df['network_out'] / (df['network_total'] + 1e-8)
            self.engineered_features.extend(['network_total', 'network_in_ratio', 'network_out_ratio'])
        
        # Power efficiency indicators
        if 'power_consumption' in df.columns and 'cpu_usage' in df.columns:
            df['power_per_cpu_usage'] = df['power_consumption'] / (df['cpu_usage'] + 1e-8)
            self.engineered_features.append('power_per_cpu_usage')
        
        # Thermal efficiency
        if 'temperature' in df.columns and 'power_consumption' in df.columns:
            df['thermal_efficiency'] = df['power_consumption'] / (df['temperature'] + 1e-8)
            self.engineered_features.append('thermal_efficiency')
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame,
                            target_col: str = 'energy_consumption',
                            timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Apply all feature engineering techniques"""
        logger.info("Starting comprehensive feature engineering")
        
        # Reset engineered features list
        self.engineered_features = []
        
        # Identify numeric columns for feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Apply feature engineering steps
        df = self.create_temporal_features(df, timestamp_col)
        df = self.create_workload_features(df)
        df = self.create_lag_features(df, [target_col] + numeric_cols[:5])  # Limit to prevent explosion
        df = self.create_rolling_features(df, [target_col] + numeric_cols[:3])
        df = self.create_interaction_features(df, numeric_cols[:5])
        df = self.create_statistical_features(df, numeric_cols[:10])
        df = self.create_polynomial_features(df, numeric_cols[:3], degree=2)
        
        logger.info(f"Feature engineering completed. Created {len(self.engineered_features)} new features")
        
        return df

class AdvancedMLPredictor:
    """
    Advanced machine learning predictor with multiple sophisticated models
    """
    
    def __init__(self, output_dir: str = "ml_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.model_performances = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_selector = None
        self.scaler = None
        self.best_model = None
        self.feature_importance_analysis = {}
        
        # Initialize model configurations
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all available models with default configurations"""
        self.models = {
            # Linear models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'bayesian_ridge': BayesianRidge(),
            'huber': HuberRegressor(),
            
            # Tree-based models
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            
            # Other models
            'svr': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, random_state=42, eval_metric='rmse'
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100, random_state=42, verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostRegressor(
                iterations=100, random_state=42, verbose=False
            )
        
        # Ensemble models
        self.models['voting_regressor'] = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=50, random_state=42))
        ])
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'energy_consumption',
                    test_size: float = 0.2,
                    apply_feature_engineering: bool = True,
                    feature_selection_k: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with feature engineering and selection"""
        logger.info("Preparing data for training")
        
        # Apply feature engineering
        if apply_feature_engineering:
            df = self.feature_engineer.engineer_all_features(df, target_col)
        
        # Remove rows with NaN values (created by lag/rolling features)
        df = df.dropna()
        
        if df.empty:
            raise ValueError("No data remaining after feature engineering and NaN removal")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if feature_selection_k and feature_selection_k < X_train.shape[1]:
            self.feature_selector = SelectKBest(
                score_func=f_regression, k=min(feature_selection_k, X_train.shape[1])
            )
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        
        logger.info(f"Final feature matrix shape: {X_train_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cv_folds: int = 5) -> Dict[str, ModelPerformance]:
        """Train all models and evaluate performance"""
        logger.info(f"Training {len(self.models)} models")
        
        performances = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Time training
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Time prediction
                start_time = datetime.now()
                y_pred = model.predict(X_test)
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                explained_var = explained_variance_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=cv_folds, scoring='neg_mean_squared_error')
                cv_rmse_scores = np.sqrt(-cv_scores)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(enumerate(model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(enumerate(np.abs(model.coef_)))
                
                # Store performance
                performance = ModelPerformance(
                    model_name=name,
                    mse=mse,
                    rmse=rmse,
                    mae=mae,
                    mape=mape,
                    r2=r2,
                    explained_variance=explained_var,
                    training_time=training_time,
                    prediction_time=prediction_time,
                    cross_val_score_mean=cv_rmse_scores.mean(),
                    cross_val_score_std=cv_rmse_scores.std(),
                    feature_importance=feature_importance,
                    hyperparameters=model.get_params() if hasattr(model, 'get_params') else None
                )
                
                performances[name] = performance
                logger.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, Training time: {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        self.model_performances = performances
        
        # Find best model based on RMSE
        if performances:
            best_model_name = min(performances.keys(), 
                                key=lambda x: performances[x].rmse)
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} (RMSE: {performances[best_model_name].rmse:.4f})")
        
        return performances
    
    def hyperparameter_optimization(self, model_name: str,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  optimization_method: str = 'random_search',
                                  n_trials: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter optimization for a specific model"""
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        if optimization_method == 'grid_search':
            search = GridSearchCV(
                model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        elif optimization_method == 'random_search':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_trials, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1, 
                random_state=42, verbose=1
            )
        elif optimization_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optuna_optimization(model_name, X_train, y_train, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = search.best_estimator_
        
        optimization_results = {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,  # Convert back to positive RMSE
            'cv_results': search.cv_results_
        }
        
        logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info(f"Best CV score: {-search.best_score_:.4f}")
        
        return optimization_results
    
    def _optuna_optimization(self, model_name: str, 
                           X_train: np.ndarray, y_train: np.ndarray,
                           n_trials: int = 50) -> Dict[str, Any]:
        """Optuna-based hyperparameter optimization"""
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(**params, random_state=42)
            
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = xgb.XGBRegressor(**params, random_state=42, eval_metric='rmse')
            
            else:
                raise ValueError(f"Optuna optimization not implemented for {model_name}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            return np.sqrt(-cv_scores.mean())
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        best_params = study.best_params
        if model_name == 'random_forest':
            self.models[model_name] = RandomForestRegressor(**best_params, random_state=42)
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            self.models[model_name] = xgb.XGBRegressor(**best_params, random_state=42, eval_metric='rmse')
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def analyze_feature_importance(self, X_train: np.ndarray, y_train: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, List[FeatureImportance]]:
        """Comprehensive feature importance analysis"""
        logger.info("Analyzing feature importance")
        
        importance_results = {}
        
        # Tree-based feature importance
        tree_models = ['random_forest', 'extra_trees', 'gradient_boosting']
        if XGBOOST_AVAILABLE:
            tree_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            tree_models.append('lightgbm')
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_list = []
                    
                    for i, importance in enumerate(importances):
                        if i < len(feature_names):
                            feature_importance_list.append(FeatureImportance(
                                feature_name=feature_names[i],
                                importance_score=importance,
                                importance_type='tree_based',
                                rank=i + 1
                            ))
                    
                    # Sort by importance
                    feature_importance_list.sort(key=lambda x: x.importance_score, reverse=True)
                    
                    # Update ranks
                    for rank, fi in enumerate(feature_importance_list, 1):
                        fi.rank = rank
                    
                    importance_results[f'{model_name}_tree_importance'] = feature_importance_list
        
        # Permutation importance
        if self.best_model:
            try:
                from sklearn.inspection import permutation_importance
                
                perm_importance = permutation_importance(
                    self.best_model, X_train, y_train, 
                    n_repeats=5, random_state=42, n_jobs=-1
                )
                
                perm_importance_list = []
                for i, importance in enumerate(perm_importance.importances_mean):
                    if i < len(feature_names):
                        perm_importance_list.append(FeatureImportance(
                            feature_name=feature_names[i],
                            importance_score=importance,
                            importance_type='permutation',
                            rank=i + 1
                        ))
                
                # Sort by importance
                perm_importance_list.sort(key=lambda x: x.importance_score, reverse=True)
                
                # Update ranks
                for rank, fi in enumerate(perm_importance_list, 1):
                    fi.rank = rank
                
                importance_results['permutation_importance'] = perm_importance_list
                
            except ImportError:
                logger.warning("Permutation importance requires scikit-learn >= 0.22")
        
        # Correlation-based importance
        if len(feature_names) == X_train.shape[1]:
            correlations = []
            for i, feature_name in enumerate(feature_names):
                corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
                correlations.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=abs(corr) if not np.isnan(corr) else 0,
                    importance_type='correlation',
                    rank=i + 1
                ))
            
            # Sort by importance
            correlations.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Update ranks
            for rank, fi in enumerate(correlations, 1):
                fi.rank = rank
            
            importance_results['correlation_importance'] = correlations
        
        self.feature_importance_analysis = importance_results
        return importance_results
    
    def create_ensemble_model(self, top_models: List[str] = None) -> Any:
        """Create an ensemble model from top performing models"""
        if not self.model_performances:
            raise ValueError("No trained models available for ensemble")
        
        if top_models is None:
            # Select top 3 models based on RMSE
            sorted_models = sorted(self.model_performances.items(), 
                                 key=lambda x: x[1].rmse)
            top_models = [name for name, _ in sorted_models[:3]]
        
        # Create ensemble
        estimators = [(name, self.models[name]) for name in top_models 
                     if name in self.models]
        
        if len(estimators) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        ensemble = VotingRegressor(estimators)
        self.models['ensemble'] = ensemble
        
        logger.info(f"Created ensemble model with: {[name for name, _ in estimators]}")
        
        return ensemble
    
    def generate_model_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive model comparison report"""
        if not self.model_performances:
            return {}
        
        # Convert performances to DataFrame for analysis
        perf_data = []
        for name, perf in self.model_performances.items():
            perf_dict = asdict(perf)
            perf_data.append(perf_dict)
        
        perf_df = pd.DataFrame(perf_data)
        
        # Rankings
        rankings = {
            'rmse_ranking': perf_df.nsmallest(5, 'rmse')[['model_name', 'rmse']].to_dict('records'),
            'r2_ranking': perf_df.nlargest(5, 'r2')[['model_name', 'r2']].to_dict('records'),
            'training_time_ranking': perf_df.nsmallest(5, 'training_time')[['model_name', 'training_time']].to_dict('records'),
            'prediction_time_ranking': perf_df.nsmallest(5, 'prediction_time')[['model_name', 'prediction_time']].to_dict('records')
        }
        
        # Statistics
        statistics = {
            'best_rmse': perf_df['rmse'].min(),
            'worst_rmse': perf_df['rmse'].max(),
            'best_r2': perf_df['r2'].max(),
            'worst_r2': perf_df['r2'].min(),
            'avg_training_time': perf_df['training_time'].mean(),
            'avg_prediction_time': perf_df['prediction_time'].mean()
        }
        
        # Model recommendations
        best_overall = perf_df.loc[perf_df['rmse'].idxmin()]['model_name']
        fastest_training = perf_df.loc[perf_df['training_time'].idxmin()]['model_name']
        fastest_prediction = perf_df.loc[perf_df['prediction_time'].idxmin()]['model_name']
        
        recommendations = {
            'best_accuracy': best_overall,
            'fastest_training': fastest_training,
            'fastest_prediction': fastest_prediction,
            'production_recommendation': best_overall if perf_df.loc[perf_df['model_name'] == best_overall]['prediction_time'].iloc[0] < 1.0 else fastest_prediction
        }
        
        return {
            'rankings': rankings,
            'statistics': statistics,
            'recommendations': recommendations,
            'detailed_results': perf_data
        }
    
    def save_models_and_results(self, experiment_name: str = None):
        """Save all models and results"""
        if experiment_name is None:
            experiment_name = f"ml_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            try:
                model_file = models_dir / f"{name}.joblib"
                joblib.dump(model, model_file)
            except Exception as e:
                logger.warning(f"Could not save model {name}: {e}")
        
        # Save feature engineer and scaler
        if self.feature_engineer:
            joblib.dump(self.feature_engineer, experiment_dir / "feature_engineer.joblib")
        
        if self.scaler:
            joblib.dump(self.scaler, experiment_dir / "scaler.joblib")
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, experiment_dir / "feature_selector.joblib")
        
        # Save performance results
        if self.model_performances:
            perf_data = {name: asdict(perf) for name, perf in self.model_performances.items()}
            with open(experiment_dir / "model_performances.json", 'w') as f:
                json.dump(perf_data, f, indent=2, default=str)
        
        # Save feature importance analysis
        if self.feature_importance_analysis:
            importance_data = {}
            for analysis_name, importance_list in self.feature_importance_analysis.items():
                importance_data[analysis_name] = [asdict(fi) for fi in importance_list]
            
            with open(experiment_dir / "feature_importance.json", 'w') as f:
                json.dump(importance_data, f, indent=2)
        
        # Generate and save comparison report
        comparison_report = self.generate_model_comparison_report()
        with open(experiment_dir / "model_comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        logger.info(f"Models and results saved to {experiment_dir}")
        
        return experiment_dir
    
    def load_experiment(self, experiment_dir: str):
        """Load a saved experiment"""
        experiment_path = Path(experiment_dir)
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment directory {experiment_dir} does not exist")
        
        # Load models
        models_dir = experiment_path / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.joblib"):
                model_name = model_file.stem
                try:
                    self.models[model_name] = joblib.load(model_file)
                except Exception as e:
                    logger.warning(f"Could not load model {model_name}: {e}")
        
        # Load feature engineer and scaler
        if (experiment_path / "feature_engineer.joblib").exists():
            self.feature_engineer = joblib.load(experiment_path / "feature_engineer.joblib")
        
        if (experiment_path / "scaler.joblib").exists():
            self.scaler = joblib.load(experiment_path / "scaler.joblib")
        
        if (experiment_path / "feature_selector.joblib").exists():
            self.feature_selector = joblib.load(experiment_path / "feature_selector.joblib")
        
        # Load performance results
        perf_file = experiment_path / "model_performances.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            
            self.model_performances = {}
            for name, data in perf_data.items():
                self.model_performances[name] = ModelPerformance(**data)
        
        logger.info(f"Experiment loaded from {experiment_dir}")

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic HPC energy data
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'cpu_usage': np.random.normal(60, 20, n_samples).clip(0, 100),
        'memory_usage': np.random.normal(70, 15, n_samples).clip(0, 100),
        'cpu_cores': np.random.choice([8, 16, 32, 64], n_samples),
        'memory_total': np.random.choice([32, 64, 128, 256], n_samples),
        'disk_read': np.random.exponential(10, n_samples),
        'disk_write': np.random.exponential(5, n_samples),
        'network_in': np.random.exponential(100, n_samples),
        'network_out': np.random.exponential(50, n_samples),
        'temperature': np.random.normal(45, 10, n_samples).clip(20, 80),
        'power_consumption': np.random.normal(150, 30, n_samples).clip(50, 300)
    }
    
    # Create energy consumption target (synthetic relationship)
    data['energy_consumption'] = (
        data['cpu_usage'] * 0.5 + 
        data['memory_usage'] * 0.3 + 
        data['power_consumption'] * 0.8 + 
        np.random.normal(0, 10, n_samples)
    )
    
    df = pd.DataFrame(data)
    
    # Initialize predictor
    predictor = AdvancedMLPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)
    
    # Train all models
    performances = predictor.train_all_models(X_train, y_train, X_test, y_test)
    
    # Optimize best model
    if performances:
        best_model_name = min(performances.keys(), key=lambda x: performances[x].rmse)
        print(f"\nOptimizing {best_model_name}...")
        
        optimization_results = predictor.hyperparameter_optimization(
            best_model_name, X_train, y_train, optimization_method='random_search'
        )
        
        # Re-train with optimized parameters
        optimized_performances = predictor.train_all_models(X_train, y_train, X_test, y_test)
    
    # Feature importance analysis
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    importance_analysis = predictor.analyze_feature_importance(X_train, y_train, feature_names)
    
    # Create ensemble
    ensemble = predictor.create_ensemble_model()
    
    # Generate comparison report
    comparison_report = predictor.generate_model_comparison_report()
    
    # Save everything
    experiment_dir = predictor.save_models_and_results("advanced_ml_demo")
    
    print(f"\nAdvanced ML experiment completed!")
    print(f"Results saved to: {experiment_dir}")
    print(f"Best model: {comparison_report['recommendations']['best_accuracy']}")
    print(f"Best RMSE: {comparison_report['statistics']['best_rmse']:.4f}")