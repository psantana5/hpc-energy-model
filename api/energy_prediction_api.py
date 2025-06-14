#!/usr/bin/env python3
"""
Energy Prediction API for HPC Energy Model

Provides REST API endpoints for energy consumption prediction,
job scheduling recommendations, and real-time monitoring.

Author: HPC Energy Model Project
License: MIT
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import psycopg2
import redis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class EnergyPredictor:
    """
    Energy consumption prediction service
    """
    
    def __init__(self, model_path: str = None, db_config: Dict = None, redis_config: Dict = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_metadata = {}
        self.db_config = db_config or {}
        self.redis_client = None
        
        # Initialize Redis if config provided
        if redis_config:
            try:
                self.redis_client = redis.Redis(**redis_config)
                self.redis_client.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a simple default model
            self._create_default_model()
    
    def _create_default_model(self):
        """
        Create a simple default model for demonstration
        """
        logger.info("Creating default energy prediction model")
        
        # Default feature names
        self.feature_names = [
            'duration_seconds', 'cpu_usage_percent', 'memory_usage_percent',
            'io_read_mbps', 'io_write_mbps', 'cpu_cores', 'job_type_encoded'
        ]
        
        # Create a simple model with reasonable coefficients
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Generate some synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Synthetic features
        duration = np.random.uniform(60, 3600, n_samples)  # 1 min to 1 hour
        cpu_usage = np.random.uniform(10, 100, n_samples)
        memory_usage = np.random.uniform(5, 95, n_samples)
        io_read = np.random.uniform(0, 100, n_samples)
        io_write = np.random.uniform(0, 50, n_samples)
        cpu_cores = np.random.choice([1, 2, 4, 8, 16], n_samples)
        job_type = np.random.choice([0, 1, 2], n_samples)  # 0=CPU, 1=IO, 2=Mixed
        
        X = np.column_stack([duration, cpu_usage, memory_usage, io_read, io_write, cpu_cores, job_type])
        
        # Synthetic energy consumption (simplified model)
        base_power = 50  # Base power in watts
        cpu_power = cpu_usage * 2.0  # 2W per % CPU
        memory_power = memory_usage * 0.5  # 0.5W per % memory
        io_power = (io_read + io_write) * 0.3  # 0.3W per MB/s
        core_power = cpu_cores * 10  # 10W per core
        
        avg_power = base_power + cpu_power + memory_power + io_power + core_power
        energy_wh = (avg_power * duration) / 3600  # Convert to Wh
        
        # Add some noise
        energy_wh += np.random.normal(0, energy_wh * 0.1)
        energy_wh = np.maximum(energy_wh, 1)  # Minimum 1 Wh
        
        # Train the model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, energy_wh)
        
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': 'RandomForestRegressor',
            'features': self.feature_names,
            'training_samples': n_samples,
            'version': '1.0.0'
        }
        
        logger.info("Default model created successfully")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from file
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_metadata = model_data.get('metadata', {})
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save current model to file
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metadata': self.model_metadata
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def predict_energy(self, job_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict energy consumption for a job
        """
        try:
            # Prepare features
            feature_vector = self._prepare_features(job_features)
            
            if feature_vector is None:
                return {'error': 'Invalid job features'}
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            energy_prediction = self.model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence interval (for Random Forest)
            if hasattr(self.model, 'estimators_'):
                predictions = [tree.predict(feature_vector_scaled)[0] for tree in self.model.estimators_]
                confidence_interval = {
                    'lower': np.percentile(predictions, 5),
                    'upper': np.percentile(predictions, 95),
                    'std': np.std(predictions)
                }
            else:
                confidence_interval = {'lower': energy_prediction * 0.9, 'upper': energy_prediction * 1.1, 'std': 0}
            
            # Calculate additional metrics
            estimated_cost = energy_prediction * 0.12  # Assume $0.12/kWh
            estimated_co2 = energy_prediction * 0.5  # Assume 0.5 kg CO2/kWh
            
            result = {
                'predicted_energy_wh': round(energy_prediction, 2),
                'predicted_energy_kwh': round(energy_prediction / 1000, 4),
                'confidence_interval': confidence_interval,
                'estimated_cost_usd': round(estimated_cost / 1000, 4),
                'estimated_co2_kg': round(estimated_co2 / 1000, 4),
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('version', 'unknown')
            }
            
            # Cache result if Redis available
            if self.redis_client:
                cache_key = f"prediction:{hash(str(sorted(job_features.items())))}"
                self.redis_client.setex(cache_key, 300, json.dumps(result))  # Cache for 5 minutes
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting energy: {e}")
            return {'error': str(e)}
    
    def _prepare_features(self, job_features: Dict[str, Any]) -> Optional[List[float]]:
        """
        Prepare feature vector from job parameters
        """
        try:
            # Map job type to numeric value
            job_type_mapping = {
                'cpu_intensive': 0,
                'cpu': 0,
                'io_intensive': 1,
                'io': 1,
                'mixed': 2,
                'memory_intensive': 3,
                'memory': 3
            }
            
            # Extract and validate features
            duration = float(job_features.get('duration_seconds', 300))
            cpu_usage = float(job_features.get('cpu_usage_percent', 50))
            memory_usage = float(job_features.get('memory_usage_percent', 30))
            io_read = float(job_features.get('io_read_mbps', 10))
            io_write = float(job_features.get('io_write_mbps', 5))
            cpu_cores = int(job_features.get('cpu_cores', 4))
            job_type = job_type_mapping.get(job_features.get('job_type', 'mixed').lower(), 2)
            
            # Validate ranges
            duration = max(1, min(duration, 86400))  # 1 second to 24 hours
            cpu_usage = max(0, min(cpu_usage, 100))
            memory_usage = max(0, min(memory_usage, 100))
            io_read = max(0, min(io_read, 1000))
            io_write = max(0, min(io_write, 1000))
            cpu_cores = max(1, min(cpu_cores, 64))
            
            return [duration, cpu_usage, memory_usage, io_read, io_write, cpu_cores, job_type]
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def get_historical_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical job data from database
        """
        if not self.db_config:
            return []
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
            SELECT 
                job_id, job_type, duration_seconds, cpu_usage, memory_usage,
                estimated_energy_wh, time
            FROM job_metrics 
            ORDER BY time DESC 
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            
            columns = ['job_id', 'job_type', 'duration_seconds', 'cpu_usage', 
                      'memory_usage', 'estimated_energy_wh', 'timestamp']
            
            result = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []

# Global predictor instance
predictor = None

def initialize_predictor():
    """
    Initialize the global predictor instance
    """
    global predictor
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'hpc_energy'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    # Redis configuration
    redis_config = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'db': int(os.getenv('REDIS_DB', 0))
    }
    
    model_path = os.getenv('MODEL_PATH', 'models/energy_model.pkl')
    
    predictor = EnergyPredictor(model_path, db_config, redis_config)

# API Routes

@app.route('/')
def index():
    """
    API documentation page
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HPC Energy Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
            pre { background: #e9ecef; padding: 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>HPC Energy Prediction API</h1>
        <p>REST API for predicting energy consumption of HPC jobs</p>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Predict energy consumption for a job</p>
            <pre>{
  "duration_seconds": 300,
  "cpu_usage_percent": 80,
  "memory_usage_percent": 60,
  "io_read_mbps": 20,
  "io_write_mbps": 10,
  "cpu_cores": 4,
  "job_type": "mixed"
}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /model/info</h3>
            <p>Get model information and metadata</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /jobs/recent</h3>
            <p>Get recent job data and predictions</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /schedule/recommend</h3>
            <p>Get job scheduling recommendations</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    global predictor
    
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None and predictor.model is not None,
        'version': '1.0.0'
    }
    
    if predictor and predictor.redis_client:
        try:
            predictor.redis_client.ping()
            status['redis_connected'] = True
        except:
            status['redis_connected'] = False
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict_energy():
    """
    Predict energy consumption for a job
    """
    global predictor
    
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        job_features = request.get_json()
        
        if not job_features:
            return jsonify({'error': 'No job features provided'}), 400
        
        # Check cache first
        if predictor.redis_client:
            cache_key = f"prediction:{hash(str(sorted(job_features.items())))}"
            cached_result = predictor.redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result['from_cache'] = True
                return jsonify(result)
        
        # Make prediction
        result = predictor.predict_energy(job_features)
        
        if 'error' in result:
            return jsonify(result), 400
        
        result['from_cache'] = False
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get model information
    """
    global predictor
    
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    info = {
        'model_metadata': predictor.model_metadata,
        'feature_names': predictor.feature_names,
        'model_type': type(predictor.model).__name__ if predictor.model else None,
        'scaler_type': type(predictor.scaler).__name__ if predictor.scaler else None
    }
    
    return jsonify(info)

@app.route('/jobs/recent', methods=['GET'])
def recent_jobs():
    """
    Get recent job data
    """
    global predictor
    
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    limit = request.args.get('limit', 50, type=int)
    limit = min(limit, 1000)  # Maximum 1000 records
    
    historical_data = predictor.get_historical_data(limit)
    
    return jsonify({
        'jobs': historical_data,
        'count': len(historical_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/schedule/recommend', methods=['POST'])
def schedule_recommend():
    """
    Provide job scheduling recommendations
    """
    global predictor
    
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        request_data = request.get_json()
        jobs = request_data.get('jobs', [])
        
        if not jobs:
            return jsonify({'error': 'No jobs provided'}), 400
        
        recommendations = []
        
        for job in jobs:
            # Predict energy for each job
            prediction = predictor.predict_energy(job)
            
            if 'error' not in prediction:
                # Calculate priority score (lower energy = higher priority)
                energy = prediction['predicted_energy_wh']
                duration = job.get('duration_seconds', 300)
                
                # Priority based on energy efficiency (energy per second)
                efficiency_score = energy / duration
                priority_score = 1.0 / (1.0 + efficiency_score / 100)  # Normalize
                
                recommendation = {
                    'job_id': job.get('job_id', f"job_{len(recommendations)}"),
                    'predicted_energy_wh': energy,
                    'priority_score': round(priority_score, 3),
                    'efficiency_score': round(efficiency_score, 3),
                    'recommendation': 'high_priority' if priority_score > 0.7 else 
                                   'medium_priority' if priority_score > 0.4 else 'low_priority'
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority score (descending)
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations,
            'total_jobs': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in schedule recommendation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus-style metrics endpoint
    """
    global predictor
    
    metrics_text = f"""
# HELP hpc_energy_api_model_loaded Whether the energy prediction model is loaded
# TYPE hpc_energy_api_model_loaded gauge
hpc_energy_api_model_loaded {1 if predictor and predictor.model else 0}

# HELP hpc_energy_api_uptime_seconds API uptime in seconds
# TYPE hpc_energy_api_uptime_seconds counter
hpc_energy_api_uptime_seconds {time.time() - app.start_time}

# HELP hpc_energy_api_requests_total Total number of API requests
# TYPE hpc_energy_api_requests_total counter
hpc_energy_api_requests_total {getattr(app, 'request_count', 0)}
"""
    
    return metrics_text, 200, {'Content-Type': 'text/plain'}

# Middleware to count requests
@app.before_request
def before_request():
    if not hasattr(app, 'request_count'):
        app.request_count = 0
    app.request_count += 1

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """
    Main function to run the API server
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HPC Energy Prediction API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model file')
    
    args = parser.parse_args()
    
    # Set model path if provided
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    
    # Initialize predictor
    app.start_time = time.time()
    initialize_predictor()
    
    logger.info(f"Starting HPC Energy Prediction API on {args.host}:{args.port}")
    
    # Run the Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()