#!/bin/bash

# HPC Energy Prediction API Entrypoint Script
# Sets up environment and starts the Flask API server

set -e

echo "Starting HPC Energy Prediction API..."

# Set default environment variables
export FLASK_APP=${FLASK_APP:-energy_prediction_api.py}
export FLASK_ENV=${FLASK_ENV:-production}
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-5000}
export API_DEBUG=${API_DEBUG:-false}

# Database configuration
export DB_HOST=${DB_HOST:-timescaledb}
export DB_PORT=${DB_PORT:-5432}
export DB_NAME=${DB_NAME:-hpc_energy}
export DB_USER=${DB_USER:-postgres}
export DB_PASSWORD=${DB_PASSWORD:-password}

# Redis configuration
export REDIS_HOST=${REDIS_HOST:-redis}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}

# Model configuration
export MODEL_PATH=${MODEL_PATH:-models/energy_model.pkl}

# Wait for dependencies
echo "Waiting for database connection..."
until python3 -c "import psycopg2; psycopg2.connect(host='$DB_HOST', port=$DB_PORT, database='$DB_NAME', user='$DB_USER', password='$DB_PASSWORD')" 2>/dev/null; do
    echo "Database not ready, waiting..."
    sleep 2
done
echo "Database connection established"

echo "Waiting for Redis connection..."
until python3 -c "import redis; redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, db=$REDIS_DB).ping()" 2>/dev/null; do
    echo "Redis not ready, waiting..."
    sleep 2
done
echo "Redis connection established"

# Create necessary directories
mkdir -p models data logs

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "API will use default model for demonstration"
else
    echo "Using model file: $MODEL_PATH"
fi

# Set up logging
export PYTHONUNBUFFERED=1

# Start the API server
echo "Starting Flask API server on $API_HOST:$API_PORT"
echo "Debug mode: $API_DEBUG"
echo "Environment: $FLASK_ENV"

if [ "$API_DEBUG" = "true" ]; then
    exec python3 energy_prediction_api.py --host "$API_HOST" --port "$API_PORT" --debug
else
    exec python3 energy_prediction_api.py --host "$API_HOST" --port "$API_PORT"
fi