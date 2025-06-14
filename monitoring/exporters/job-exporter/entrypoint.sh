#!/bin/bash

# Entrypoint script for HPC Job Exporter

set -e

echo "Starting HPC Job Exporter..."
echo "Port: ${EXPORTER_PORT:-9300}"
echo "Polling Interval: ${POLLING_INTERVAL:-30}s"
echo "TimescaleDB Host: ${TIMESCALEDB_HOST:-timescaledb}"

# Wait for TimescaleDB to be ready
echo "Waiting for TimescaleDB to be ready..."
while ! nc -z "${TIMESCALEDB_HOST:-timescaledb}" "${TIMESCALEDB_PORT:-5432}"; do
    echo "TimescaleDB is not ready yet, waiting..."
    sleep 2
done
echo "TimescaleDB is ready!"

# Check if Slurm commands are available
if command -v sacct >/dev/null 2>&1; then
    echo "Slurm sacct command found"
else
    echo "Warning: Slurm sacct command not found. Job metrics may be limited."
fi

if command -v sstat >/dev/null 2>&1; then
    echo "Slurm sstat command found"
else
    echo "Warning: Slurm sstat command not found. Running job stats may be limited."
fi

if command -v sinfo >/dev/null 2>&1; then
    echo "Slurm sinfo command found"
else
    echo "Warning: Slurm sinfo command not found. Cluster info may be limited."
fi

# Start the job exporter
exec python3 job_exporter.py