#!/bin/bash

# Entrypoint script for HPC Thermal Exporter

set -e

echo "Starting HPC Thermal Exporter..."
echo "Port: ${EXPORTER_PORT:-9200}"
echo "Polling Interval: ${POLLING_INTERVAL:-5}s"

# Check if thermal sensors are available
if [ -d "/sys/class/thermal" ]; then
    echo "Thermal zones found: $(ls /sys/class/thermal/ | wc -l)"
else
    echo "Warning: No thermal zones found at /sys/class/thermal"
fi

# Check if CPU info is available
if [ -f "/proc/cpuinfo" ]; then
    echo "CPU cores detected: $(grep -c processor /proc/cpuinfo)"
else
    echo "Warning: No CPU info found at /proc/cpuinfo"
fi

# Start the thermal exporter
exec python3 thermal_exporter.py