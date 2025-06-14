#!/bin/bash

# Energy Monitoring Script for HPC Jobs
# This script runs continuously during job execution to collect energy metrics

set -e

# Script configuration
JOB_ID="${1:-unknown}"
MONITORING_INTERVAL="${2:-30}"  # seconds
LOG_FILE="/var/log/slurm/energy_monitor.log"
ENERGY_LOG_DIR="/var/log/slurm/energy"
METRICS_ENDPOINT="http://localhost:9300/metrics"
TIMESCALE_HOST="${TIMESCALE_HOST:-timescaledb}"
TIMESCALE_PORT="${TIMESCALE_PORT:-5432}"
TIMESCALE_DB="${TIMESCALE_DB:-hpc_energy}"
TIMESCALE_USER="${TIMESCALE_USER:-postgres}"
TIMESCALE_PASS="${TIMESCALE_PASS:-password}"

# Thermal sensor paths
THERMAL_ZONE_PATH="/sys/class/thermal"
CPU_FREQ_PATH="/sys/devices/system/cpu"
POWER_SUPPLY_PATH="/sys/class/power_supply"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MONITOR[$JOB_ID]: $1" >> "$LOG_FILE"
}

# Function to get current timestamp
get_timestamp() {
    date -u '+%Y-%m-%d %H:%M:%S UTC'
}

# Function to read CPU temperature
get_cpu_temperature() {
    local max_temp=0
    local temp_count=0
    local total_temp=0
    
    # Try different thermal zone sources
    for thermal_zone in "$THERMAL_ZONE_PATH"/thermal_zone*; do
        if [ -d "$thermal_zone" ]; then
            local type_file="$thermal_zone/type"
            local temp_file="$thermal_zone/temp"
            
            if [ -f "$type_file" ] && [ -f "$temp_file" ]; then
                local zone_type=$(cat "$type_file" 2>/dev/null || echo "unknown")
                
                # Focus on CPU-related thermal zones
                if [[ "$zone_type" =~ (cpu|core|package|x86_pkg_temp) ]]; then
                    local temp_millicelsius=$(cat "$temp_file" 2>/dev/null || echo "0")
                    local temp_celsius=$((temp_millicelsius / 1000))
                    
                    if [ "$temp_celsius" -gt 0 ] && [ "$temp_celsius" -lt 150 ]; then
                        total_temp=$((total_temp + temp_celsius))
                        temp_count=$((temp_count + 1))
                        
                        if [ "$temp_celsius" -gt "$max_temp" ]; then
                            max_temp="$temp_celsius"
                        fi
                    fi
                fi
            fi
        fi
    done
    
    # Calculate average temperature
    local avg_temp=0
    if [ "$temp_count" -gt 0 ]; then
        avg_temp=$((total_temp / temp_count))
    else
        # Fallback: try to read from common locations
        if [ -f "/sys/class/hwmon/hwmon0/temp1_input" ]; then
            local temp_millicelsius=$(cat "/sys/class/hwmon/hwmon0/temp1_input" 2>/dev/null || echo "0")
            avg_temp=$((temp_millicelsius / 1000))
            max_temp="$avg_temp"
        elif [ -f "/proc/acpi/thermal_zone/THRM/temperature" ]; then
            avg_temp=$(grep -o '[0-9]*' "/proc/acpi/thermal_zone/THRM/temperature" 2>/dev/null | head -1 || echo "0")
            max_temp="$avg_temp"
        else
            # Simulate temperature based on CPU usage if no sensors available
            local cpu_usage=$(get_cpu_usage)
            avg_temp=$((35 + cpu_usage / 3))  # Base temp 35°C + usage factor
            max_temp="$avg_temp"
        fi
    fi
    
    echo "$avg_temp,$max_temp"
}

# Function to get GPU temperature (if available)
get_gpu_temperature() {
    local gpu_temp=0
    
    # Try nvidia-smi first
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    elif [ -f "/sys/class/drm/card0/device/hwmon/hwmon1/temp1_input" ]; then
        # Try AMD GPU temperature
        local temp_millicelsius=$(cat "/sys/class/drm/card0/device/hwmon/hwmon1/temp1_input" 2>/dev/null || echo "0")
        gpu_temp=$((temp_millicelsius / 1000))
    fi
    
    echo "$gpu_temp"
}

# Function to get CPU usage
get_cpu_usage() {
    # Read CPU stats from /proc/stat
    local cpu_line=$(grep '^cpu ' /proc/stat)
    local cpu_times=($cpu_line)
    
    # Calculate total and idle time
    local idle=${cpu_times[4]}
    local total=0
    for time in "${cpu_times[@]:1:7}"; do
        total=$((total + time))
    done
    
    # Store current values for next calculation
    local current_total="$total"
    local current_idle="$idle"
    
    # Read previous values if available
    local prev_total_file="/tmp/cpu_total_${JOB_ID}"
    local prev_idle_file="/tmp/cpu_idle_${JOB_ID}"
    
    if [ -f "$prev_total_file" ] && [ -f "$prev_idle_file" ]; then
        local prev_total=$(cat "$prev_total_file")
        local prev_idle=$(cat "$prev_idle_file")
        
        # Calculate differences
        local total_diff=$((current_total - prev_total))
        local idle_diff=$((current_idle - prev_idle))
        
        # Calculate CPU usage percentage
        local cpu_usage=0
        if [ "$total_diff" -gt 0 ]; then
            cpu_usage=$(echo "scale=2; (($total_diff - $idle_diff) * 100) / $total_diff" | bc -l)
        fi
        
        echo "$cpu_usage"
    else
        echo "0"
    fi
    
    # Store current values for next iteration
    echo "$current_total" > "$prev_total_file"
    echo "$current_idle" > "$prev_idle_file"
}

# Function to get memory usage
get_memory_usage() {
    local mem_info=$(cat /proc/meminfo)
    local mem_total=$(echo "$mem_info" | grep '^MemTotal:' | awk '{print $2}')
    local mem_available=$(echo "$mem_info" | grep '^MemAvailable:' | awk '{print $2}')
    
    if [ -z "$mem_available" ]; then
        # Fallback calculation for older systems
        local mem_free=$(echo "$mem_info" | grep '^MemFree:' | awk '{print $2}')
        local buffers=$(echo "$mem_info" | grep '^Buffers:' | awk '{print $2}')
        local cached=$(echo "$mem_info" | grep '^Cached:' | awk '{print $2}')
        mem_available=$((mem_free + buffers + cached))
    fi
    
    local mem_used=$((mem_total - mem_available))
    local mem_usage_percent=0
    
    if [ "$mem_total" -gt 0 ]; then
        mem_usage_percent=$(echo "scale=2; ($mem_used * 100) / $mem_total" | bc -l)
    fi
    
    local mem_used_mb=$((mem_used / 1024))
    local mem_total_mb=$((mem_total / 1024))
    
    echo "$mem_usage_percent,$mem_used_mb,$mem_total_mb"
}

# Function to get disk I/O statistics
get_disk_io() {
    local io_stats_file="/proc/diskstats"
    local total_read_sectors=0
    local total_write_sectors=0
    
    # Sum up I/O for all block devices
    while read -r line; do
        local fields=($line)
        local device_name="${fields[2]}"
        
        # Skip loop devices and ram disks
        if [[ "$device_name" =~ ^(loop|ram) ]]; then
            continue
        fi
        
        # Only consider main block devices (not partitions)
        if [[ "$device_name" =~ ^(sd[a-z]|nvme[0-9]+n[0-9]+|vd[a-z])$ ]]; then
            local read_sectors="${fields[5]}"
            local write_sectors="${fields[9]}"
            
            total_read_sectors=$((total_read_sectors + read_sectors))
            total_write_sectors=$((total_write_sectors + write_sectors))
        fi
    done < "$io_stats_file"
    
    # Store current values for rate calculation
    local current_read="$total_read_sectors"
    local current_write="$total_write_sectors"
    local current_time=$(date +%s)
    
    # Read previous values if available
    local prev_read_file="/tmp/io_read_${JOB_ID}"
    local prev_write_file="/tmp/io_write_${JOB_ID}"
    local prev_time_file="/tmp/io_time_${JOB_ID}"
    
    local read_rate_mbps=0
    local write_rate_mbps=0
    
    if [ -f "$prev_read_file" ] && [ -f "$prev_write_file" ] && [ -f "$prev_time_file" ]; then
        local prev_read=$(cat "$prev_read_file")
        local prev_write=$(cat "$prev_write_file")
        local prev_time=$(cat "$prev_time_file")
        
        local time_diff=$((current_time - prev_time))
        
        if [ "$time_diff" -gt 0 ]; then
            # Calculate rates (sectors are 512 bytes)
            local read_diff=$((current_read - prev_read))
            local write_diff=$((current_write - prev_write))
            
            read_rate_mbps=$(echo "scale=2; ($read_diff * 512) / (1024 * 1024 * $time_diff)" | bc -l)
            write_rate_mbps=$(echo "scale=2; ($write_diff * 512) / (1024 * 1024 * $time_diff)" | bc -l)
        fi
    fi
    
    # Store current values for next iteration
    echo "$current_read" > "$prev_read_file"
    echo "$current_write" > "$prev_write_file"
    echo "$current_time" > "$prev_time_file"
    
    echo "$read_rate_mbps,$write_rate_mbps"
}

# Function to get network I/O statistics
get_network_io() {
    local net_stats_file="/proc/net/dev"
    local total_rx_bytes=0
    local total_tx_bytes=0
    
    # Sum up network I/O for all interfaces (excluding loopback)
    while read -r line; do
        # Skip header lines
        [[ "$line" =~ ^[[:space:]]*(Inter-|face) ]] && continue
        
        local interface=$(echo "$line" | cut -d':' -f1 | tr -d ' ')
        
        # Skip loopback interface
        [[ "$interface" == "lo" ]] && continue
        
        local stats=$(echo "$line" | cut -d':' -f2)
        local fields=($stats)
        
        local rx_bytes="${fields[0]}"
        local tx_bytes="${fields[8]}"
        
        total_rx_bytes=$((total_rx_bytes + rx_bytes))
        total_tx_bytes=$((total_tx_bytes + tx_bytes))
    done < "$net_stats_file"
    
    # Store current values for rate calculation
    local current_rx="$total_rx_bytes"
    local current_tx="$total_tx_bytes"
    local current_time=$(date +%s)
    
    # Read previous values if available
    local prev_rx_file="/tmp/net_rx_${JOB_ID}"
    local prev_tx_file="/tmp/net_tx_${JOB_ID}"
    local prev_net_time_file="/tmp/net_time_${JOB_ID}"
    
    local rx_rate_mbps=0
    local tx_rate_mbps=0
    
    if [ -f "$prev_rx_file" ] && [ -f "$prev_tx_file" ] && [ -f "$prev_net_time_file" ]; then
        local prev_rx=$(cat "$prev_rx_file")
        local prev_tx=$(cat "$prev_tx_file")
        local prev_time=$(cat "$prev_net_time_file")
        
        local time_diff=$((current_time - prev_time))
        
        if [ "$time_diff" -gt 0 ]; then
            local rx_diff=$((current_rx - prev_rx))
            local tx_diff=$((current_tx - prev_tx))
            
            rx_rate_mbps=$(echo "scale=2; ($rx_diff) / (1024 * 1024 * $time_diff)" | bc -l)
            tx_rate_mbps=$(echo "scale=2; ($tx_diff) / (1024 * 1024 * $time_diff)" | bc -l)
        fi
    fi
    
    # Store current values for next iteration
    echo "$current_rx" > "$prev_rx_file"
    echo "$current_tx" > "$prev_tx_file"
    echo "$current_time" > "$prev_net_time_file"
    
    echo "$rx_rate_mbps,$tx_rate_mbps"
}

# Function to get system load
get_system_load() {
    local load_avg=$(cat /proc/loadavg | cut -d' ' -f1-3)
    echo "$load_avg" | tr ' ' ','
}

# Function to get CPU frequency
get_cpu_frequency() {
    local total_freq=0
    local cpu_count=0
    
    # Try to read current CPU frequencies
    for cpu_dir in "$CPU_FREQ_PATH"/cpu[0-9]*; do
        if [ -d "$cpu_dir" ]; then
            local freq_file="$cpu_dir/cpufreq/scaling_cur_freq"
            if [ -f "$freq_file" ]; then
                local freq_khz=$(cat "$freq_file" 2>/dev/null || echo "0")
                if [ "$freq_khz" -gt 0 ]; then
                    total_freq=$((total_freq + freq_khz))
                    cpu_count=$((cpu_count + 1))
                fi
            fi
        fi
    done
    
    local avg_freq_mhz=0
    if [ "$cpu_count" -gt 0 ]; then
        avg_freq_mhz=$((total_freq / cpu_count / 1000))
    else
        # Fallback: try to get base frequency from /proc/cpuinfo
        avg_freq_mhz=$(grep '^cpu MHz' /proc/cpuinfo | head -1 | awk '{print int($4)}' || echo "0")
    fi
    
    echo "$avg_freq_mhz"
}

# Function to estimate power consumption
estimate_power_consumption() {
    local cpu_usage="$1"
    local cpu_temp="$2"
    local cpu_freq_mhz="$3"
    local memory_usage_percent="$4"
    local gpu_temp="$5"
    
    # Base power consumption (idle system)
    local base_power=50  # watts
    
    # CPU power estimation
    local cpu_power=0
    if [ "$cpu_usage" != "0" ] && [ "$cpu_freq_mhz" != "0" ]; then
        # Simplified power model: P = base + (usage * freq_factor * temp_factor)
        local freq_factor=$(echo "scale=4; $cpu_freq_mhz / 2000" | bc -l)  # Normalize to ~2GHz
        local temp_factor=$(echo "scale=4; 1 + ($cpu_temp - 40) / 100" | bc -l)  # Temperature impact
        
        cpu_power=$(echo "scale=2; $cpu_usage * 0.8 * $freq_factor * $temp_factor" | bc -l)
    fi
    
    # Memory power estimation (rough approximation)
    local memory_power=$(echo "scale=2; $memory_usage_percent * 0.1" | bc -l)
    
    # GPU power estimation (if GPU is present)
    local gpu_power=0
    if [ "$gpu_temp" -gt 30 ]; then
        # Estimate GPU power based on temperature
        gpu_power=$(echo "scale=2; ($gpu_temp - 30) * 2" | bc -l)
    fi
    
    # Total estimated power
    local total_power=$(echo "scale=2; $base_power + $cpu_power + $memory_power + $gpu_power" | bc -l)
    
    echo "$total_power"
}

# Function to collect and log metrics
collect_metrics() {
    local timestamp=$(get_timestamp)
    
    # Get system metrics
    local cpu_usage=$(get_cpu_usage)
    local memory_data=$(get_memory_usage)
    IFS=',' read -r memory_usage_percent memory_used_mb memory_total_mb <<< "$memory_data"
    
    local temp_data=$(get_cpu_temperature)
    IFS=',' read -r cpu_temp_avg cpu_temp_max <<< "$temp_data"
    
    local gpu_temp=$(get_gpu_temperature)
    local cpu_freq=$(get_cpu_frequency)
    
    local disk_io_data=$(get_disk_io)
    IFS=',' read -r disk_read_mbps disk_write_mbps <<< "$disk_io_data"
    
    local network_io_data=$(get_network_io)
    IFS=',' read -r net_rx_mbps net_tx_mbps <<< "$network_io_data"
    
    local load_avg=$(get_system_load)
    
    # Estimate power consumption
    local estimated_power=$(estimate_power_consumption "$cpu_usage" "$cpu_temp_avg" "$cpu_freq" "$memory_usage_percent" "$gpu_temp")
    
    # Log to energy log file
    local energy_log_file="$ENERGY_LOG_DIR/job_${JOB_ID}_energy.log"
    echo "$timestamp,$cpu_usage,$memory_usage_percent,$cpu_temp_avg,$estimated_power,$cpu_temp_max,$gpu_temp,$cpu_freq,$memory_used_mb,$disk_read_mbps,$disk_write_mbps,$net_rx_mbps,$net_tx_mbps,$load_avg" >> "$energy_log_file"
    
    # Insert into TimescaleDB (if available)
    if command -v psql >/dev/null 2>&1; then
        local node_name="$(hostname)"
        local sql="INSERT INTO hpc_energy.node_metrics (
            time, node_id, cpu_usage_percent, memory_usage_percent,
            cpu_temp_celsius, gpu_temp_celsius, cpu_freq_mhz,
            estimated_power_watts, disk_read_mbps, disk_write_mbps,
            network_rx_mbps, network_tx_mbps, load_1min, load_5min, load_15min
        ) VALUES (
            '$timestamp', '$node_name', $cpu_usage, $memory_usage_percent,
            $cpu_temp_avg, $gpu_temp, $cpu_freq, $estimated_power,
            $disk_read_mbps, $disk_write_mbps, $net_rx_mbps, $net_tx_mbps,
            $(echo "$load_avg" | cut -d',' -f1),
            $(echo "$load_avg" | cut -d',' -f2),
            $(echo "$load_avg" | cut -d',' -f3)
        );"
        
        echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || {
            log_message "Warning: Failed to insert node metrics into database"
        }
    fi
    
    # Log summary
    log_message "Metrics collected: CPU=${cpu_usage}%, Mem=${memory_usage_percent}%, Temp=${cpu_temp_avg}°C, Power=${estimated_power}W"
}

# Function to check for thermal events
check_thermal_events() {
    local cpu_temp="$1"
    local gpu_temp="$2"
    local timestamp="$3"
    
    # Define thermal thresholds
    local cpu_warning_temp=75
    local cpu_critical_temp=85
    local gpu_warning_temp=80
    local gpu_critical_temp=90
    
    # Check CPU temperature
    if [ "$cpu_temp" -gt "$cpu_critical_temp" ]; then
        log_message "CRITICAL: CPU temperature $cpu_temp°C exceeds critical threshold ($cpu_critical_temp°C)"
        
        # Insert thermal event into database
        if command -v psql >/dev/null 2>&1; then
            local node_name="$(hostname)"
            local sql="INSERT INTO hpc_energy.thermal_events (
                time, node_id, job_id, event_type, component,
                temperature_celsius, threshold_celsius, severity
            ) VALUES (
                '$timestamp', '$node_name', '$JOB_ID', 'temperature_critical',
                'cpu', $cpu_temp, $cpu_critical_temp, 'critical'
            );"
            
            echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || true
        fi
    elif [ "$cpu_temp" -gt "$cpu_warning_temp" ]; then
        log_message "WARNING: CPU temperature $cpu_temp°C exceeds warning threshold ($cpu_warning_temp°C)"
    fi
    
    # Check GPU temperature
    if [ "$gpu_temp" -gt "$gpu_critical_temp" ]; then
        log_message "CRITICAL: GPU temperature $gpu_temp°C exceeds critical threshold ($gpu_critical_temp°C)"
        
        if command -v psql >/dev/null 2>&1; then
            local node_name="$(hostname)"
            local sql="INSERT INTO hpc_energy.thermal_events (
                time, node_id, job_id, event_type, component,
                temperature_celsius, threshold_celsius, severity
            ) VALUES (
                '$timestamp', '$node_name', '$JOB_ID', 'temperature_critical',
                'gpu', $gpu_temp, $gpu_critical_temp, 'critical'
            );"
            
            echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || true
        fi
    elif [ "$gpu_temp" -gt "$gpu_warning_temp" ]; then
        log_message "WARNING: GPU temperature $gpu_temp°C exceeds warning threshold ($gpu_warning_temp°C)"
    fi
}

# Function to cleanup on exit
cleanup() {
    log_message "Energy monitoring stopped for job $JOB_ID"
    
    # Clean up temporary files
    rm -f "/tmp/cpu_total_${JOB_ID}"
    rm -f "/tmp/cpu_idle_${JOB_ID}"
    rm -f "/tmp/io_read_${JOB_ID}"
    rm -f "/tmp/io_write_${JOB_ID}"
    rm -f "/tmp/io_time_${JOB_ID}"
    rm -f "/tmp/net_rx_${JOB_ID}"
    rm -f "/tmp/net_tx_${JOB_ID}"
    rm -f "/tmp/net_time_${JOB_ID}"
    
    exit 0
}

# Main monitoring loop
main() {
    # Validate input parameters
    if [ "$JOB_ID" = "unknown" ]; then
        echo "Error: Job ID not provided" >&2
        exit 1
    fi
    
    # Create energy log directory if it doesn't exist
    mkdir -p "$ENERGY_LOG_DIR"
    
    # Initialize energy log file with header
    local energy_log_file="$ENERGY_LOG_DIR/job_${JOB_ID}_energy.log"
    echo "# Energy monitoring log for job $JOB_ID" > "$energy_log_file"
    echo "# timestamp,cpu_usage,memory_usage,cpu_temp,power_estimate,cpu_temp_max,gpu_temp,cpu_freq,memory_used_mb,disk_read_mbps,disk_write_mbps,net_rx_mbps,net_tx_mbps,load_1min,load_5min,load_15min" >> "$energy_log_file"
    
    log_message "Starting energy monitoring for job $JOB_ID (interval: ${MONITORING_INTERVAL}s)"
    
    # Set up signal handlers
    trap cleanup SIGTERM SIGINT
    
    # Main monitoring loop
    while [ -f "/tmp/job_${JOB_ID}_running" ]; do
        local timestamp=$(get_timestamp)
        
        # Collect metrics
        collect_metrics
        
        # Get temperature data for thermal event checking
        local temp_data=$(get_cpu_temperature)
        IFS=',' read -r cpu_temp_avg cpu_temp_max <<< "$temp_data"
        local gpu_temp=$(get_gpu_temperature)
        
        # Check for thermal events
        check_thermal_events "$cpu_temp_avg" "$gpu_temp" "$timestamp"
        
        # Sleep for monitoring interval
        sleep "$MONITORING_INTERVAL"
    done
    
    log_message "Job $JOB_ID completed, stopping energy monitoring"
    cleanup
}

# Error handling
trap 'log_message "Error occurred in energy monitoring script for job $JOB_ID"' ERR

# Execute main function
main "$@"