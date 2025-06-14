#!/bin/bash

# Slurm Job Epilog Script for Energy Monitoring
# This script runs after each job completes to finalize energy monitoring

set -e

# Script configuration
LOG_FILE="/var/log/slurm/job_epilog.log"
ENERGY_LOG_DIR="/var/log/slurm/energy"
METRICS_ENDPOINT="http://localhost:9300/metrics"
TIMESCALE_HOST="${TIMESCALE_HOST:-timescaledb}"
TIMESCALE_PORT="${TIMESCALE_PORT:-5432}"
TIMESCALE_DB="${TIMESCALE_DB:-hpc_energy}"
TIMESCALE_USER="${TIMESCALE_USER:-postgres}"
TIMESCALE_PASS="${TIMESCALE_PASS:-password}"
API_ENDPOINT="${API_ENDPOINT:-http://energy-prediction-api:5000}"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] EPILOG: $1" >> "$LOG_FILE"
}

# Function to get current timestamp
get_timestamp() {
    date -u '+%Y-%m-%d %H:%M:%S UTC'
}

# Function to stop background monitoring
stop_background_monitoring() {
    local job_id="$1"
    
    # Remove job running flag to stop background monitoring
    rm -f "/tmp/job_${job_id}_running"
    
    # Clean up monitoring script
    rm -f "/tmp/energy_monitor_${job_id}.sh"
    
    # Kill any remaining monitoring processes
    pkill -f "energy_monitor_${job_id}" 2>/dev/null || true
    
    log_message "Stopped background monitoring for job $job_id"
}

# Function to calculate final energy consumption
calculate_energy_consumption() {
    local job_id="$1"
    local duration_seconds="$2"
    
    local energy_log_file="$ENERGY_LOG_DIR/job_${job_id}_energy.log"
    
    if [ ! -f "$energy_log_file" ]; then
        log_message "Warning: Energy log file not found for job $job_id"
        echo "0,0,0,0"  # avg_power, peak_power, total_energy, avg_temp
        return
    fi
    
    # Parse energy log file (skip header)
    local total_power=0
    local peak_power=0
    local sample_count=0
    local total_temp=0
    local peak_temp=0
    
    while IFS=',' read -r timestamp cpu_usage mem_usage cpu_temp power_estimate; do
        # Skip comments and empty lines
        [[ "$timestamp" =~ ^#.*$ ]] && continue
        [[ -z "$timestamp" ]] && continue
        
        # Validate numeric values
        if [[ "$power_estimate" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            total_power=$(echo "$total_power + $power_estimate" | bc -l)
            if (( $(echo "$power_estimate > $peak_power" | bc -l) )); then
                peak_power="$power_estimate"
            fi
        fi
        
        if [[ "$cpu_temp" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            total_temp=$(echo "$total_temp + $cpu_temp" | bc -l)
            if (( $(echo "$cpu_temp > $peak_temp" | bc -l) )); then
                peak_temp="$cpu_temp"
            fi
        fi
        
        ((sample_count++))
    done < "$energy_log_file"
    
    if [ "$sample_count" -eq 0 ]; then
        log_message "Warning: No valid samples found in energy log for job $job_id"
        echo "0,0,0,0"
        return
    fi
    
    # Calculate averages
    local avg_power=$(echo "scale=2; $total_power / $sample_count" | bc -l)
    local avg_temp=$(echo "scale=2; $total_temp / $sample_count" | bc -l)
    
    # Calculate total energy consumption (power * time)
    # Assuming monitoring interval of 30 seconds
    local monitoring_interval=30
    local total_energy_wh=$(echo "scale=2; $avg_power * $duration_seconds / 3600" | bc -l)
    
    echo "$avg_power,$peak_power,$total_energy_wh,$avg_temp,$peak_temp"
    
    log_message "Calculated energy for job $job_id: avg_power=${avg_power}W, total_energy=${total_energy_wh}Wh, avg_temp=${avg_temp}°C"
}

# Function to get job classification
classify_job_type() {
    local job_id="$1"
    local cpu_usage="$2"
    local mem_usage="$3"
    local io_usage="$4"
    
    # Simple classification based on resource usage patterns
    if (( $(echo "$cpu_usage > 70" | bc -l) )) && (( $(echo "$mem_usage < 50" | bc -l) )); then
        echo "cpu_intensive"
    elif (( $(echo "$mem_usage > 70" | bc -l) )) && (( $(echo "$cpu_usage < 50" | bc -l) )); then
        echo "memory_intensive"
    elif (( $(echo "$io_usage > 50" | bc -l) )); then
        echo "io_intensive"
    elif (( $(echo "$cpu_usage > 40" | bc -l) )) && (( $(echo "$mem_usage > 40" | bc -l) )); then
        echo "mixed"
    else
        echo "light"
    fi
}

# Function to get cgroup statistics
get_cgroup_stats() {
    local job_id="$1"
    local user_id="$2"
    
    local cgroup_path="/sys/fs/cgroup/slurm/uid_${user_id}/job_${job_id}"
    
    # Default values
    local max_memory_mb=0
    local total_cpu_time=0
    local io_read_bytes=0
    local io_write_bytes=0
    
    if [ -d "$cgroup_path" ]; then
        # Memory statistics
        if [ -f "$cgroup_path/memory.max_usage_in_bytes" ]; then
            local max_memory_bytes=$(cat "$cgroup_path/memory.max_usage_in_bytes" 2>/dev/null || echo "0")
            max_memory_mb=$(echo "scale=0; $max_memory_bytes / 1024 / 1024" | bc -l)
        fi
        
        # CPU statistics
        if [ -f "$cgroup_path/cpuacct.usage" ]; then
            local cpu_usage_ns=$(cat "$cgroup_path/cpuacct.usage" 2>/dev/null || echo "0")
            total_cpu_time=$(echo "scale=2; $cpu_usage_ns / 1000000000" | bc -l)  # Convert to seconds
        fi
        
        # I/O statistics (if available)
        if [ -f "$cgroup_path/blkio.throttle.io_service_bytes" ]; then
            io_read_bytes=$(grep "Read" "$cgroup_path/blkio.throttle.io_service_bytes" 2>/dev/null | awk '{sum+=$3} END {print sum+0}')
            io_write_bytes=$(grep "Write" "$cgroup_path/blkio.throttle.io_service_bytes" 2>/dev/null | awk '{sum+=$3} END {print sum+0}')
        fi
    fi
    
    echo "$max_memory_mb,$total_cpu_time,$io_read_bytes,$io_write_bytes"
}

# Function to update job metrics in database
update_job_metrics() {
    local job_id="$1"
    local user_id="$2"
    local node_name="$3"
    local partition="$4"
    local exit_code="$5"
    local duration_seconds="$6"
    local cpu_cores="$7"
    local memory_mb="$8"
    
    # Calculate energy consumption
    local energy_data=$(calculate_energy_consumption "$job_id" "$duration_seconds")
    IFS=',' read -r avg_power peak_power total_energy_wh avg_temp peak_temp <<< "$energy_data"
    
    # Get cgroup statistics
    local cgroup_data=$(get_cgroup_stats "$job_id" "$user_id")
    IFS=',' read -r max_memory_mb total_cpu_time io_read_bytes io_write_bytes <<< "$cgroup_data"
    
    # Calculate average resource usage
    local avg_cpu_usage=0
    local avg_memory_usage=0
    local avg_io_usage=0
    
    if [ "$duration_seconds" -gt 0 ]; then
        avg_cpu_usage=$(echo "scale=2; $total_cpu_time * 100 / ($duration_seconds * $cpu_cores)" | bc -l)
        avg_memory_usage=$(echo "scale=2; $max_memory_mb * 100 / $memory_mb" | bc -l)
        
        # Calculate I/O usage in MB/s
        local total_io_mb=$(echo "scale=2; ($io_read_bytes + $io_write_bytes) / 1024 / 1024" | bc -l)
        avg_io_usage=$(echo "scale=2; $total_io_mb / $duration_seconds" | bc -l)
    fi
    
    # Classify job type
    local job_type=$(classify_job_type "$job_id" "$avg_cpu_usage" "$avg_memory_usage" "$avg_io_usage")
    
    # Determine job state
    local job_state="COMPLETED"
    if [ "$exit_code" -ne 0 ]; then
        job_state="FAILED"
    fi
    
    # Calculate I/O rates
    local io_read_mbps=0
    local io_write_mbps=0
    if [ "$duration_seconds" -gt 0 ]; then
        io_read_mbps=$(echo "scale=2; $io_read_bytes / 1024 / 1024 / $duration_seconds" | bc -l)
        io_write_mbps=$(echo "scale=2; $io_write_bytes / 1024 / 1024 / $duration_seconds" | bc -l)
    fi
    
    # Prepare SQL update statement
    local timestamp=$(get_timestamp)
    local sql="UPDATE hpc_energy.job_metrics SET
        job_type = '$job_type',
        duration_seconds = $duration_seconds,
        cpu_usage = $avg_cpu_usage,
        memory_usage = $avg_memory_usage,
        io_read_mbps = $io_read_mbps,
        io_write_mbps = $io_write_mbps,
        avg_cpu_temp = $avg_temp,
        peak_cpu_temp = $peak_temp,
        avg_power_watts = $avg_power,
        peak_power_watts = $peak_power,
        estimated_energy_wh = $total_energy_wh,
        job_state = '$job_state',
        exit_code = $exit_code
    WHERE job_id = '$job_id' AND node_id = '$node_name';"
    
    # Execute SQL (with error handling)
    if command -v psql >/dev/null 2>&1; then
        echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || {
            log_message "Warning: Failed to update job metrics in database for job $job_id"
        }
    else
        log_message "Warning: psql not available, skipping database update for job $job_id"
    fi
    
    log_message "Updated job metrics for job $job_id: type=$job_type, duration=${duration_seconds}s, energy=${total_energy_wh}Wh, state=$job_state"
}

# Function to generate energy prediction
generate_energy_prediction() {
    local job_id="$1"
    local job_type="$2"
    local duration_seconds="$3"
    local cpu_usage="$4"
    local memory_usage="$5"
    local io_read_mbps="$6"
    local io_write_mbps="$7"
    local cpu_cores="$8"
    local actual_energy_wh="$9"
    
    # Prepare prediction request
    local prediction_request="{
        \"duration_seconds\": $duration_seconds,
        \"cpu_usage_percent\": $cpu_usage,
        \"memory_usage_percent\": $memory_usage,
        \"io_read_mbps\": $io_read_mbps,
        \"io_write_mbps\": $io_write_mbps,
        \"cpu_cores\": $cpu_cores,
        \"job_type\": \"$job_type\"
    }"
    
    # Call prediction API
    if command -v curl >/dev/null 2>&1; then
        local prediction_response=$(curl -s -X POST "$API_ENDPOINT/predict" \
            -H "Content-Type: application/json" \
            -d "$prediction_request" 2>/dev/null || echo "{}")
        
        # Parse prediction response
        local predicted_energy=$(echo "$prediction_response" | grep -o '"predicted_energy_wh":[0-9.]*' | cut -d':' -f2 || echo "0")
        local confidence_score=$(echo "$prediction_response" | grep -o '"confidence_interval":{[^}]*}' | grep -o '"std":[0-9.]*' | cut -d':' -f2 || echo "0")
        
        # Calculate prediction error
        local prediction_error=0
        if [ "$actual_energy_wh" != "0" ] && [ "$predicted_energy" != "0" ]; then
            prediction_error=$(echo "scale=2; (($predicted_energy - $actual_energy_wh) * 100) / $actual_energy_wh" | bc -l)
        fi
        
        # Insert prediction record
        local timestamp=$(get_timestamp)
        local sql="INSERT INTO hpc_energy.energy_predictions (
            time, job_id, predicted_duration_seconds, predicted_cpu_usage,
            predicted_memory_usage, predicted_energy_wh, model_version,
            confidence_score, prediction_method, actual_energy_wh,
            actual_duration_seconds, prediction_error_percent
        ) VALUES (
            '$timestamp', '$job_id', $duration_seconds, $cpu_usage,
            $memory_usage, $predicted_energy, '1.0.0',
            $confidence_score, 'post_job_analysis', $actual_energy_wh,
            $duration_seconds, $prediction_error
        );"
        
        # Execute SQL
        if command -v psql >/dev/null 2>&1; then
            echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || {
                log_message "Warning: Failed to insert prediction record for job $job_id"
            }
        fi
        
        log_message "Generated energy prediction for job $job_id: predicted=${predicted_energy}Wh, actual=${actual_energy_wh}Wh, error=${prediction_error}%"
    else
        log_message "Warning: curl not available, skipping energy prediction for job $job_id"
    fi
}

# Function to cleanup job files
cleanup_job_files() {
    local job_id="$1"
    
    # Archive energy log file
    local energy_log_file="$ENERGY_LOG_DIR/job_${job_id}_energy.log"
    if [ -f "$energy_log_file" ]; then
        # Compress and move to archive directory
        local archive_dir="$ENERGY_LOG_DIR/archive"
        mkdir -p "$archive_dir"
        
        gzip "$energy_log_file" 2>/dev/null || true
        mv "${energy_log_file}.gz" "$archive_dir/" 2>/dev/null || true
        
        log_message "Archived energy log for job $job_id"
    fi
    
    # Clean up temporary files
    rm -f "/tmp/job_${job_id}_running"
    rm -f "/tmp/energy_monitor_${job_id}.sh"
    
    log_message "Cleaned up temporary files for job $job_id"
}

# Function to notify external systems
notify_external_systems() {
    local job_id="$1"
    local node_name="$2"
    local job_state="$3"
    local energy_wh="$4"
    
    # Notify Prometheus job exporter
    if command -v curl >/dev/null 2>&1; then
        curl -s -X POST "$METRICS_ENDPOINT/job/complete" \
            -H "Content-Type: application/json" \
            -d "{\"job_id\": \"$job_id\", \"node_id\": \"$node_name\", \"state\": \"$job_state\", \"energy_wh\": $energy_wh}" \
            >/dev/null 2>&1 || {
            log_message "Warning: Failed to notify metrics endpoint for job $job_id completion"
        }
    fi
    
    log_message "Notified external systems about job $job_id completion"
}

# Main execution
main() {
    # Get Slurm environment variables
    local job_id="${SLURM_JOB_ID:-unknown}"
    local user_id="${SLURM_JOB_USER:-unknown}"
    local node_name="${SLURMD_NODENAME:-$(hostname)}"
    local partition="${SLURM_JOB_PARTITION:-unknown}"
    local exit_code="${SLURM_JOB_EXIT_CODE:-0}"
    local cpu_cores="${SLURM_CPUS_ON_NODE:-1}"
    local memory_mb="${SLURM_MEM_PER_NODE:-1024}"
    
    # Calculate job duration
    local start_time="${SLURM_JOB_START_TIME:-0}"
    local end_time="$(date +%s)"
    local duration_seconds=$((end_time - start_time))
    
    log_message "Starting epilog for job $job_id (user: $user_id, node: $node_name, duration: ${duration_seconds}s, exit: $exit_code)"
    
    # Stop background monitoring
    stop_background_monitoring "$job_id"
    
    # Update job metrics in database
    update_job_metrics "$job_id" "$user_id" "$node_name" "$partition" "$exit_code" "$duration_seconds" "$cpu_cores" "$memory_mb"
    
    # Get final energy consumption for prediction
    local energy_data=$(calculate_energy_consumption "$job_id" "$duration_seconds")
    IFS=',' read -r avg_power peak_power total_energy_wh avg_temp peak_temp <<< "$energy_data"
    
    # Generate energy prediction for model training
    if [ "$total_energy_wh" != "0" ]; then
        # Get job classification and usage data from database or calculate
        local cgroup_data=$(get_cgroup_stats "$job_id" "$user_id")
        IFS=',' read -r max_memory_mb total_cpu_time io_read_bytes io_write_bytes <<< "$cgroup_data"
        
        local avg_cpu_usage=0
        local avg_memory_usage=0
        local io_read_mbps=0
        local io_write_mbps=0
        
        if [ "$duration_seconds" -gt 0 ]; then
            avg_cpu_usage=$(echo "scale=2; $total_cpu_time * 100 / ($duration_seconds * $cpu_cores)" | bc -l)
            avg_memory_usage=$(echo "scale=2; $max_memory_mb * 100 / $memory_mb" | bc -l)
            io_read_mbps=$(echo "scale=2; $io_read_bytes / 1024 / 1024 / $duration_seconds" | bc -l)
            io_write_mbps=$(echo "scale=2; $io_write_bytes / 1024 / 1024 / $duration_seconds" | bc -l)
        fi
        
        local job_type=$(classify_job_type "$job_id" "$avg_cpu_usage" "$avg_memory_usage" "$(echo "$io_read_mbps + $io_write_mbps" | bc -l)")
        
        generate_energy_prediction "$job_id" "$job_type" "$duration_seconds" "$avg_cpu_usage" "$avg_memory_usage" "$io_read_mbps" "$io_write_mbps" "$cpu_cores" "$total_energy_wh"
    fi
    
    # Determine final job state
    local job_state="COMPLETED"
    if [ "$exit_code" -ne 0 ]; then
        job_state="FAILED"
    fi
    
    # Notify external systems
    notify_external_systems "$job_id" "$node_name" "$job_state" "$total_energy_wh"
    
    # Cleanup job files
    cleanup_job_files "$job_id"
    
    log_message "Epilog completed successfully for job $job_id (energy: ${total_energy_wh}Wh, state: $job_state)"
}

# Error handling
trap 'log_message "Error occurred in epilog script for job ${SLURM_JOB_ID:-unknown}"' ERR

# Execute main function
main "$@"

exit 0