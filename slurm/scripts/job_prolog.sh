#!/bin/bash

# Slurm Job Prolog Script for Energy Monitoring
# This script runs before each job starts to initialize energy monitoring

set -e

# Script configuration
LOG_FILE="/var/log/slurm/job_prolog.log"
ENERGY_LOG_DIR="/var/log/slurm/energy"
METRICS_ENDPOINT="http://localhost:9300/metrics"
TIMESCALE_HOST="${TIMESCALE_HOST:-timescaledb}"
TIMESCALE_PORT="${TIMESCALE_PORT:-5432}"
TIMESCALE_DB="${TIMESCALE_DB:-hpc_energy}"
TIMESCALE_USER="${TIMESCALE_USER:-postgres}"
TIMESCALE_PASS="${TIMESCALE_PASS:-password}"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PROLOG: $1" >> "$LOG_FILE"
}

# Function to get current timestamp
get_timestamp() {
    date -u '+%Y-%m-%d %H:%M:%S UTC'
}

# Function to get system metrics
get_system_metrics() {
    local node_name="$1"
    
    # CPU information
    local cpu_cores=$(nproc)
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    
    # Memory information
    local mem_total=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    local mem_used=$(free -m | awk 'NR==2{printf "%.0f", $3}')
    local mem_usage=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc -l)
    
    # Temperature information (if available)
    local cpu_temp="0"
    if command -v sensors >/dev/null 2>&1; then
        cpu_temp=$(sensors 2>/dev/null | grep -E "Core 0|Package id 0" | head -1 | awk '{print $3}' | sed 's/+//;s/°C//' || echo "0")
    fi
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    
    echo "$cpu_cores,$cpu_usage,$mem_total,$mem_used,$mem_usage,$cpu_temp,$load_avg"
}

# Function to initialize energy monitoring
init_energy_monitoring() {
    local job_id="$1"
    local user_id="$2"
    local node_name="$3"
    local partition="$4"
    local job_name="$5"
    
    # Create energy log directory if it doesn't exist
    mkdir -p "$ENERGY_LOG_DIR"
    
    # Create job-specific energy log file
    local energy_log_file="$ENERGY_LOG_DIR/job_${job_id}_energy.log"
    
    # Write initial energy monitoring data
    cat > "$energy_log_file" << EOF
# Energy monitoring log for job $job_id
# Job: $job_name
# User: $user_id
# Node: $node_name
# Partition: $partition
# Start time: $(get_timestamp)
# Format: timestamp,cpu_usage,memory_usage,cpu_temp,power_estimate
EOF
    
    # Set permissions
    chmod 644 "$energy_log_file"
    
    log_message "Initialized energy monitoring for job $job_id on node $node_name"
}

# Function to record initial job metrics
record_initial_metrics() {
    local job_id="$1"
    local user_id="$2"
    local node_name="$3"
    local partition="$4"
    local cpu_cores="$5"
    local memory_mb="$6"
    
    # Get system metrics
    local metrics=$(get_system_metrics "$node_name")
    IFS=',' read -r sys_cpu_cores sys_cpu_usage sys_mem_total sys_mem_used sys_mem_usage sys_cpu_temp sys_load_avg <<< "$metrics"
    
    # Calculate initial power estimate (simplified model)
    local base_power=50  # Base power consumption in watts
    local cpu_power=$(echo "scale=2; $sys_cpu_usage * 2.0" | bc -l)  # 2W per % CPU
    local mem_power=$(echo "scale=2; $sys_mem_usage * 0.5" | bc -l)  # 0.5W per % memory
    local estimated_power=$(echo "scale=2; $base_power + $cpu_power + $mem_power" | bc -l)
    
    # Insert initial record into TimescaleDB
    local timestamp=$(get_timestamp)
    
    # Prepare SQL insert statement
    local sql="INSERT INTO hpc_energy.job_metrics (
        time, job_id, job_type, node_id, user_id, partition,
        cpu_cores, memory_mb, cpu_usage, memory_usage,
        avg_cpu_temp, estimated_energy_wh, job_state
    ) VALUES (
        '$timestamp', '$job_id', 'unknown', '$node_name', '$user_id', '$partition',
        $cpu_cores, $memory_mb, $sys_cpu_usage, $sys_mem_usage,
        $sys_cpu_temp, 0, 'STARTING'
    );"
    
    # Execute SQL (with error handling)
    if command -v psql >/dev/null 2>&1; then
        echo "$sql" | psql -h "$TIMESCALE_HOST" -p "$TIMESCALE_PORT" -d "$TIMESCALE_DB" -U "$TIMESCALE_USER" -q 2>/dev/null || {
            log_message "Warning: Failed to insert initial metrics to database for job $job_id"
        }
    else
        log_message "Warning: psql not available, skipping database insert for job $job_id"
    fi
    
    log_message "Recorded initial metrics for job $job_id: CPU=$sys_cpu_usage%, Mem=$sys_mem_usage%, Temp=${sys_cpu_temp}°C, Power=${estimated_power}W"
}

# Function to setup cgroup monitoring
setup_cgroup_monitoring() {
    local job_id="$1"
    local user_id="$2"
    
    # Setup cgroup path for the job
    local cgroup_path="/sys/fs/cgroup/slurm/uid_${user_id}/job_${job_id}"
    
    if [ -d "$cgroup_path" ]; then
        # Enable memory and CPU accounting
        echo 1 > "$cgroup_path/memory.use_hierarchy" 2>/dev/null || true
        echo 1 > "$cgroup_path/cpu.cfs_quota_us" 2>/dev/null || true
        
        log_message "Setup cgroup monitoring for job $job_id at $cgroup_path"
    else
        log_message "Warning: Cgroup path not found for job $job_id"
    fi
}

# Function to start background monitoring
start_background_monitoring() {
    local job_id="$1"
    local node_name="$2"
    
    # Create background monitoring script
    local monitor_script="/tmp/energy_monitor_${job_id}.sh"
    
    cat > "$monitor_script" << 'EOF'
#!/bin/bash

# Background energy monitoring for job
JOB_ID="$1"
NODE_NAME="$2"
ENERGY_LOG_FILE="$3"
MONITOR_INTERVAL=30  # seconds

while [ -f "/tmp/job_${JOB_ID}_running" ]; do
    timestamp=$(date -u '+%Y-%m-%d %H:%M:%S')
    
    # Get current metrics
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "0")
    mem_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}' || echo "0")
    
    # Get temperature
    cpu_temp="0"
    if command -v sensors >/dev/null 2>&1; then
        cpu_temp=$(sensors 2>/dev/null | grep -E "Core 0|Package id 0" | head -1 | awk '{print $3}' | sed 's/+//;s/°C//' || echo "0")
    fi
    
    # Estimate power consumption
    base_power=50
    cpu_power=$(echo "scale=2; $cpu_usage * 2.0" | bc -l 2>/dev/null || echo "0")
    mem_power=$(echo "scale=2; $mem_usage * 0.5" | bc -l 2>/dev/null || echo "0")
    power_estimate=$(echo "scale=2; $base_power + $cpu_power + $mem_power" | bc -l 2>/dev/null || echo "50")
    
    # Log metrics
    echo "$timestamp,$cpu_usage,$mem_usage,$cpu_temp,$power_estimate" >> "$ENERGY_LOG_FILE"
    
    sleep $MONITOR_INTERVAL
done
EOF
    
    chmod +x "$monitor_script"
    
    # Create job running flag
    touch "/tmp/job_${job_id}_running"
    
    # Start background monitoring
    local energy_log_file="$ENERGY_LOG_DIR/job_${job_id}_energy.log"
    nohup "$monitor_script" "$job_id" "$node_name" "$energy_log_file" > "/dev/null" 2>&1 &
    
    log_message "Started background energy monitoring for job $job_id (PID: $!)"
}

# Function to notify external systems
notify_external_systems() {
    local job_id="$1"
    local node_name="$2"
    local user_id="$3"
    
    # Notify Prometheus job exporter
    if command -v curl >/dev/null 2>&1; then
        curl -s -X POST "$METRICS_ENDPOINT/job/start" \
            -H "Content-Type: application/json" \
            -d "{\"job_id\": \"$job_id\", \"node_id\": \"$node_name\", \"user_id\": \"$user_id\"}" \
            >/dev/null 2>&1 || {
            log_message "Warning: Failed to notify metrics endpoint for job $job_id"
        }
    fi
    
    log_message "Notified external systems about job $job_id start"
}

# Main execution
main() {
    # Get Slurm environment variables
    local job_id="${SLURM_JOB_ID:-unknown}"
    local user_id="${SLURM_JOB_USER:-unknown}"
    local node_name="${SLURMD_NODENAME:-$(hostname)}"
    local partition="${SLURM_JOB_PARTITION:-unknown}"
    local job_name="${SLURM_JOB_NAME:-unknown}"
    local cpu_cores="${SLURM_CPUS_ON_NODE:-1}"
    local memory_mb="${SLURM_MEM_PER_NODE:-1024}"
    
    log_message "Starting prolog for job $job_id (user: $user_id, node: $node_name, partition: $partition)"
    
    # Initialize energy monitoring
    init_energy_monitoring "$job_id" "$user_id" "$node_name" "$partition" "$job_name"
    
    # Record initial metrics
    record_initial_metrics "$job_id" "$user_id" "$node_name" "$partition" "$cpu_cores" "$memory_mb"
    
    # Setup cgroup monitoring
    setup_cgroup_monitoring "$job_id" "$user_id"
    
    # Start background monitoring
    start_background_monitoring "$job_id" "$node_name"
    
    # Notify external systems
    notify_external_systems "$job_id" "$node_name" "$user_id"
    
    log_message "Prolog completed successfully for job $job_id"
}

# Error handling
trap 'log_message "Error occurred in prolog script for job ${SLURM_JOB_ID:-unknown}"' ERR

# Execute main function
main "$@"

exit 0