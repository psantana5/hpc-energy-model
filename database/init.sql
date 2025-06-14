-- HPC Energy Model Database Initialization Script
-- Creates TimescaleDB hypertables and necessary schemas

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create database schema
CREATE SCHEMA IF NOT EXISTS hpc_energy;
SET search_path TO hpc_energy, public;

-- Job metrics table (main hypertable)
CREATE TABLE IF NOT EXISTS job_metrics (
    time TIMESTAMPTZ NOT NULL,
    job_id VARCHAR(50) NOT NULL,
    job_type VARCHAR(20) NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    partition VARCHAR(50),
    
    -- Job execution metrics
    duration_seconds INTEGER,
    cpu_cores INTEGER,
    memory_mb INTEGER,
    
    -- Resource usage
    cpu_usage REAL,
    memory_usage REAL,
    io_read_mbps REAL,
    io_write_mbps REAL,
    network_rx_mbps REAL,
    network_tx_mbps REAL,
    
    -- Thermal metrics
    avg_cpu_temp REAL,
    peak_cpu_temp REAL,
    avg_gpu_temp REAL,
    peak_gpu_temp REAL,
    thermal_throttling BOOLEAN DEFAULT FALSE,
    
    -- Power and energy
    avg_power_watts REAL,
    peak_power_watts REAL,
    estimated_energy_wh REAL,
    
    -- Job status
    job_state VARCHAR(20),
    exit_code INTEGER,
    
    -- Additional metadata
    workload_pattern VARCHAR(50),
    prediction_accuracy REAL,
    
    PRIMARY KEY (time, job_id)
);

-- Convert to hypertable
SELECT create_hypertable('job_metrics', 'time', if_not_exists => TRUE);

-- Node metrics table
CREATE TABLE IF NOT EXISTS node_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    
    -- System metrics
    cpu_usage REAL,
    memory_usage REAL,
    disk_usage REAL,
    load_avg_1m REAL,
    load_avg_5m REAL,
    load_avg_15m REAL,
    
    -- Thermal metrics
    cpu_temp REAL,
    gpu_temp REAL,
    system_temp REAL,
    fan_speed_rpm INTEGER,
    
    -- Power metrics
    power_consumption_watts REAL,
    energy_consumed_wh REAL,
    
    -- Network metrics
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    
    -- Disk I/O metrics
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    
    PRIMARY KEY (time, node_id)
);

-- Convert to hypertable
SELECT create_hypertable('node_metrics', 'time', if_not_exists => TRUE);

-- Energy predictions table
CREATE TABLE IF NOT EXISTS energy_predictions (
    time TIMESTAMPTZ NOT NULL,
    job_id VARCHAR(50) NOT NULL,
    
    -- Prediction inputs
    predicted_duration_seconds INTEGER,
    predicted_cpu_usage REAL,
    predicted_memory_usage REAL,
    predicted_io_usage REAL,
    
    -- Prediction outputs
    predicted_energy_wh REAL,
    predicted_power_watts REAL,
    predicted_temp_celsius REAL,
    
    -- Model metadata
    model_version VARCHAR(20),
    confidence_score REAL,
    prediction_method VARCHAR(50),
    
    -- Actual vs predicted (filled after job completion)
    actual_energy_wh REAL,
    actual_duration_seconds INTEGER,
    prediction_error_percent REAL,
    
    PRIMARY KEY (time, job_id)
);

-- Convert to hypertable
SELECT create_hypertable('energy_predictions', 'time', if_not_exists => TRUE);

-- Thermal events table
CREATE TABLE IF NOT EXISTS thermal_events (
    time TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(30) NOT NULL, -- 'throttling', 'overheat', 'cooling'
    
    -- Event details
    temperature REAL,
    threshold REAL,
    duration_seconds INTEGER,
    severity VARCHAR(10), -- 'low', 'medium', 'high', 'critical'
    
    -- Associated job (if any)
    job_id VARCHAR(50),
    
    -- Impact metrics
    performance_impact_percent REAL,
    energy_impact_wh REAL,
    
    PRIMARY KEY (time, node_id, event_type)
);

-- Convert to hypertable
SELECT create_hypertable('thermal_events', 'time', if_not_exists => TRUE);

-- Workload patterns table
CREATE TABLE IF NOT EXISTS workload_patterns (
    pattern_id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(50) UNIQUE NOT NULL,
    
    -- Pattern characteristics
    avg_cpu_usage REAL,
    avg_memory_usage REAL,
    avg_io_usage REAL,
    avg_duration_seconds INTEGER,
    avg_energy_wh REAL,
    
    -- Pattern metadata
    sample_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

-- Energy efficiency metrics (aggregated)
CREATE TABLE IF NOT EXISTS energy_efficiency (
    time TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    
    -- Efficiency metrics
    energy_per_job_wh REAL,
    energy_per_cpu_hour_wh REAL,
    energy_per_gb_hour_wh REAL,
    
    -- Performance metrics
    jobs_completed INTEGER,
    total_cpu_hours REAL,
    total_memory_gb_hours REAL,
    
    -- Thermal efficiency
    avg_thermal_efficiency REAL, -- performance per degree
    cooling_efficiency REAL,
    
    PRIMARY KEY (time, node_id)
);

-- Convert to hypertable
SELECT create_hypertable('energy_efficiency', 'time', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_job_metrics_job_id ON job_metrics (job_id);
CREATE INDEX IF NOT EXISTS idx_job_metrics_job_type ON job_metrics (job_type);
CREATE INDEX IF NOT EXISTS idx_job_metrics_node_id ON job_metrics (node_id);
CREATE INDEX IF NOT EXISTS idx_job_metrics_energy ON job_metrics (estimated_energy_wh);
CREATE INDEX IF NOT EXISTS idx_job_metrics_state ON job_metrics (job_state);

CREATE INDEX IF NOT EXISTS idx_node_metrics_node_id ON node_metrics (node_id);
CREATE INDEX IF NOT EXISTS idx_node_metrics_power ON node_metrics (power_consumption_watts);

CREATE INDEX IF NOT EXISTS idx_energy_predictions_job_id ON energy_predictions (job_id);
CREATE INDEX IF NOT EXISTS idx_energy_predictions_model ON energy_predictions (model_version);

CREATE INDEX IF NOT EXISTS idx_thermal_events_node_id ON thermal_events (node_id);
CREATE INDEX IF NOT EXISTS idx_thermal_events_type ON thermal_events (event_type);
CREATE INDEX IF NOT EXISTS idx_thermal_events_severity ON thermal_events (severity);

-- Create continuous aggregates for common queries

-- Hourly job metrics aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS job_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    job_type,
    node_id,
    COUNT(*) as job_count,
    AVG(duration_seconds) as avg_duration,
    AVG(cpu_usage) as avg_cpu_usage,
    AVG(memory_usage) as avg_memory_usage,
    AVG(estimated_energy_wh) as avg_energy,
    SUM(estimated_energy_wh) as total_energy,
    AVG(avg_cpu_temp) as avg_temp,
    MAX(peak_cpu_temp) as max_temp
FROM job_metrics
GROUP BY bucket, job_type, node_id;

-- Daily energy efficiency aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS energy_efficiency_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    node_id,
    AVG(energy_per_job_wh) as avg_energy_per_job,
    AVG(energy_per_cpu_hour_wh) as avg_energy_per_cpu_hour,
    SUM(jobs_completed) as total_jobs,
    SUM(total_cpu_hours) as total_cpu_hours,
    AVG(avg_thermal_efficiency) as avg_thermal_efficiency
FROM energy_efficiency
GROUP BY bucket, node_id;

-- Set up refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('job_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('energy_efficiency_daily',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create retention policies (keep raw data for 30 days, aggregates for 1 year)
SELECT add_retention_policy('job_metrics', INTERVAL '30 days');
SELECT add_retention_policy('node_metrics', INTERVAL '30 days');
SELECT add_retention_policy('thermal_events', INTERVAL '90 days');
SELECT add_retention_policy('energy_predictions', INTERVAL '90 days');

-- Create useful views

-- Current running jobs view
CREATE OR REPLACE VIEW current_jobs AS
SELECT 
    job_id,
    job_type,
    node_id,
    user_id,
    time as start_time,
    cpu_cores,
    memory_mb,
    cpu_usage,
    memory_usage,
    avg_cpu_temp,
    estimated_energy_wh
FROM job_metrics
WHERE job_state = 'RUNNING'
AND time >= NOW() - INTERVAL '1 hour';

-- Energy efficiency ranking view
CREATE OR REPLACE VIEW node_efficiency_ranking AS
SELECT 
    node_id,
    AVG(energy_per_job_wh) as avg_energy_per_job,
    AVG(energy_per_cpu_hour_wh) as avg_energy_per_cpu_hour,
    COUNT(*) as measurement_count,
    RANK() OVER (ORDER BY AVG(energy_per_job_wh) ASC) as efficiency_rank
FROM energy_efficiency
WHERE time >= NOW() - INTERVAL '7 days'
GROUP BY node_id
ORDER BY efficiency_rank;

-- Thermal hotspots view
CREATE OR REPLACE VIEW thermal_hotspots AS
SELECT 
    node_id,
    COUNT(*) as event_count,
    AVG(temperature) as avg_temp,
    MAX(temperature) as max_temp,
    SUM(duration_seconds) as total_duration_seconds,
    AVG(performance_impact_percent) as avg_performance_impact
FROM thermal_events
WHERE time >= NOW() - INTERVAL '24 hours'
AND severity IN ('high', 'critical')
GROUP BY node_id
ORDER BY event_count DESC, max_temp DESC;

-- Job energy prediction accuracy view
CREATE OR REPLACE VIEW prediction_accuracy AS
SELECT 
    model_version,
    prediction_method,
    COUNT(*) as prediction_count,
    AVG(ABS(prediction_error_percent)) as avg_absolute_error,
    STDDEV(prediction_error_percent) as error_stddev,
    COUNT(CASE WHEN ABS(prediction_error_percent) <= 10 THEN 1 END) * 100.0 / COUNT(*) as accuracy_within_10_percent
FROM energy_predictions
WHERE actual_energy_wh IS NOT NULL
AND time >= NOW() - INTERVAL '7 days'
GROUP BY model_version, prediction_method
ORDER BY avg_absolute_error ASC;

-- Insert sample data for testing
INSERT INTO workload_patterns (pattern_name, avg_cpu_usage, avg_memory_usage, avg_io_usage, avg_duration_seconds, avg_energy_wh, description) VALUES
('cpu_intensive', 85.0, 30.0, 5.0, 1800, 150.0, 'CPU-bound computational workloads'),
('io_intensive', 20.0, 60.0, 80.0, 3600, 120.0, 'I/O-bound data processing workloads'),
('mixed_workload', 50.0, 45.0, 40.0, 2400, 135.0, 'Balanced CPU and I/O workloads'),
('memory_intensive', 40.0, 90.0, 15.0, 2700, 140.0, 'Memory-bound analytical workloads')
ON CONFLICT (pattern_name) DO NOTHING;

-- Create functions for common operations

-- Function to calculate energy efficiency
CREATE OR REPLACE FUNCTION calculate_energy_efficiency(
    p_node_id VARCHAR(50),
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    energy_per_job REAL,
    energy_per_cpu_hour REAL,
    total_jobs INTEGER,
    total_energy REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SUM(estimated_energy_wh) / COUNT(*))::REAL as energy_per_job,
        (SUM(estimated_energy_wh) / SUM(duration_seconds * cpu_cores / 3600.0))::REAL as energy_per_cpu_hour,
        COUNT(*)::INTEGER as total_jobs,
        SUM(estimated_energy_wh)::REAL as total_energy
    FROM job_metrics
    WHERE node_id = p_node_id
    AND time BETWEEN p_start_time AND p_end_time
    AND job_state = 'COMPLETED';
END;
$$ LANGUAGE plpgsql;

-- Function to get thermal events summary
CREATE OR REPLACE FUNCTION get_thermal_events_summary(
    p_node_id VARCHAR(50) DEFAULT NULL,
    p_hours_back INTEGER DEFAULT 24
)
RETURNS TABLE (
    node_id VARCHAR(50),
    event_type VARCHAR(30),
    event_count BIGINT,
    avg_temperature REAL,
    total_duration INTEGER,
    avg_impact REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        te.node_id,
        te.event_type,
        COUNT(*) as event_count,
        AVG(te.temperature)::REAL as avg_temperature,
        SUM(te.duration_seconds)::INTEGER as total_duration,
        AVG(te.performance_impact_percent)::REAL as avg_impact
    FROM thermal_events te
    WHERE (p_node_id IS NULL OR te.node_id = p_node_id)
    AND te.time >= NOW() - (p_hours_back || ' hours')::INTERVAL
    GROUP BY te.node_id, te.event_type
    ORDER BY te.node_id, event_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA hpc_energy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA hpc_energy TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA hpc_energy TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA hpc_energy TO PUBLIC;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA hpc_energy GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA hpc_energy GRANT USAGE, SELECT ON SEQUENCES TO PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA hpc_energy GRANT EXECUTE ON FUNCTIONS TO PUBLIC;

COMMIT;

-- Display initialization summary
\echo 'HPC Energy Model database initialized successfully!'
\echo 'Created tables: job_metrics, node_metrics, energy_predictions, thermal_events, workload_patterns, energy_efficiency'
\echo 'Created continuous aggregates: job_metrics_hourly, energy_efficiency_daily'
\echo 'Created views: current_jobs, node_efficiency_ranking, thermal_hotspots, prediction_accuracy'
\echo 'Created functions: calculate_energy_efficiency, get_thermal_events_summary'