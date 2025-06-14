#!/usr/bin/env python3
"""
Slurm Job Exporter for HPC Energy Model

Custom Prometheus exporter that collects job-specific metrics from Slurm,
including job states, resource usage, and energy consumption estimates.

Author: HPC Energy Model Project
License: MIT
"""

import time
import os
import subprocess
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client.core import CollectorRegistry
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlurmJobCollector:
    """
    Collects job-specific metrics from Slurm and stores them in TimescaleDB:
    - Job states and transitions
    - Resource utilization per job
    - Energy consumption estimates
    - Queue times and execution times
    - Node allocation patterns
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Database connection parameters
        self.db_config = {
            'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
            'port': int(os.getenv('TIMESCALEDB_PORT', 5432)),
            'database': os.getenv('TIMESCALEDB_DB', 'hpc_energy'),
            'user': os.getenv('TIMESCALEDB_USER', 'hpc_user'),
            'password': os.getenv('TIMESCALEDB_PASSWORD', 'hpc_password')
        }
        
        # Prometheus metrics
        self.job_state_gauge = Gauge(
            'hpc_slurm_job_state',
            'Current state of Slurm jobs (0=pending, 1=running, 2=completed, 3=failed)',
            ['job_id', 'user', 'partition', 'job_name'],
            registry=self.registry
        )
        
        self.job_runtime_gauge = Gauge(
            'hpc_slurm_job_runtime_seconds',
            'Job runtime in seconds',
            ['job_id', 'user', 'partition'],
            registry=self.registry
        )
        
        self.job_queue_time_gauge = Gauge(
            'hpc_slurm_job_queue_time_seconds',
            'Time job spent in queue before execution',
            ['job_id', 'user', 'partition'],
            registry=self.registry
        )
        
        self.job_cpu_usage_gauge = Gauge(
            'hpc_slurm_job_cpu_usage_percent',
            'CPU usage percentage for running jobs',
            ['job_id', 'node'],
            registry=self.registry
        )
        
        self.job_memory_usage_gauge = Gauge(
            'hpc_slurm_job_memory_usage_bytes',
            'Memory usage in bytes for running jobs',
            ['job_id', 'node'],
            registry=self.registry
        )
        
        self.job_energy_estimate_gauge = Gauge(
            'hpc_slurm_job_energy_estimate_joules',
            'Estimated energy consumption for jobs',
            ['job_id', 'job_type'],
            registry=self.registry
        )
        
        self.jobs_total_counter = Counter(
            'hpc_slurm_jobs_total',
            'Total number of jobs processed',
            ['state', 'partition'],
            registry=self.registry
        )
        
        self.job_completion_time_histogram = Histogram(
            'hpc_slurm_job_completion_time_seconds',
            'Histogram of job completion times',
            ['partition', 'job_type'],
            registry=self.registry
        )
        
        # System info
        self.cluster_info = Info(
            'hpc_slurm_cluster_info',
            'Slurm cluster information',
            registry=self.registry
        )
        
        self._update_cluster_info()
        self._init_database()
    
    def _update_cluster_info(self):
        """Update cluster information metrics"""
        try:
            # Get cluster info from sinfo
            result = subprocess.run(
                ['sinfo', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                cluster_data = json.loads(result.stdout)
                nodes = cluster_data.get('nodes', [])
                
                total_nodes = len(nodes)
                total_cpus = sum(node.get('cpus', 0) for node in nodes)
                total_memory = sum(node.get('real_memory', 0) for node in nodes)
                
                info_dict = {
                    'total_nodes': str(total_nodes),
                    'total_cpus': str(total_cpus),
                    'total_memory_mb': str(total_memory),
                    'slurm_version': self._get_slurm_version(),
                }
                
                self.cluster_info.info(info_dict)
            
        except Exception as e:
            logger.error(f"Error updating cluster info: {e}")
    
    def _get_slurm_version(self) -> str:
        """Get Slurm version"""
        try:
            result = subprocess.run(
                ['sinfo', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Error getting Slurm version: {e}")
        return 'unknown'
    
    def _init_database(self):
        """Initialize TimescaleDB tables for job metrics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create jobs table with hypertable
            cur.execute("""
                CREATE TABLE IF NOT EXISTS job_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    job_id VARCHAR(50) NOT NULL,
                    user_name VARCHAR(100),
                    partition VARCHAR(100),
                    job_name VARCHAR(200),
                    job_state VARCHAR(20),
                    job_type VARCHAR(50),
                    nodes_allocated INTEGER,
                    cpus_allocated INTEGER,
                    memory_allocated_mb BIGINT,
                    runtime_seconds INTEGER,
                    queue_time_seconds INTEGER,
                    cpu_usage_percent REAL,
                    memory_usage_bytes BIGINT,
                    energy_estimate_joules REAL,
                    node_list TEXT,
                    submit_time TIMESTAMPTZ,
                    start_time TIMESTAMPTZ,
                    end_time TIMESTAMPTZ,
                    exit_code INTEGER,
                    PRIMARY KEY (timestamp, job_id)
                );
            """)
            
            # Create hypertable if not exists
            cur.execute("""
                SELECT create_hypertable('job_metrics', 'timestamp', 
                                       if_not_exists => TRUE);
            """)
            
            # Create indexes for better query performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_metrics_job_id 
                ON job_metrics (job_id);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_metrics_user 
                ON job_metrics (user_name);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_metrics_state 
                ON job_metrics (job_state);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _get_job_data(self) -> List[Dict[str, Any]]:
        """Get job data from Slurm using sacct"""
        try:
            # Get jobs from last 24 hours
            start_time = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d')
            
            cmd = [
                'sacct',
                '--json',
                '--starttime', start_time,
                '--allusers',
                '--allocations',
                '--format=JobID,JobName,User,Partition,State,Submit,Start,End,'
                        'Elapsed,CPUTime,ReqCPUS,ReqMem,NodeList,ExitCode'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('jobs', [])
            else:
                logger.error(f"sacct command failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting job data: {e}")
            return []
    
    def _get_running_job_stats(self, job_id: str) -> Dict[str, float]:
        """Get detailed stats for running jobs using sstat"""
        try:
            cmd = [
                'sstat',
                '--json',
                '--jobs', job_id,
                '--format=JobID,AveCPU,AveRSS,MaxRSS'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stats = data.get('statistics', [])
                if stats:
                    stat = stats[0]
                    return {
                        'cpu_usage': self._parse_cpu_time(stat.get('average', {}).get('cpu', '0')),
                        'memory_usage': self._parse_memory(stat.get('average', {}).get('rss', '0')),
                        'max_memory': self._parse_memory(stat.get('maximum', {}).get('rss', '0'))
                    }
            
        except Exception as e:
            logger.debug(f"Error getting job stats for {job_id}: {e}")
        
        return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'max_memory': 0.0}
    
    def _parse_cpu_time(self, cpu_str: str) -> float:
        """Parse CPU time string to percentage"""
        try:
            # CPU time format: "00:01:30" or similar
            if ':' in cpu_str:
                parts = cpu_str.split(':')
                if len(parts) >= 2:
                    minutes = int(parts[-2])
                    seconds = int(parts[-1])
                    return (minutes * 60 + seconds) / 60.0  # Convert to percentage approximation
        except Exception:
            pass
        return 0.0
    
    def _parse_memory(self, mem_str: str) -> int:
        """Parse memory string to bytes"""
        try:
            if not mem_str or mem_str == '0':
                return 0
            
            # Remove units and convert
            mem_str = mem_str.upper().replace('B', '')
            
            if 'K' in mem_str:
                return int(float(mem_str.replace('K', '')) * 1024)
            elif 'M' in mem_str:
                return int(float(mem_str.replace('M', '')) * 1024 * 1024)
            elif 'G' in mem_str:
                return int(float(mem_str.replace('G', '')) * 1024 * 1024 * 1024)
            else:
                return int(mem_str)
                
        except Exception:
            return 0
    
    def _estimate_job_energy(self, job_data: Dict[str, Any], stats: Dict[str, float]) -> float:
        """Estimate energy consumption for a job"""
        try:
            # Simple energy estimation model
            # In production, this would use actual power measurements
            
            runtime_seconds = self._parse_time_to_seconds(job_data.get('time', {}).get('elapsed', '0'))
            cpus_allocated = job_data.get('required', {}).get('cpus', 1)
            
            # Base power consumption per CPU (watts)
            base_power_per_cpu = 20
            
            # Dynamic power based on CPU usage
            cpu_usage_factor = stats.get('cpu_usage', 50) / 100.0
            dynamic_power_per_cpu = 30 * cpu_usage_factor
            
            total_power = (base_power_per_cpu + dynamic_power_per_cpu) * cpus_allocated
            energy_joules = total_power * runtime_seconds
            
            return energy_joules
            
        except Exception as e:
            logger.debug(f"Error estimating energy: {e}")
            return 0.0
    
    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parse time string to seconds"""
        try:
            if not time_str or time_str == '0':
                return 0
            
            # Format: "HH:MM:SS" or "MM:SS" or "SS"
            parts = time_str.split(':')
            
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                return int(parts[0])
                
        except Exception:
            return 0
    
    def _classify_job_type(self, job_data: Dict[str, Any]) -> str:
        """Classify job type based on resource requirements"""
        try:
            job_name = job_data.get('name', '').lower()
            
            # Simple classification based on job name patterns
            if any(keyword in job_name for keyword in ['cpu', 'compute', 'calc']):
                return 'cpu_intensive'
            elif any(keyword in job_name for keyword in ['io', 'read', 'write', 'copy']):
                return 'io_intensive'
            elif any(keyword in job_name for keyword in ['mixed', 'hybrid']):
                return 'mixed_workload'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _store_job_metrics(self, job_data: Dict[str, Any], stats: Dict[str, float]):
        """Store job metrics in TimescaleDB"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Prepare data
            timestamp = datetime.now()
            job_id = job_data.get('job_id', '')
            user_name = job_data.get('user', '')
            partition = job_data.get('partition', '')
            job_name = job_data.get('name', '')
            job_state = job_data.get('state', {}).get('current', '')
            job_type = self._classify_job_type(job_data)
            
            runtime_seconds = self._parse_time_to_seconds(
                job_data.get('time', {}).get('elapsed', '0')
            )
            
            energy_estimate = self._estimate_job_energy(job_data, stats)
            
            # Insert data
            cur.execute("""
                INSERT INTO job_metrics (
                    timestamp, job_id, user_name, partition, job_name,
                    job_state, job_type, cpus_allocated, runtime_seconds,
                    cpu_usage_percent, memory_usage_bytes, energy_estimate_joules
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp, job_id) DO UPDATE SET
                    job_state = EXCLUDED.job_state,
                    runtime_seconds = EXCLUDED.runtime_seconds,
                    cpu_usage_percent = EXCLUDED.cpu_usage_percent,
                    memory_usage_bytes = EXCLUDED.memory_usage_bytes,
                    energy_estimate_joules = EXCLUDED.energy_estimate_joules
            """, (
                timestamp, job_id, user_name, partition, job_name,
                job_state, job_type, job_data.get('required', {}).get('cpus', 1),
                runtime_seconds, stats.get('cpu_usage', 0),
                stats.get('memory_usage', 0), energy_estimate
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing job metrics: {e}")
    
    def collect_metrics(self):
        """Collect all job metrics"""
        logger.info("Collecting Slurm job metrics...")
        
        try:
            jobs = self._get_job_data()
            
            for job in jobs:
                job_id = job.get('job_id', '')
                job_state = job.get('state', {}).get('current', '')
                user = job.get('user', '')
                partition = job.get('partition', '')
                job_name = job.get('name', '')
                
                # Update Prometheus metrics
                state_value = {
                    'PENDING': 0, 'RUNNING': 1, 'COMPLETED': 2, 
                    'FAILED': 3, 'CANCELLED': 3
                }.get(job_state, 0)
                
                self.job_state_gauge.labels(
                    job_id=job_id, user=user, partition=partition, job_name=job_name
                ).set(state_value)
                
                # Get detailed stats for running jobs
                if job_state == 'RUNNING':
                    stats = self._get_running_job_stats(job_id)
                    
                    self.job_cpu_usage_gauge.labels(
                        job_id=job_id, node='unknown'
                    ).set(stats.get('cpu_usage', 0))
                    
                    self.job_memory_usage_gauge.labels(
                        job_id=job_id, node='unknown'
                    ).set(stats.get('memory_usage', 0))
                else:
                    stats = {'cpu_usage': 0, 'memory_usage': 0}
                
                # Store in database
                self._store_job_metrics(job, stats)
                
                # Update counters
                self.jobs_total_counter.labels(
                    state=job_state, partition=partition
                ).inc()
            
            logger.info(f"Processed {len(jobs)} jobs")
            
        except Exception as e:
            logger.error(f"Error collecting job metrics: {e}")

def main():
    """Main function to start the job exporter"""
    port = int(os.getenv('EXPORTER_PORT', 9300))
    polling_interval = int(os.getenv('POLLING_INTERVAL', 30))
    
    logger.info(f"Starting HPC Job Exporter on port {port}")
    logger.info(f"Polling interval: {polling_interval} seconds")
    
    # Create collector
    collector = SlurmJobCollector()
    
    # Start HTTP server
    start_http_server(port, registry=collector.registry)
    
    logger.info(f"Job exporter started successfully on port {port}")
    
    # Main collection loop
    try:
        while True:
            collector.collect_metrics()
            time.sleep(polling_interval)
    except KeyboardInterrupt:
        logger.info("Job exporter stopped by user")
    except Exception as e:
        logger.error(f"Job exporter error: {e}")
        raise

if __name__ == '__main__':
    main()