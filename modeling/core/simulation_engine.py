# HPC Cluster Simulation Engine using SimPy

import simpy
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from collections import defaultdict

try:
    from ..utils.config import ModelingConfig, SimulationConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import ModelingConfig, SimulationConfig
from .data_loader import HistoricalDataLoader

logger = logging.getLogger(__name__)

@dataclass
class JobSpec:
    """Job specification for simulation"""
    job_id: str
    job_type: str
    cpu_cores: int
    memory_mb: int
    duration_seconds: int
    arrival_time: float
    priority: int = 1
    user_id: str = "default"
    partition: str = "compute"
    workload_pattern: str = "cpu_intensive"
    expected_power_profile: Optional[List[float]] = None
    expected_thermal_profile: Optional[List[float]] = None

@dataclass
class NodeSpec:
    """Node specification for simulation"""
    node_id: str
    cpu_cores: int
    memory_mb: int
    max_power_watts: float
    idle_power_watts: float
    thermal_capacity: float
    cooling_efficiency: float
    max_temperature: float = 85.0
    base_temperature: float = 25.0
    location: str = "rack1"

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    job_metrics: List[Dict] = field(default_factory=list)
    node_metrics: List[Dict] = field(default_factory=list)
    thermal_events: List[Dict] = field(default_factory=list)
    energy_consumption: List[Dict] = field(default_factory=list)
    queue_statistics: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)

class ThermalModel:
    """Simple thermal model for nodes"""
    
    def __init__(self, node_spec: NodeSpec):
        self.node_spec = node_spec
        self.current_temp = node_spec.base_temperature
        self.ambient_temp = 22.0
        self.thermal_time_constant = 300.0  # seconds
    
    def update_temperature(self, power_watts: float, dt: float) -> float:
        """
        Update node temperature based on power consumption
        
        Args:
            power_watts: Current power consumption
            dt: Time step in seconds
            
        Returns:
            Updated temperature in Celsius
        """
        # Simple thermal model: dT/dt = (P - cooling) / thermal_capacity
        power_to_heat = power_watts * 0.8  # 80% of power becomes heat
        cooling_power = (self.current_temp - self.ambient_temp) * self.node_spec.cooling_efficiency
        
        temp_change = (power_to_heat - cooling_power) / self.node_spec.thermal_capacity * dt
        self.current_temp += temp_change
        
        # Ensure temperature doesn't go below ambient
        self.current_temp = max(self.current_temp, self.ambient_temp)
        
        return self.current_temp
    
    def is_throttling(self) -> bool:
        """Check if node should throttle due to temperature"""
        return self.current_temp > self.node_spec.max_temperature * 0.9

class PowerModel:
    """Power consumption model for jobs and nodes"""
    
    def __init__(self, node_spec: NodeSpec):
        self.node_spec = node_spec
        self.base_models = {
            'cpu_intensive': self._cpu_intensive_model,
            'memory_intensive': self._memory_intensive_model,
            'io_intensive': self._io_intensive_model,
            'mixed': self._mixed_model
        }
    
    def calculate_power(self, job_spec: JobSpec, cpu_utilization: float, 
                      memory_utilization: float, time_in_job: float) -> float:
        """
        Calculate power consumption for a job
        
        Args:
            job_spec: Job specification
            cpu_utilization: CPU utilization (0-1)
            memory_utilization: Memory utilization (0-1)
            time_in_job: Time elapsed in job (seconds)
            
        Returns:
            Power consumption in watts
        """
        model_func = self.base_models.get(job_spec.workload_pattern, self._mixed_model)
        return model_func(job_spec, cpu_utilization, memory_utilization, time_in_job)
    
    def _cpu_intensive_model(self, job_spec: JobSpec, cpu_util: float, 
                           mem_util: float, time_in_job: float) -> float:
        """Power model for CPU-intensive workloads"""
        base_power = self.node_spec.idle_power_watts
        cpu_power = (self.node_spec.max_power_watts - base_power) * cpu_util
        memory_power = 20 * mem_util  # Memory power is relatively low
        
        # Add some variability based on time
        variability = 1 + 0.1 * np.sin(time_in_job / 60.0)  # 10% variation over time
        
        return (base_power + cpu_power + memory_power) * variability
    
    def _memory_intensive_model(self, job_spec: JobSpec, cpu_util: float, 
                              mem_util: float, time_in_job: float) -> float:
        """Power model for memory-intensive workloads"""
        base_power = self.node_spec.idle_power_watts
        cpu_power = (self.node_spec.max_power_watts - base_power) * 0.6 * cpu_util
        memory_power = 40 * mem_util  # Higher memory power
        
        variability = 1 + 0.05 * np.sin(time_in_job / 120.0)
        
        return (base_power + cpu_power + memory_power) * variability
    
    def _io_intensive_model(self, job_spec: JobSpec, cpu_util: float, 
                          mem_util: float, time_in_job: float) -> float:
        """Power model for I/O-intensive workloads"""
        base_power = self.node_spec.idle_power_watts
        cpu_power = (self.node_spec.max_power_watts - base_power) * 0.4 * cpu_util
        memory_power = 15 * mem_util
        io_power = 30  # Constant I/O power
        
        # I/O workloads often have burst patterns
        burst_factor = 1 + 0.3 * (np.sin(time_in_job / 30.0) > 0.5)
        
        return (base_power + cpu_power + memory_power + io_power) * burst_factor
    
    def _mixed_model(self, job_spec: JobSpec, cpu_util: float, 
                   mem_util: float, time_in_job: float) -> float:
        """Power model for mixed workloads"""
        base_power = self.node_spec.idle_power_watts
        cpu_power = (self.node_spec.max_power_watts - base_power) * 0.7 * cpu_util
        memory_power = 25 * mem_util
        
        # Mixed workloads have more complex patterns
        pattern1 = 0.1 * np.sin(time_in_job / 60.0)
        pattern2 = 0.05 * np.cos(time_in_job / 180.0)
        variability = 1 + pattern1 + pattern2
        
        return (base_power + cpu_power + memory_power) * variability

class SimulatedNode:
    """Simulated compute node"""
    
    def __init__(self, env: simpy.Environment, node_spec: NodeSpec):
        self.env = env
        self.spec = node_spec
        self.cpu_resource = simpy.Resource(env, capacity=node_spec.cpu_cores)
        self.memory_resource = simpy.Container(env, capacity=node_spec.memory_mb, init=node_spec.memory_mb)
        
        self.thermal_model = ThermalModel(node_spec)
        self.power_model = PowerModel(node_spec)
        
        self.current_jobs = []
        self.metrics_history = []
        
        # Start monitoring process
        self.env.process(self._monitor())
    
    def _monitor(self):
        """Monitor node metrics continuously"""
        while True:
            # Calculate current utilization
            cpu_utilization = (self.spec.cpu_cores - self.cpu_resource.count) / self.spec.cpu_cores
            memory_utilization = (self.spec.memory_mb - self.memory_resource.level) / self.spec.memory_mb
            
            # Calculate total power consumption
            total_power = self.spec.idle_power_watts
            for job in self.current_jobs:
                job_power = self.power_model.calculate_power(
                    job['spec'], cpu_utilization, memory_utilization, 
                    self.env.now - job['start_time']
                )
                total_power += job_power
            
            # Update thermal model
            temperature = self.thermal_model.update_temperature(total_power, 60.0)  # 1-minute updates
            
            # Record metrics
            metrics = {
                'time': self.env.now,
                'node_id': self.spec.node_id,
                'cpu_utilization': cpu_utilization,
                'memory_utilization': memory_utilization,
                'power_consumption': total_power,
                'temperature': temperature,
                'thermal_throttling': self.thermal_model.is_throttling(),
                'active_jobs': len(self.current_jobs)
            }
            self.metrics_history.append(metrics)
            
            yield self.env.timeout(60)  # Monitor every minute
    
    def execute_job(self, job_spec: JobSpec):
        """Execute a job on this node"""
        job_info = {
            'spec': job_spec,
            'start_time': self.env.now,
            'node_id': self.spec.node_id
        }
        
        # Request resources
        with self.cpu_resource.request() as cpu_req:
            yield cpu_req
            
            # Request memory
            yield self.memory_resource.get(job_spec.memory_mb)
            
            self.current_jobs.append(job_info)
            logger.debug(f"Job {job_spec.job_id} started on node {self.spec.node_id}")
            
            # Simulate job execution
            execution_time = job_spec.duration_seconds
            
            # Apply thermal throttling if necessary
            if self.thermal_model.is_throttling():
                execution_time *= 1.2  # 20% slowdown due to throttling
                logger.warning(f"Thermal throttling on node {self.spec.node_id}")
            
            yield self.env.timeout(execution_time)
            
            # Job completed
            self.current_jobs.remove(job_info)
            self.memory_resource.put(job_spec.memory_mb)
            
            logger.debug(f"Job {job_spec.job_id} completed on node {self.spec.node_id}")
            
            return {
                'job_id': job_spec.job_id,
                'node_id': self.spec.node_id,
                'start_time': job_info['start_time'],
                'end_time': self.env.now,
                'actual_duration': self.env.now - job_info['start_time'],
                'expected_duration': job_spec.duration_seconds
            }

class HPCClusterSimulator:
    """Main HPC cluster simulator"""
    
    def __init__(self, config: ModelingConfig):
        self.config = config
        self.sim_config = config.simulation
        
        self.env = simpy.Environment()
        self.nodes = {}
        self.job_queue = []
        self.completed_jobs = []
        self.metrics = SimulationMetrics()
        
        # Scheduler
        self.scheduler_process = None
        
    def add_node(self, node_spec: NodeSpec):
        """Add a compute node to the cluster"""
        node = SimulatedNode(self.env, node_spec)
        self.nodes[node_spec.node_id] = node
        logger.info(f"Added node {node_spec.node_id} to cluster")
    
    def submit_job(self, job_spec: JobSpec):
        """Submit a job to the cluster"""
        self.job_queue.append(job_spec)
        logger.debug(f"Job {job_spec.job_id} submitted to queue")
    
    def _scheduler(self):
        """Simple FIFO scheduler with resource matching"""
        while True:
            if self.job_queue:
                job = self.job_queue[0]
                
                # Find suitable node
                suitable_node = None
                for node_id, node in self.nodes.items():
                    # Check if node has enough available CPU cores and memory
                    available_cpu_cores = node.cpu_resource.capacity - node.cpu_resource.count
                    if (available_cpu_cores >= job.cpu_cores and 
                        node.memory_resource.level >= job.memory_mb):
                        suitable_node = node
                        break
                
                if suitable_node:
                    # Remove job from queue and schedule it
                    self.job_queue.pop(0)
                    self.env.process(self._execute_job(job, suitable_node))
                    
                    # Record queue statistics
                    self.metrics.queue_statistics.append({
                        'time': self.env.now,
                        'queue_length': len(self.job_queue),
                        'job_scheduled': job.job_id,
                        'node_assigned': suitable_node.spec.node_id
                    })
            
            yield self.env.timeout(10)  # Check every 10 seconds
    
    def _execute_job(self, job_spec: JobSpec, node: SimulatedNode):
        """Execute a job on a specific node"""
        start_time = self.env.now
        
        try:
            result = yield self.env.process(node.execute_job(job_spec))
            
            # Record job completion
            job_metrics = {
                'job_id': job_spec.job_id,
                'job_type': job_spec.job_type,
                'node_id': result['node_id'],
                'start_time': result['start_time'],
                'end_time': result['end_time'],
                'duration': result['actual_duration'],
                'expected_duration': job_spec.duration_seconds,
                'cpu_cores': job_spec.cpu_cores,
                'memory_mb': job_spec.memory_mb,
                'workload_pattern': job_spec.workload_pattern,
                'completion_status': 'completed'
            }
            
            self.metrics.job_metrics.append(job_metrics)
            self.completed_jobs.append(result)
            
            logger.info(f"Job {job_spec.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_spec.job_id} failed: {e}")
            
            # Record failed job
            job_metrics = {
                'job_id': job_spec.job_id,
                'job_type': job_spec.job_type,
                'start_time': start_time,
                'end_time': self.env.now,
                'duration': self.env.now - start_time,
                'completion_status': 'failed',
                'error': str(e)
            }
            
            self.metrics.job_metrics.append(job_metrics)
    
    def run_simulation(self, duration_seconds: int):
        """Run the simulation for specified duration"""
        logger.info(f"Starting simulation for {duration_seconds} seconds")
        
        # Start scheduler
        self.scheduler_process = self.env.process(self._scheduler())
        
        # Start metrics collection
        self.env.process(self._collect_metrics())
        
        # Run simulation
        self.env.run(until=duration_seconds)
        
        logger.info("Simulation completed")
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
    
    def _collect_metrics(self):
        """Collect cluster-wide metrics"""
        while True:
            # Collect node metrics
            for node_id, node in self.nodes.items():
                if node.metrics_history:
                    latest_metrics = node.metrics_history[-1].copy()
                    self.metrics.node_metrics.append(latest_metrics)
            
            # Collect energy consumption
            total_power = sum(
                node.metrics_history[-1]['power_consumption'] 
                for node in self.nodes.values() 
                if node.metrics_history
            )
            
            self.metrics.energy_consumption.append({
                'time': self.env.now,
                'total_power_watts': total_power,
                'energy_wh': total_power / 60.0  # Assuming 1-minute intervals
            })
            
            yield self.env.timeout(60)  # Collect every minute
    
    def _calculate_performance_metrics(self):
        """Calculate overall performance metrics"""
        if not self.metrics.job_metrics:
            return
        
        completed_jobs = [j for j in self.metrics.job_metrics if j['completion_status'] == 'completed']
        
        if completed_jobs:
            durations = [j['duration'] for j in completed_jobs]
            expected_durations = [j['expected_duration'] for j in completed_jobs]
            
            self.metrics.performance_metrics = {
                'total_jobs_submitted': len(self.metrics.job_metrics),
                'total_jobs_completed': len(completed_jobs),
                'completion_rate': len(completed_jobs) / len(self.metrics.job_metrics),
                'average_job_duration': np.mean(durations),
                'average_expected_duration': np.mean(expected_durations),
                'average_slowdown': np.mean([d/e for d, e in zip(durations, expected_durations)]),
                'total_energy_consumed': sum(e['energy_wh'] for e in self.metrics.energy_consumption),
                'average_power_consumption': np.mean([e['total_power_watts'] for e in self.metrics.energy_consumption]),
                'simulation_duration': self.env.now
            }
    
    def export_results(self, output_dir: str):
        """Export simulation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export job metrics
        if self.metrics.job_metrics:
            job_df = pd.DataFrame(self.metrics.job_metrics)
            job_df.to_csv(output_path / 'simulated_job_metrics.csv', index=False)
            job_df.to_parquet(output_path / 'simulated_job_metrics.parquet')
        
        # Export node metrics
        if self.metrics.node_metrics:
            node_df = pd.DataFrame(self.metrics.node_metrics)
            node_df.to_csv(output_path / 'simulated_node_metrics.csv', index=False)
            node_df.to_parquet(output_path / 'simulated_node_metrics.parquet')
        
        # Export energy consumption
        if self.metrics.energy_consumption:
            energy_df = pd.DataFrame(self.metrics.energy_consumption)
            energy_df.to_csv(output_path / 'simulated_energy_consumption.csv', index=False)
            energy_df.to_parquet(output_path / 'simulated_energy_consumption.parquet')
        
        # Export performance metrics
        with open(output_path / 'performance_metrics.json', 'w') as f:
            json.dump(self.metrics.performance_metrics, f, indent=2)
        
        # Export queue statistics
        if self.metrics.queue_statistics:
            queue_df = pd.DataFrame(self.metrics.queue_statistics)
            queue_df.to_csv(output_path / 'queue_statistics.csv', index=False)
        
        logger.info(f"Simulation results exported to {output_path}")
    
    def get_summary(self) -> Dict:
        """Get simulation summary"""
        return {
            'simulation_config': {
                'duration': self.env.now,
                'nodes': len(self.nodes),
                'total_cores': sum(node.spec.cpu_cores for node in self.nodes.values()),
                'total_memory_mb': sum(node.spec.memory_mb for node in self.nodes.values())
            },
            'job_statistics': {
                'total_submitted': len(self.metrics.job_metrics),
                'total_completed': len([j for j in self.metrics.job_metrics if j['completion_status'] == 'completed']),
                'jobs_in_queue': len(self.job_queue)
            },
            'performance_metrics': self.metrics.performance_metrics,
            'energy_summary': {
                'total_energy_wh': sum(e['energy_wh'] for e in self.metrics.energy_consumption),
                'average_power_w': np.mean([e['total_power_watts'] for e in self.metrics.energy_consumption]) if self.metrics.energy_consumption else 0
            }
        }