#!/usr/bin/env python3
"""
Advanced Performance Benchmarking Module for HPC Energy Model

Provides comprehensive performance analysis, profiling, and optimization
recommendations for HPC workloads with detailed energy efficiency metrics.

Author: HPC Energy Model Project
License: MIT
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import threading
import queue
import subprocess
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import pstats
import io
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics data structure"""
    timestamp: str
    duration_ms: float
    cpu_usage_avg: float
    cpu_usage_max: float
    cpu_usage_std: float
    memory_usage_avg: float
    memory_usage_max: float
    memory_usage_std: float
    disk_read_mbps: float
    disk_write_mbps: float
    disk_iops_read: float
    disk_iops_write: float
    network_recv_mbps: float
    network_sent_mbps: float
    power_consumption_w: float
    energy_consumption_wh: float
    thermal_cpu_avg: float
    thermal_cpu_max: float
    thermal_throttling_events: int
    cache_hit_ratio: float
    context_switches: int
    page_faults: int
    instructions_per_cycle: float
    branch_misses: int
    l1_cache_misses: int
    l2_cache_misses: int
    l3_cache_misses: int
    memory_bandwidth_gbps: float
    numa_remote_accesses: int
    energy_efficiency_score: float
    performance_per_watt: float
    carbon_footprint_g: float

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data structure"""
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    potential_savings_percent: float
    implementation_effort: str  # 'easy', 'medium', 'hard'
    estimated_energy_savings_wh: float
    estimated_cost_savings_usd: float
    technical_details: Dict[str, Any]

class AdvancedPerformanceBenchmarker:
    """
    Advanced performance benchmarking with detailed energy efficiency analysis
    """
    
    def __init__(self, output_dir: str = "performance_analysis", 
                 sampling_interval: float = 0.1,
                 enable_profiling: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sampling_interval = sampling_interval
        self.enable_profiling = enable_profiling
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.benchmark_results: Dict[str, Any] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        
        # Performance baselines
        self.baselines = {
            'cpu_efficiency': 85.0,  # Target CPU efficiency %
            'memory_efficiency': 80.0,  # Target memory efficiency %
            'energy_efficiency': 75.0,  # Target energy efficiency score
            'thermal_threshold': 80.0,  # CPU temperature threshold
            'power_budget_w': 200.0,  # Power budget in watts
        }
        
        # Initialize hardware info
        self.hardware_info = self._get_hardware_info()
        
        logger.info(f"Performance benchmarker initialized with {psutil.cpu_count()} CPU cores")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Collect detailed hardware information"""
        try:
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'cpu_freq_min': psutil.cpu_freq().min if psutil.cpu_freq() else 0,
            }
            
            memory_info = {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            }
            
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total_gb': usage.total / (1024**3),
                        'free_gb': usage.free / (1024**3)
                    })
                except PermissionError:
                    continue
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disks': disk_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not collect complete hardware info: {e}")
            return {}
    
    def _collect_detailed_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_times = psutil.cpu_times()
        cpu_stats = psutil.cpu_stats()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read_mbps = (disk_io.read_bytes / (1024*1024)) if disk_io else 0
        disk_write_mbps = (disk_io.write_bytes / (1024*1024)) if disk_io else 0
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        network_recv_mbps = (network_io.bytes_recv / (1024*1024)) if network_io else 0
        network_sent_mbps = (network_io.bytes_sent / (1024*1024)) if network_io else 0
        
        # Thermal metrics (if available)
        thermal_cpu_avg = 0
        thermal_cpu_max = 0
        thermal_throttling_events = 0
        
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    cpu_temps = []
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            cpu_temps.extend([entry.current for entry in entries if entry.current])
                    
                    if cpu_temps:
                        thermal_cpu_avg = statistics.mean(cpu_temps)
                        thermal_cpu_max = max(cpu_temps)
        except Exception:
            pass
        
        # Power estimation (simplified model)
        base_power = 50  # Base system power
        cpu_power = cpu_percent * 1.5  # Estimated CPU power scaling
        memory_power = (memory.percent / 100) * 20  # Estimated memory power
        estimated_power = base_power + cpu_power + memory_power
        
        # Calculate derived metrics
        energy_efficiency_score = self._calculate_energy_efficiency_score(
            cpu_percent, memory.percent, estimated_power
        )
        
        performance_per_watt = (cpu_percent + (100 - memory.percent)) / estimated_power if estimated_power > 0 else 0
        carbon_footprint_g = estimated_power * 0.0005  # Simplified carbon calculation
        
        return PerformanceMetrics(
            timestamp=timestamp,
            duration_ms=self.sampling_interval * 1000,
            cpu_usage_avg=cpu_percent,
            cpu_usage_max=cpu_percent,  # Will be aggregated later
            cpu_usage_std=0,  # Will be calculated from history
            memory_usage_avg=memory.percent,
            memory_usage_max=memory.percent,
            memory_usage_std=0,
            disk_read_mbps=disk_read_mbps,
            disk_write_mbps=disk_write_mbps,
            disk_iops_read=disk_io.read_count if disk_io else 0,
            disk_iops_write=disk_io.write_count if disk_io else 0,
            network_recv_mbps=network_recv_mbps,
            network_sent_mbps=network_sent_mbps,
            power_consumption_w=estimated_power,
            energy_consumption_wh=estimated_power * (self.sampling_interval / 3600),
            thermal_cpu_avg=thermal_cpu_avg,
            thermal_cpu_max=thermal_cpu_max,
            thermal_throttling_events=thermal_throttling_events,
            cache_hit_ratio=95.0,  # Placeholder - would need hardware counters
            context_switches=cpu_stats.ctx_switches,
            page_faults=0,  # Would need process-specific monitoring
            instructions_per_cycle=2.5,  # Placeholder - would need hardware counters
            branch_misses=0,  # Would need hardware counters
            l1_cache_misses=0,  # Would need hardware counters
            l2_cache_misses=0,  # Would need hardware counters
            l3_cache_misses=0,  # Would need hardware counters
            memory_bandwidth_gbps=10.0,  # Placeholder - would need benchmarking
            numa_remote_accesses=0,  # Would need NUMA-aware monitoring
            energy_efficiency_score=energy_efficiency_score,
            performance_per_watt=performance_per_watt,
            carbon_footprint_g=carbon_footprint_g
        )
    
    def _calculate_energy_efficiency_score(self, cpu_percent: float, 
                                         memory_percent: float, 
                                         power_w: float) -> float:
        """Calculate energy efficiency score (0-100)"""
        # Normalize metrics
        cpu_efficiency = min(cpu_percent / self.baselines['cpu_efficiency'], 1.0) * 100
        memory_efficiency = min(memory_percent / self.baselines['memory_efficiency'], 1.0) * 100
        power_efficiency = min(self.baselines['power_budget_w'] / power_w, 1.0) * 100 if power_w > 0 else 0
        
        # Weighted average
        efficiency_score = (cpu_efficiency * 0.4 + memory_efficiency * 0.3 + power_efficiency * 0.3)
        return min(efficiency_score, 100.0)
    
    def _monitor_metrics(self):
        """Background thread for continuous metrics collection"""
        logger.info("Starting metrics monitoring thread")
        
        while self.monitoring_active:
            try:
                metrics = self._collect_detailed_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(1)
        
        logger.info("Metrics monitoring thread stopped")
    
    @contextmanager
    def benchmark_context(self, benchmark_name: str, 
                         workload_type: str = "unknown",
                         expected_duration: float = None):
        """Context manager for benchmarking code blocks"""
        logger.info(f"Starting benchmark: {benchmark_name}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Setup profiling if enabled
        profiler = None
        if self.enable_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        
        start_time = time.time()
        
        try:
            yield self
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Stop profiling
            if profiler:
                profiler.disable()
            
            # Stop monitoring and collect results
            self.stop_monitoring()
            
            # Analyze results
            results = self.analyze_benchmark_results(
                benchmark_name, workload_type, duration, profiler
            )
            
            # Generate recommendations
            recommendations = self.generate_optimization_recommendations(results)
            
            # Save results
            self.save_benchmark_results(benchmark_name, results, recommendations)
            
            logger.info(f"Benchmark '{benchmark_name}' completed in {duration:.2f}s")
    
    def start_monitoring(self):
        """Start continuous metrics monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.metrics_queue = queue.Queue()
        self.monitor_thread = threading.Thread(target=self._monitor_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop metrics monitoring and collect results"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Collect all metrics from queue
        metrics_list = []
        while not self.metrics_queue.empty():
            try:
                metrics = self.metrics_queue.get_nowait()
                metrics_list.append(metrics)
            except queue.Empty:
                break
        
        self.metrics_history.extend(metrics_list)
        logger.info(f"Collected {len(metrics_list)} metric samples")
    
    def analyze_benchmark_results(self, benchmark_name: str, 
                                workload_type: str,
                                duration: float,
                                profiler: Optional[cProfile.Profile] = None) -> Dict[str, Any]:
        """Analyze collected benchmark metrics"""
        if not self.metrics_history:
            logger.warning("No metrics collected for analysis")
            return {}
        
        # Convert metrics to DataFrame for analysis
        metrics_df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        # Calculate aggregate statistics
        cpu_stats = {
            'avg': metrics_df['cpu_usage_avg'].mean(),
            'max': metrics_df['cpu_usage_avg'].max(),
            'min': metrics_df['cpu_usage_avg'].min(),
            'std': metrics_df['cpu_usage_avg'].std(),
            'p95': metrics_df['cpu_usage_avg'].quantile(0.95),
            'p99': metrics_df['cpu_usage_avg'].quantile(0.99)
        }
        
        memory_stats = {
            'avg': metrics_df['memory_usage_avg'].mean(),
            'max': metrics_df['memory_usage_avg'].max(),
            'min': metrics_df['memory_usage_avg'].min(),
            'std': metrics_df['memory_usage_avg'].std(),
            'p95': metrics_df['memory_usage_avg'].quantile(0.95),
            'p99': metrics_df['memory_usage_avg'].quantile(0.99)
        }
        
        power_stats = {
            'avg': metrics_df['power_consumption_w'].mean(),
            'max': metrics_df['power_consumption_w'].max(),
            'min': metrics_df['power_consumption_w'].min(),
            'total_energy_wh': metrics_df['energy_consumption_wh'].sum(),
            'energy_efficiency_avg': metrics_df['energy_efficiency_score'].mean()
        }
        
        thermal_stats = {
            'cpu_temp_avg': metrics_df['thermal_cpu_avg'].mean(),
            'cpu_temp_max': metrics_df['thermal_cpu_max'].max(),
            'throttling_events': metrics_df['thermal_throttling_events'].sum()
        }
        
        io_stats = {
            'disk_read_total_mb': metrics_df['disk_read_mbps'].sum(),
            'disk_write_total_mb': metrics_df['disk_write_mbps'].sum(),
            'network_recv_total_mb': metrics_df['network_recv_mbps'].sum(),
            'network_sent_total_mb': metrics_df['network_sent_mbps'].sum()
        }
        
        # Performance analysis
        performance_analysis = {
            'throughput_ops_per_sec': len(self.metrics_history) / duration,
            'efficiency_score': metrics_df['energy_efficiency_score'].mean(),
            'performance_per_watt_avg': metrics_df['performance_per_watt'].mean(),
            'carbon_footprint_total_g': metrics_df['carbon_footprint_g'].sum()
        }
        
        # Profiling analysis
        profiling_analysis = {}
        if profiler:
            profiling_analysis = self._analyze_profiling_data(profiler)
        
        # Bottleneck detection
        bottlenecks = self._detect_bottlenecks(metrics_df)
        
        results = {
            'benchmark_info': {
                'name': benchmark_name,
                'workload_type': workload_type,
                'duration_seconds': duration,
                'samples_collected': len(self.metrics_history),
                'sampling_interval': self.sampling_interval,
                'timestamp': datetime.now().isoformat()
            },
            'hardware_info': self.hardware_info,
            'cpu_metrics': cpu_stats,
            'memory_metrics': memory_stats,
            'power_metrics': power_stats,
            'thermal_metrics': thermal_stats,
            'io_metrics': io_stats,
            'performance_analysis': performance_analysis,
            'profiling_analysis': profiling_analysis,
            'bottlenecks': bottlenecks,
            'raw_metrics': [asdict(m) for m in self.metrics_history[-100:]]  # Last 100 samples
        }
        
        return results
    
    def _analyze_profiling_data(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Analyze profiling data for hotspots"""
        try:
            # Capture profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            profiling_output = s.getvalue()
            
            # Parse top functions
            lines = profiling_output.split('\n')
            top_functions = []
            
            for line in lines:
                if line.strip() and not line.startswith(' ') and '(' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            top_functions.append({
                                'ncalls': parts[0],
                                'tottime': float(parts[1]),
                                'percall': float(parts[2]),
                                'cumtime': float(parts[3]),
                                'percall_cum': float(parts[4]),
                                'filename_function': ' '.join(parts[5:])
                            })
                        except (ValueError, IndexError):
                            continue
            
            return {
                'top_functions': top_functions[:10],
                'profiling_output': profiling_output
            }
        
        except Exception as e:
            logger.warning(f"Could not analyze profiling data: {e}")
            return {}
    
    def _detect_bottlenecks(self, metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks from metrics"""
        bottlenecks = []
        
        # CPU bottleneck
        if metrics_df['cpu_usage_avg'].mean() > 90:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high',
                'description': f"High CPU utilization (avg: {metrics_df['cpu_usage_avg'].mean():.1f}%)",
                'recommendation': 'Consider CPU optimization or scaling'
            })
        
        # Memory bottleneck
        if metrics_df['memory_usage_avg'].mean() > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high',
                'description': f"High memory utilization (avg: {metrics_df['memory_usage_avg'].mean():.1f}%)",
                'recommendation': 'Consider memory optimization or increase available memory'
            })
        
        # Thermal throttling
        if metrics_df['thermal_throttling_events'].sum() > 0:
            bottlenecks.append({
                'type': 'thermal',
                'severity': 'critical',
                'description': f"Thermal throttling detected ({metrics_df['thermal_throttling_events'].sum()} events)",
                'recommendation': 'Improve cooling or reduce workload intensity'
            })
        
        # Power efficiency
        if metrics_df['energy_efficiency_score'].mean() < 60:
            bottlenecks.append({
                'type': 'energy_efficiency',
                'severity': 'medium',
                'description': f"Low energy efficiency (avg: {metrics_df['energy_efficiency_score'].mean():.1f}%)",
                'recommendation': 'Optimize workload for better energy efficiency'
            })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        cpu_avg = results['cpu_metrics']['avg']
        memory_avg = results['memory_metrics']['avg']
        energy_efficiency = results['performance_analysis']['efficiency_score']
        power_avg = results['power_metrics']['avg']
        
        # CPU optimization recommendations
        if cpu_avg > 90:
            recommendations.append(OptimizationRecommendation(
                category='cpu',
                priority='high',
                description='CPU utilization is very high. Consider parallelization or algorithm optimization.',
                potential_savings_percent=15.0,
                implementation_effort='medium',
                estimated_energy_savings_wh=power_avg * 0.15 * (results['benchmark_info']['duration_seconds'] / 3600),
                estimated_cost_savings_usd=power_avg * 0.15 * 0.12 / 1000,
                technical_details={
                    'current_cpu_usage': cpu_avg,
                    'target_cpu_usage': 75.0,
                    'optimization_techniques': ['vectorization', 'parallel_processing', 'algorithm_optimization']
                }
            ))
        
        # Memory optimization recommendations
        if memory_avg > 80:
            recommendations.append(OptimizationRecommendation(
                category='memory',
                priority='high',
                description='Memory usage is high. Consider memory optimization or caching strategies.',
                potential_savings_percent=10.0,
                implementation_effort='medium',
                estimated_energy_savings_wh=power_avg * 0.10 * (results['benchmark_info']['duration_seconds'] / 3600),
                estimated_cost_savings_usd=power_avg * 0.10 * 0.12 / 1000,
                technical_details={
                    'current_memory_usage': memory_avg,
                    'target_memory_usage': 70.0,
                    'optimization_techniques': ['memory_pooling', 'data_structure_optimization', 'garbage_collection_tuning']
                }
            ))
        
        # Energy efficiency recommendations
        if energy_efficiency < 70:
            recommendations.append(OptimizationRecommendation(
                category='energy_efficiency',
                priority='medium',
                description='Energy efficiency is below optimal. Consider workload scheduling and power management.',
                potential_savings_percent=20.0,
                implementation_effort='easy',
                estimated_energy_savings_wh=power_avg * 0.20 * (results['benchmark_info']['duration_seconds'] / 3600),
                estimated_cost_savings_usd=power_avg * 0.20 * 0.12 / 1000,
                technical_details={
                    'current_efficiency_score': energy_efficiency,
                    'target_efficiency_score': 80.0,
                    'optimization_techniques': ['dynamic_frequency_scaling', 'workload_consolidation', 'idle_power_management']
                }
            ))
        
        # I/O optimization recommendations
        disk_read_total = results['io_metrics']['disk_read_total_mb']
        disk_write_total = results['io_metrics']['disk_write_total_mb']
        
        if disk_read_total + disk_write_total > 1000:  # High I/O workload
            recommendations.append(OptimizationRecommendation(
                category='io',
                priority='medium',
                description='High I/O activity detected. Consider I/O optimization strategies.',
                potential_savings_percent=12.0,
                implementation_effort='medium',
                estimated_energy_savings_wh=power_avg * 0.12 * (results['benchmark_info']['duration_seconds'] / 3600),
                estimated_cost_savings_usd=power_avg * 0.12 * 0.12 / 1000,
                technical_details={
                    'total_io_mb': disk_read_total + disk_write_total,
                    'optimization_techniques': ['io_batching', 'async_io', 'buffer_optimization', 'ssd_migration']
                }
            ))
        
        return recommendations
    
    def save_benchmark_results(self, benchmark_name: str, 
                             results: Dict[str, Any],
                             recommendations: List[OptimizationRecommendation]):
        """Save benchmark results and recommendations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"{benchmark_name}_{timestamp}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save recommendations
        recommendations_data = [asdict(rec) for rec in recommendations]
        recommendations_file = self.output_dir / f"{benchmark_name}_{timestamp}_recommendations.json"
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        
        # Generate performance report
        self.generate_performance_report(benchmark_name, results, recommendations)
        
        logger.info(f"Benchmark results saved to {results_file}")
        logger.info(f"Recommendations saved to {recommendations_file}")
    
    def generate_performance_report(self, benchmark_name: str,
                                  results: Dict[str, Any],
                                  recommendations: List[OptimizationRecommendation]):
        """Generate comprehensive performance report with visualizations"""
        try:
            # Create visualizations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Performance Analysis Report: {benchmark_name}', fontsize=16)
            
            # Extract metrics for plotting
            raw_metrics = results.get('raw_metrics', [])
            if not raw_metrics:
                logger.warning("No raw metrics available for visualization")
                return
            
            metrics_df = pd.DataFrame(raw_metrics)
            
            # CPU Usage over time
            axes[0, 0].plot(metrics_df['cpu_usage_avg'], label='CPU Usage %')
            axes[0, 0].set_title('CPU Usage Over Time')
            axes[0, 0].set_ylabel('CPU Usage (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Memory Usage over time
            axes[0, 1].plot(metrics_df['memory_usage_avg'], label='Memory Usage %', color='orange')
            axes[0, 1].set_title('Memory Usage Over Time')
            axes[0, 1].set_ylabel('Memory Usage (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Power Consumption over time
            axes[0, 2].plot(metrics_df['power_consumption_w'], label='Power (W)', color='red')
            axes[0, 2].set_title('Power Consumption Over Time')
            axes[0, 2].set_ylabel('Power (W)')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # Energy Efficiency Score
            axes[1, 0].plot(metrics_df['energy_efficiency_score'], label='Efficiency Score', color='green')
            axes[1, 0].set_title('Energy Efficiency Score Over Time')
            axes[1, 0].set_ylabel('Efficiency Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # I/O Activity
            axes[1, 1].plot(metrics_df['disk_read_mbps'], label='Disk Read (MB/s)', alpha=0.7)
            axes[1, 1].plot(metrics_df['disk_write_mbps'], label='Disk Write (MB/s)', alpha=0.7)
            axes[1, 1].set_title('Disk I/O Activity')
            axes[1, 1].set_ylabel('MB/s')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # Performance Summary (Bar chart)
            summary_metrics = [
                results['cpu_metrics']['avg'],
                results['memory_metrics']['avg'],
                results['performance_analysis']['efficiency_score'],
                results['power_metrics']['avg'] / 2  # Scale for visibility
            ]
            summary_labels = ['CPU Avg %', 'Memory Avg %', 'Efficiency Score', 'Power Avg (W/2)']
            
            axes[1, 2].bar(summary_labels, summary_metrics, 
                          color=['blue', 'orange', 'green', 'red'], alpha=0.7)
            axes[1, 2].set_title('Performance Summary')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"{benchmark_name}_{timestamp}_performance_report.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate text report
            report_file = self.output_dir / f"{benchmark_name}_{timestamp}_report.txt"
            with open(report_file, 'w') as f:
                f.write(f"Performance Analysis Report: {benchmark_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("BENCHMARK SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Duration: {results['benchmark_info']['duration_seconds']:.2f} seconds\n")
                f.write(f"Workload Type: {results['benchmark_info']['workload_type']}\n")
                f.write(f"Samples Collected: {results['benchmark_info']['samples_collected']}\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"CPU Usage (avg/max): {results['cpu_metrics']['avg']:.1f}% / {results['cpu_metrics']['max']:.1f}%\n")
                f.write(f"Memory Usage (avg/max): {results['memory_metrics']['avg']:.1f}% / {results['memory_metrics']['max']:.1f}%\n")
                f.write(f"Power Consumption (avg): {results['power_metrics']['avg']:.1f} W\n")
                f.write(f"Total Energy Consumed: {results['power_metrics']['total_energy_wh']:.2f} Wh\n")
                f.write(f"Energy Efficiency Score: {results['performance_analysis']['efficiency_score']:.1f}%\n")
                f.write(f"Performance per Watt: {results['performance_analysis']['performance_per_watt_avg']:.2f}\n\n")
                
                f.write("OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec.category.upper()} - {rec.priority.upper()} PRIORITY\n")
                    f.write(f"   Description: {rec.description}\n")
                    f.write(f"   Potential Savings: {rec.potential_savings_percent:.1f}%\n")
                    f.write(f"   Implementation Effort: {rec.implementation_effort}\n")
                    f.write(f"   Estimated Energy Savings: {rec.estimated_energy_savings_wh:.2f} Wh\n")
                    f.write(f"   Estimated Cost Savings: ${rec.estimated_cost_savings_usd:.4f}\n\n")
            
            logger.info(f"Performance report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
    
    def compare_benchmarks(self, benchmark_files: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark results"""
        comparison_data = []
        
        for file_path in benchmark_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    comparison_data.append(data)
            except Exception as e:
                logger.error(f"Error loading benchmark file {file_path}: {e}")
        
        if len(comparison_data) < 2:
            logger.warning("Need at least 2 benchmark files for comparison")
            return {}
        
        # Extract key metrics for comparison
        comparison_metrics = []
        for data in comparison_data:
            metrics = {
                'name': data['benchmark_info']['name'],
                'duration': data['benchmark_info']['duration_seconds'],
                'cpu_avg': data['cpu_metrics']['avg'],
                'memory_avg': data['memory_metrics']['avg'],
                'power_avg': data['power_metrics']['avg'],
                'energy_total': data['power_metrics']['total_energy_wh'],
                'efficiency_score': data['performance_analysis']['efficiency_score'],
                'performance_per_watt': data['performance_analysis']['performance_per_watt_avg']
            }
            comparison_metrics.append(metrics)
        
        # Generate comparison report
        comparison_df = pd.DataFrame(comparison_metrics)
        
        # Find best and worst performers
        best_efficiency = comparison_df.loc[comparison_df['efficiency_score'].idxmax()]
        worst_efficiency = comparison_df.loc[comparison_df['efficiency_score'].idxmin()]
        
        best_performance_per_watt = comparison_df.loc[comparison_df['performance_per_watt'].idxmax()]
        lowest_energy = comparison_df.loc[comparison_df['energy_total'].idxmin()]
        
        comparison_results = {
            'comparison_summary': {
                'benchmarks_compared': len(comparison_data),
                'best_efficiency': {
                    'name': best_efficiency['name'],
                    'score': best_efficiency['efficiency_score']
                },
                'worst_efficiency': {
                    'name': worst_efficiency['name'],
                    'score': worst_efficiency['efficiency_score']
                },
                'best_performance_per_watt': {
                    'name': best_performance_per_watt['name'],
                    'score': best_performance_per_watt['performance_per_watt']
                },
                'lowest_energy_consumption': {
                    'name': lowest_energy['name'],
                    'energy_wh': lowest_energy['energy_total']
                }
            },
            'detailed_comparison': comparison_metrics,
            'statistics': {
                'avg_efficiency_score': comparison_df['efficiency_score'].mean(),
                'avg_energy_consumption': comparison_df['energy_total'].mean(),
                'avg_performance_per_watt': comparison_df['performance_per_watt'].mean()
            }
        }
        
        return comparison_results
    
    def reset_metrics(self):
        """Reset collected metrics"""
        self.metrics_history.clear()
        logger.info("Metrics history reset")

# Example usage and testing functions
def example_cpu_intensive_benchmark():
    """Example CPU-intensive benchmark"""
    # Simulate CPU-intensive work
    result = 0
    for i in range(1000000):
        result += i ** 2
    return result

def example_memory_intensive_benchmark():
    """Example memory-intensive benchmark"""
    # Simulate memory-intensive work
    data = []
    for i in range(100000):
        data.append([j for j in range(100)])
    return len(data)

def example_io_intensive_benchmark():
    """Example I/O-intensive benchmark"""
    # Simulate I/O-intensive work
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Write data
        for i in range(10000):
            f.write(f"Line {i}: {'x' * 100}\n".encode())
        temp_file = f.name
    
    # Read data back
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    
    # Cleanup
    os.unlink(temp_file)
    return len(lines)

if __name__ == "__main__":
    # Example usage
    benchmarker = AdvancedPerformanceBenchmarker()
    
    # CPU-intensive benchmark
    with benchmarker.benchmark_context("cpu_intensive_test", "cpu") as bench:
        result = example_cpu_intensive_benchmark()
        print(f"CPU benchmark result: {result}")
    
    # Memory-intensive benchmark
    with benchmarker.benchmark_context("memory_intensive_test", "memory") as bench:
        result = example_memory_intensive_benchmark()
        print(f"Memory benchmark result: {result}")
    
    # I/O-intensive benchmark
    with benchmarker.benchmark_context("io_intensive_test", "io") as bench:
        result = example_io_intensive_benchmark()
        print(f"I/O benchmark result: {result}")
    
    print("\nBenchmarking completed. Check the 'performance_analysis' directory for detailed reports.")