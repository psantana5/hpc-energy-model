#!/usr/bin/env python3
"""
Mixed Workload Generator for HPC Energy Model

Generates mixed CPU and I/O workloads to simulate realistic HPC jobs
that combine computation with data processing.

Author: HPC Energy Model Project
License: MIT
"""

import time
import os
import random
import string
import threading
import argparse
import json
import math
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path
import psutil

class MixedBenchmark:
    """
    Mixed workload benchmark that combines CPU and I/O operations
    """
    
    def __init__(self, duration: int = 300, intensity: float = 1.0, 
                 cpu_threads: int = None, io_threads: int = 2, 
                 workload_pattern: str = "alternating", 
                 cpu_io_ratio: float = 0.5, data_size_mb: int = 50,
                 temp_dir: str = None):
        self.duration = duration
        self.intensity = intensity
        self.cpu_threads = cpu_threads or multiprocessing.cpu_count()
        self.io_threads = io_threads
        self.workload_pattern = workload_pattern
        self.cpu_io_ratio = cpu_io_ratio  # 0.0 = all I/O, 1.0 = all CPU
        self.data_size_mb = data_size_mb
        self.temp_dir = Path(temp_dir) if temp_dir else Path("/tmp/hpc_mixed_benchmark")
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.stop_flag = threading.Event()
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def cpu_intensive_task(self, task_type: str = "prime", iterations: int = 1000) -> Dict[str, Any]:
        """CPU-intensive computation task"""
        start_time = time.time()
        operations = 0
        
        try:
            if task_type == "prime":
                # Prime number calculation
                for i in range(2, iterations + 2):
                    if self.stop_flag.is_set():
                        break
                    is_prime = True
                    for j in range(2, int(math.sqrt(i)) + 1):
                        if i % j == 0:
                            is_prime = False
                            break
                    operations += 1
                    
            elif task_type == "matrix":
                # Matrix multiplication
                size = min(100, int(math.sqrt(iterations)))
                matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
                matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
                
                for _ in range(iterations // (size * size)):
                    if self.stop_flag.is_set():
                        break
                    result = [[0 for _ in range(size)] for _ in range(size)]
                    for i in range(size):
                        for j in range(size):
                            for k in range(size):
                                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
                    operations += size * size * size
                    
            elif task_type == "floating_point":
                # Floating point operations
                x = random.random()
                for i in range(iterations):
                    if self.stop_flag.is_set():
                        break
                    x = math.sin(x) * math.cos(x) + math.sqrt(abs(x))
                    x = x - int(x)  # Keep in range [0,1)
                    operations += 4  # sin, cos, sqrt, arithmetic
                    
            else:  # fibonacci
                # Fibonacci calculation
                for i in range(min(iterations, 40)):  # Limit to prevent overflow
                    if self.stop_flag.is_set():
                        break
                    a, b = 0, 1
                    for _ in range(i):
                        a, b = b, a + b
                    operations += i
        
        except Exception as e:
            return {'error': str(e)}
        
        duration = time.time() - start_time
        
        return {
            'task_type': task_type,
            'operations': operations,
            'duration': duration,
            'ops_per_second': operations / duration if duration > 0 else 0
        }
    
    def io_task(self, task_type: str = "sequential", file_size_kb: int = 1024) -> Dict[str, Any]:
        """I/O operation task"""
        start_time = time.time()
        operations = 0
        bytes_processed = 0
        
        filename = self.temp_dir / f"io_task_{threading.current_thread().ident}_{int(time.time())}.tmp"
        
        try:
            if task_type == "sequential":
                # Sequential write then read
                data = ''.join(random.choices(string.ascii_letters, k=file_size_kb * 1024))
                
                # Write
                with open(filename, 'w') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                bytes_processed += len(data)
                operations += 1
                
                # Read
                with open(filename, 'r') as f:
                    read_data = f.read()
                bytes_processed += len(read_data)
                operations += 1
                
            elif task_type == "random":
                # Random access I/O
                data = ''.join(random.choices(string.ascii_letters, k=file_size_kb * 1024))
                
                with open(filename, 'w') as f:
                    f.write(data)
                    f.flush()
                
                # Random reads
                with open(filename, 'r') as f:
                    for _ in range(10):  # 10 random reads
                        if self.stop_flag.is_set():
                            break
                        pos = random.randint(0, max(0, len(data) - 100))
                        f.seek(pos)
                        chunk = f.read(100)
                        bytes_processed += len(chunk)
                        operations += 1
                        
            else:  # append
                # Append operations
                chunk_size = file_size_kb * 1024 // 10
                for i in range(10):
                    if self.stop_flag.is_set():
                        break
                    data = f"Chunk {i}: " + ''.join(random.choices(string.ascii_letters, k=chunk_size))
                    with open(filename, 'a') as f:
                        f.write(data + "\n")
                        f.flush()
                    bytes_processed += len(data) + 1
                    operations += 1
        
        except Exception as e:
            return {'error': str(e)}
        
        finally:
            # Clean up
            try:
                filename.unlink()
            except Exception:
                pass
        
        duration = time.time() - start_time
        
        return {
            'task_type': task_type,
            'operations': operations,
            'bytes_processed': bytes_processed,
            'duration': duration,
            'throughput_kbps': (bytes_processed / 1024) / duration if duration > 0 else 0,
            'ops_per_second': operations / duration if duration > 0 else 0
        }
    
    def mixed_worker_alternating(self, worker_id: int, duration: int) -> Dict[str, Any]:
        """Worker that alternates between CPU and I/O tasks"""
        start_time = time.time()
        end_time = start_time + duration
        
        cpu_results = []
        io_results = []
        cycle_count = 0
        
        cpu_tasks = ["prime", "matrix", "floating_point", "fibonacci"]
        io_tasks = ["sequential", "random", "append"]
        
        while time.time() < end_time and not self.stop_flag.is_set():
            cycle_start = time.time()
            
            # Determine task durations based on ratio
            cycle_duration = min(10, end_time - time.time())  # Max 10 seconds per cycle
            cpu_duration = cycle_duration * self.cpu_io_ratio
            io_duration = cycle_duration * (1 - self.cpu_io_ratio)
            
            # CPU phase
            if cpu_duration > 0.1:  # At least 100ms
                cpu_task = random.choice(cpu_tasks)
                iterations = int(1000 * cpu_duration * self.intensity)
                
                cpu_phase_start = time.time()
                while time.time() - cpu_phase_start < cpu_duration and not self.stop_flag.is_set():
                    result = self.cpu_intensive_task(cpu_task, iterations // 10)
                    if 'error' not in result:
                        cpu_results.append(result)
                    time.sleep(0.01)  # Small break
            
            # I/O phase
            if io_duration > 0.1 and not self.stop_flag.is_set():  # At least 100ms
                io_task_type = random.choice(io_tasks)
                file_size = int(self.data_size_mb * 1024 * self.intensity) // 10  # KB
                
                io_phase_start = time.time()
                while time.time() - io_phase_start < io_duration and not self.stop_flag.is_set():
                    result = self.io_task(io_task_type, file_size)
                    if 'error' not in result:
                        io_results.append(result)
                    time.sleep(0.01)  # Small break
            
            cycle_count += 1
            
            # Control overall intensity
            if self.intensity < 1.0:
                time.sleep((1.0 - self.intensity) * 0.1)
        
        # Aggregate results
        total_cpu_ops = sum(r['operations'] for r in cpu_results)
        total_io_ops = sum(r['operations'] for r in io_results)
        total_bytes = sum(r['bytes_processed'] for r in io_results)
        
        actual_duration = time.time() - start_time
        
        return {
            'worker_id': worker_id,
            'pattern': 'alternating',
            'cycles_completed': cycle_count,
            'cpu_tasks_completed': len(cpu_results),
            'io_tasks_completed': len(io_results),
            'total_cpu_operations': total_cpu_ops,
            'total_io_operations': total_io_ops,
            'total_bytes_processed': total_bytes,
            'duration': actual_duration,
            'cpu_ops_per_second': total_cpu_ops / actual_duration if actual_duration > 0 else 0,
            'io_ops_per_second': total_io_ops / actual_duration if actual_duration > 0 else 0,
            'io_throughput_kbps': (total_bytes / 1024) / actual_duration if actual_duration > 0 else 0
        }
    
    def mixed_worker_concurrent(self, worker_id: int, duration: int) -> Dict[str, Any]:
        """Worker that runs CPU and I/O tasks concurrently"""
        start_time = time.time()
        
        cpu_results = []
        io_results = []
        
        def cpu_thread():
            cpu_tasks = ["prime", "matrix", "floating_point", "fibonacci"]
            while time.time() - start_time < duration and not self.stop_flag.is_set():
                task = random.choice(cpu_tasks)
                iterations = int(500 * self.intensity)
                result = self.cpu_intensive_task(task, iterations)
                if 'error' not in result:
                    cpu_results.append(result)
                time.sleep(0.05)  # Small break
        
        def io_thread():
            io_tasks = ["sequential", "random", "append"]
            while time.time() - start_time < duration and not self.stop_flag.is_set():
                task = random.choice(io_tasks)
                file_size = int(self.data_size_mb * 1024 * self.intensity) // 20  # KB
                result = self.io_task(task, file_size)
                if 'error' not in result:
                    io_results.append(result)
                time.sleep(0.1)  # Small break
        
        # Start concurrent threads
        threads = []
        if self.cpu_io_ratio > 0:
            cpu_t = threading.Thread(target=cpu_thread)
            threads.append(cpu_t)
            cpu_t.start()
        
        if self.cpu_io_ratio < 1:
            io_t = threading.Thread(target=io_thread)
            threads.append(io_t)
            io_t.start()
        
        # Wait for completion
        time.sleep(duration)
        self.stop_flag.set()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # Aggregate results
        total_cpu_ops = sum(r['operations'] for r in cpu_results)
        total_io_ops = sum(r['operations'] for r in io_results)
        total_bytes = sum(r['bytes_processed'] for r in io_results)
        
        actual_duration = time.time() - start_time
        
        return {
            'worker_id': worker_id,
            'pattern': 'concurrent',
            'cpu_tasks_completed': len(cpu_results),
            'io_tasks_completed': len(io_results),
            'total_cpu_operations': total_cpu_ops,
            'total_io_operations': total_io_ops,
            'total_bytes_processed': total_bytes,
            'duration': actual_duration,
            'cpu_ops_per_second': total_cpu_ops / actual_duration if actual_duration > 0 else 0,
            'io_ops_per_second': total_io_ops / actual_duration if actual_duration > 0 else 0,
            'io_throughput_kbps': (total_bytes / 1024) / actual_duration if actual_duration > 0 else 0
        }
    
    def mixed_worker_burst(self, worker_id: int, duration: int) -> Dict[str, Any]:
        """Worker that creates bursts of CPU or I/O activity"""
        start_time = time.time()
        end_time = start_time + duration
        
        cpu_results = []
        io_results = []
        burst_count = 0
        
        while time.time() < end_time and not self.stop_flag.is_set():
            # Determine burst type and duration
            burst_duration = random.uniform(5, 15)  # 5-15 second bursts
            burst_duration = min(burst_duration, end_time - time.time())
            
            if random.random() < self.cpu_io_ratio:
                # CPU burst
                burst_start = time.time()
                while time.time() - burst_start < burst_duration and not self.stop_flag.is_set():
                    task = random.choice(["prime", "matrix", "floating_point"])
                    iterations = int(2000 * self.intensity)
                    result = self.cpu_intensive_task(task, iterations)
                    if 'error' not in result:
                        cpu_results.append(result)
            else:
                # I/O burst
                burst_start = time.time()
                while time.time() - burst_start < burst_duration and not self.stop_flag.is_set():
                    task = random.choice(["sequential", "random", "append"])
                    file_size = int(self.data_size_mb * 1024 * self.intensity) // 5  # KB
                    result = self.io_task(task, file_size)
                    if 'error' not in result:
                        io_results.append(result)
                    time.sleep(0.05)
            
            burst_count += 1
            
            # Brief pause between bursts
            time.sleep(random.uniform(1, 3))
        
        # Aggregate results
        total_cpu_ops = sum(r['operations'] for r in cpu_results)
        total_io_ops = sum(r['operations'] for r in io_results)
        total_bytes = sum(r['bytes_processed'] for r in io_results)
        
        actual_duration = time.time() - start_time
        
        return {
            'worker_id': worker_id,
            'pattern': 'burst',
            'bursts_completed': burst_count,
            'cpu_tasks_completed': len(cpu_results),
            'io_tasks_completed': len(io_results),
            'total_cpu_operations': total_cpu_ops,
            'total_io_operations': total_io_ops,
            'total_bytes_processed': total_bytes,
            'duration': actual_duration,
            'cpu_ops_per_second': total_cpu_ops / actual_duration if actual_duration > 0 else 0,
            'io_ops_per_second': total_io_ops / actual_duration if actual_duration > 0 else 0,
            'io_throughput_kbps': (total_bytes / 1024) / actual_duration if actual_duration > 0 else 0
        }
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            memory_info = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage(str(self.temp_dir))
            
            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except Exception:
                network_stats = {}
            
            # Process count and load
            process_count = len(psutil.pids())
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                load_avg = [0, 0, 0]  # Windows doesn't have load average
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3),
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'disk_read_count': disk_io.read_count if disk_io else 0,
                'disk_write_count': disk_io.write_count if disk_io else 0,
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'disk_free_gb': disk_usage.free / (1024**3),
                'process_count': process_count,
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2],
                **network_stats
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the mixed workload benchmark"""
        print(f"Starting Mixed Workload Benchmark:")
        print(f"  Duration: {self.duration} seconds")
        print(f"  Intensity: {self.intensity}")
        print(f"  CPU Threads: {self.cpu_threads}")
        print(f"  I/O Threads: {self.io_threads}")
        print(f"  Workload Pattern: {self.workload_pattern}")
        print(f"  CPU/I/O Ratio: {self.cpu_io_ratio}")
        print(f"  Data Size: {self.data_size_mb} MB")
        print(f"  Temp Directory: {self.temp_dir}")
        
        self.start_time = time.time()
        
        # Collect initial metrics
        initial_metrics = self.collect_system_metrics()
        
        # Start worker threads
        total_workers = max(self.cpu_threads, self.io_threads)
        threads = []
        
        for i in range(total_workers):
            if self.workload_pattern == "alternating":
                worker_func = self.mixed_worker_alternating
            elif self.workload_pattern == "concurrent":
                worker_func = self.mixed_worker_concurrent
            else:  # burst
                worker_func = self.mixed_worker_burst
            
            thread = threading.Thread(
                target=lambda wid=i: setattr(self, f'result_{wid}', 
                                            worker_func(wid, self.duration))
            )
            threads.append(thread)
            thread.start()
        
        # Monitor system metrics
        metrics_history = [initial_metrics]
        monitor_start = time.time()
        
        while time.time() - monitor_start < self.duration:
            time.sleep(5)  # Collect every 5 seconds
            metrics_history.append(self.collect_system_metrics())
        
        # Wait for threads to complete
        self.stop_flag.set()
        for thread in threads:
            thread.join(timeout=10)
        
        self.end_time = time.time()
        actual_duration = self.end_time - self.start_time
        
        # Collect final metrics
        final_metrics = self.collect_system_metrics()
        metrics_history.append(final_metrics)
        
        # Gather results
        worker_results = []
        for i in range(total_workers):
            result = getattr(self, f'result_{i}', {'error': 'Worker did not complete'})
            worker_results.append(result)
        
        # Calculate aggregates
        successful_workers = [r for r in worker_results if 'error' not in r]
        if successful_workers:
            total_cpu_ops = sum(r.get('total_cpu_operations', 0) for r in successful_workers)
            total_io_ops = sum(r.get('total_io_operations', 0) for r in successful_workers)
            total_bytes = sum(r.get('total_bytes_processed', 0) for r in successful_workers)
            total_cpu_tasks = sum(r.get('cpu_tasks_completed', 0) for r in successful_workers)
            total_io_tasks = sum(r.get('io_tasks_completed', 0) for r in successful_workers)
        else:
            total_cpu_ops = total_io_ops = total_bytes = total_cpu_tasks = total_io_tasks = 0
        
        # Calculate system metrics deltas
        valid_metrics = [m for m in metrics_history if 'error' not in m]
        if valid_metrics and len(valid_metrics) > 1:
            start_sys = valid_metrics[0]
            end_sys = valid_metrics[-1]
            
            disk_read_delta = end_sys.get('disk_read_bytes', 0) - start_sys.get('disk_read_bytes', 0)
            disk_write_delta = end_sys.get('disk_write_bytes', 0) - start_sys.get('disk_write_bytes', 0)
            
            avg_cpu = sum(m.get('cpu_percent', 0) for m in valid_metrics) / len(valid_metrics)
            avg_memory = sum(m.get('memory_percent', 0) for m in valid_metrics) / len(valid_metrics)
            avg_freq = sum(m.get('cpu_freq_current', 0) for m in valid_metrics) / len(valid_metrics)
        else:
            disk_read_delta = disk_write_delta = avg_cpu = avg_memory = avg_freq = 0
        
        self.metrics = {
            'benchmark_info': {
                'duration_requested': self.duration,
                'duration_actual': actual_duration,
                'intensity': self.intensity,
                'cpu_threads': self.cpu_threads,
                'io_threads': self.io_threads,
                'total_workers': total_workers,
                'workload_pattern': self.workload_pattern,
                'cpu_io_ratio': self.cpu_io_ratio,
                'data_size_mb': self.data_size_mb,
                'temp_directory': str(self.temp_dir),
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat()
            },
            'performance_metrics': {
                'total_cpu_operations': total_cpu_ops,
                'total_io_operations': total_io_ops,
                'total_bytes_processed': total_bytes,
                'total_cpu_tasks': total_cpu_tasks,
                'total_io_tasks': total_io_tasks,
                'cpu_ops_per_second': total_cpu_ops / actual_duration if actual_duration > 0 else 0,
                'io_ops_per_second': total_io_ops / actual_duration if actual_duration > 0 else 0,
                'io_throughput_mbps': (total_bytes / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0,
                'avg_cpu_ops_per_worker': total_cpu_ops / len(successful_workers) if successful_workers else 0,
                'avg_io_ops_per_worker': total_io_ops / len(successful_workers) if successful_workers else 0
            },
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_cpu_frequency': avg_freq,
                'total_disk_read_mb': disk_read_delta / (1024 * 1024),
                'total_disk_write_mb': disk_write_delta / (1024 * 1024),
                'disk_read_rate_mbps': (disk_read_delta / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0,
                'disk_write_rate_mbps': (disk_write_delta / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0
            },
            'worker_results': worker_results,
            'metrics_history': metrics_history
        }
        
        # Clean up
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
        
        return self.metrics
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mixed_benchmark_{self.workload_pattern}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.metrics:
            print("No metrics available. Run benchmark first.")
            return
        
        print("\n" + "="*70)
        print("MIXED WORKLOAD BENCHMARK SUMMARY")
        print("="*70)
        
        info = self.metrics['benchmark_info']
        perf = self.metrics['performance_metrics']
        sys_metrics = self.metrics['system_metrics']
        
        print(f"Duration: {info['duration_actual']:.2f}s (requested: {info['duration_requested']}s)")
        print(f"Pattern: {info['workload_pattern']}")
        print(f"CPU/I/O Ratio: {info['cpu_io_ratio']}")
        print(f"Intensity: {info['intensity']}")
        print(f"Workers: {info['total_workers']} (CPU: {info['cpu_threads']}, I/O: {info['io_threads']})")
        print(f"\nPerformance:")
        print(f"  CPU Operations: {perf['total_cpu_operations']:,} ({perf['cpu_ops_per_second']:.0f} ops/s)")
        print(f"  I/O Operations: {perf['total_io_operations']:,} ({perf['io_ops_per_second']:.0f} ops/s)")
        print(f"  Data Processed: {perf['total_bytes_processed'] / (1024*1024):.2f} MB")
        print(f"  I/O Throughput: {perf['io_throughput_mbps']:.2f} MB/s")
        print(f"  CPU Tasks: {perf['total_cpu_tasks']:,}")
        print(f"  I/O Tasks: {perf['total_io_tasks']:,}")
        print(f"\nSystem Metrics:")
        print(f"  Average CPU: {sys_metrics['avg_cpu_percent']:.1f}%")
        print(f"  Average Memory: {sys_metrics['avg_memory_percent']:.1f}%")
        print(f"  Average CPU Freq: {sys_metrics['avg_cpu_frequency']:.0f} MHz")
        print(f"  Disk Read: {sys_metrics['total_disk_read_mb']:.2f} MB ({sys_metrics['disk_read_rate_mbps']:.2f} MB/s)")
        print(f"  Disk Write: {sys_metrics['total_disk_write_mb']:.2f} MB ({sys_metrics['disk_write_rate_mbps']:.2f} MB/s)")
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Mixed Workload Generator')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Benchmark duration in seconds (default: 300)')
    parser.add_argument('--intensity', type=float, default=1.0, 
                       help='Workload intensity 0.1-1.0 (default: 1.0)')
    parser.add_argument('--cpu-threads', type=int, default=None, 
                       help='Number of CPU threads (default: CPU count)')
    parser.add_argument('--io-threads', type=int, default=2, 
                       help='Number of I/O threads (default: 2)')
    parser.add_argument('--pattern', 
                       choices=['alternating', 'concurrent', 'burst'], 
                       default='alternating', 
                       help='Workload pattern (default: alternating)')
    parser.add_argument('--cpu-io-ratio', type=float, default=0.5, 
                       help='CPU to I/O ratio 0.0-1.0 (default: 0.5)')
    parser.add_argument('--data-size', type=int, default=50, 
                       help='Data size for I/O operations in MB (default: 50)')
    parser.add_argument('--temp-dir', type=str, default=None, 
                       help='Temporary directory (default: /tmp/hpc_mixed_benchmark)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output filename for results (default: auto-generated)')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 0.1 <= args.intensity <= 1.0:
        print("Error: Intensity must be between 0.1 and 1.0")
        return 1
    
    if not 0.0 <= args.cpu_io_ratio <= 1.0:
        print("Error: CPU/I/O ratio must be between 0.0 and 1.0")
        return 1
    
    # Create and run benchmark
    benchmark = MixedBenchmark(
        duration=args.duration,
        intensity=args.intensity,
        cpu_threads=args.cpu_threads,
        io_threads=args.io_threads,
        workload_pattern=args.pattern,
        cpu_io_ratio=args.cpu_io_ratio,
        data_size_mb=args.data_size,
        temp_dir=args.temp_dir
    )
    
    try:
        results = benchmark.run_benchmark()
        
        if not args.quiet:
            benchmark.print_summary()
        
        # Save results
        output_file = benchmark.save_results(args.output)
        
        if not args.quiet:
            print(f"\nBenchmark completed successfully!")
            print(f"Results saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1

if __name__ == '__main__':
    exit(main())