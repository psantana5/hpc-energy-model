#!/usr/bin/env python3
"""
CPU-Intensive Workload Generator for HPC Energy Model

Generates CPU-bound workloads with configurable intensity and duration
to simulate compute-heavy HPC jobs and measure their thermal/energy impact.

Author: HPC Energy Model Project
License: MIT
"""

import time
import math
import multiprocessing
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any
import psutil

class CPUBenchmark:
    """
    CPU-intensive benchmark that generates configurable computational load
    """
    
    def __init__(self, duration: int = 300, intensity: float = 1.0, 
                 cores: int = None, workload_type: str = "mixed"):
        self.duration = duration
        self.intensity = intensity
        self.cores = cores or multiprocessing.cpu_count()
        self.workload_type = workload_type
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def prime_calculation(self, n: int) -> int:
        """CPU-intensive prime number calculation"""
        count = 0
        num = 2
        while count < n:
            is_prime = True
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                count += 1
            num += 1
        return num - 1
    
    def matrix_multiplication(self, size: int = 500) -> float:
        """CPU-intensive matrix operations"""
        import random
        
        # Generate random matrices
        matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
        matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
        
        # Multiply matrices
        result = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        
        # Return sum of all elements as verification
        return sum(sum(row) for row in result)
    
    def floating_point_operations(self, iterations: int = 1000000) -> float:
        """CPU-intensive floating point calculations"""
        result = 0.0
        for i in range(iterations):
            result += math.sin(i) * math.cos(i) * math.sqrt(i + 1)
            result += math.log(i + 1) * math.exp(i / 1000000)
        return result
    
    def fibonacci_calculation(self, n: int = 40) -> int:
        """CPU-intensive recursive Fibonacci calculation"""
        if n <= 1:
            return n
        return self.fibonacci_calculation(n - 1) + self.fibonacci_calculation(n - 2)
    
    def worker_process(self, worker_id: int, duration: int, intensity: float) -> Dict[str, Any]:
        """Worker process that runs CPU-intensive tasks"""
        start_time = time.time()
        end_time = start_time + duration
        
        operations_count = 0
        total_result = 0
        
        while time.time() < end_time:
            if self.workload_type == "prime":
                result = self.prime_calculation(int(100 * intensity))
            elif self.workload_type == "matrix":
                result = self.matrix_multiplication(int(200 * intensity))
            elif self.workload_type == "floating":
                result = self.floating_point_operations(int(500000 * intensity))
            elif self.workload_type == "fibonacci":
                result = self.fibonacci_calculation(int(30 + 5 * intensity))
            else:  # mixed workload
                if operations_count % 4 == 0:
                    result = self.prime_calculation(int(50 * intensity))
                elif operations_count % 4 == 1:
                    result = self.matrix_multiplication(int(100 * intensity))
                elif operations_count % 4 == 2:
                    result = self.floating_point_operations(int(250000 * intensity))
                else:
                    result = self.fibonacci_calculation(int(25 + 3 * intensity))
            
            total_result += hash(str(result)) % 1000000  # Prevent optimization
            operations_count += 1
            
            # Control intensity by adding small delays
            if intensity < 1.0:
                time.sleep((1.0 - intensity) * 0.001)
        
        actual_duration = time.time() - start_time
        
        return {
            'worker_id': worker_id,
            'operations_count': operations_count,
            'total_result': total_result,
            'actual_duration': actual_duration,
            'ops_per_second': operations_count / actual_duration
        }
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics during benchmark"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            
            # Try to get temperature if available
            temperature = None
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get first available temperature sensor
                        for sensor_name, sensor_list in temps.items():
                            if sensor_list:
                                temperature = sensor_list[0].current
                                break
            except Exception:
                pass
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'cpu_freq_mhz': cpu_freq.current if cpu_freq else None,
                'temperature_celsius': temperature,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the CPU benchmark with multiple processes"""
        print(f"Starting CPU benchmark:")
        print(f"  Duration: {self.duration} seconds")
        print(f"  Intensity: {self.intensity}")
        print(f"  Cores: {self.cores}")
        print(f"  Workload type: {self.workload_type}")
        
        self.start_time = time.time()
        
        # Collect initial system metrics
        initial_metrics = self.collect_system_metrics()
        
        # Start worker processes
        with multiprocessing.Pool(processes=self.cores) as pool:
            # Create tasks for each worker
            tasks = [
                pool.apply_async(
                    self.worker_process, 
                    (i, self.duration, self.intensity)
                ) for i in range(self.cores)
            ]
            
            # Monitor system metrics during execution
            metrics_history = [initial_metrics]
            monitor_start = time.time()
            
            while time.time() - monitor_start < self.duration:
                time.sleep(5)  # Collect metrics every 5 seconds
                metrics_history.append(self.collect_system_metrics())
            
            # Wait for all workers to complete
            worker_results = [task.get() for task in tasks]
        
        self.end_time = time.time()
        actual_duration = self.end_time - self.start_time
        
        # Collect final metrics
        final_metrics = self.collect_system_metrics()
        metrics_history.append(final_metrics)
        
        # Calculate aggregate results
        total_operations = sum(result['operations_count'] for result in worker_results)
        avg_ops_per_second = sum(result['ops_per_second'] for result in worker_results)
        
        # Calculate average metrics
        valid_metrics = [m for m in metrics_history if 'error' not in m]
        if valid_metrics:
            avg_cpu = sum(m.get('cpu_percent', 0) for m in valid_metrics) / len(valid_metrics)
            avg_memory = sum(m.get('memory_percent', 0) for m in valid_metrics) / len(valid_metrics)
            avg_temp = None
            temp_readings = [m.get('temperature_celsius') for m in valid_metrics if m.get('temperature_celsius')]
            if temp_readings:
                avg_temp = sum(temp_readings) / len(temp_readings)
        else:
            avg_cpu = avg_memory = avg_temp = None
        
        self.metrics = {
            'benchmark_info': {
                'duration_requested': self.duration,
                'duration_actual': actual_duration,
                'intensity': self.intensity,
                'cores_used': self.cores,
                'workload_type': self.workload_type,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat()
            },
            'performance_metrics': {
                'total_operations': total_operations,
                'operations_per_second': avg_ops_per_second,
                'operations_per_core': total_operations / self.cores
            },
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_temperature_celsius': avg_temp,
                'peak_cpu': max((m.get('cpu_percent', 0) for m in valid_metrics), default=0),
                'peak_memory': max((m.get('memory_percent', 0) for m in valid_metrics), default=0)
            },
            'worker_results': worker_results,
            'metrics_history': metrics_history
        }
        
        return self.metrics
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cpu_benchmark_{self.workload_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.metrics:
            print("No metrics available. Run benchmark first.")
            return
        
        print("\n" + "="*60)
        print("CPU BENCHMARK SUMMARY")
        print("="*60)
        
        info = self.metrics['benchmark_info']
        perf = self.metrics['performance_metrics']
        sys_metrics = self.metrics['system_metrics']
        
        print(f"Duration: {info['duration_actual']:.2f}s (requested: {info['duration_requested']}s)")
        print(f"Workload Type: {info['workload_type']}")
        print(f"Intensity: {info['intensity']}")
        print(f"Cores Used: {info['cores_used']}")
        print(f"\nPerformance:")
        print(f"  Total Operations: {perf['total_operations']:,}")
        print(f"  Operations/Second: {perf['operations_per_second']:.2f}")
        print(f"  Operations/Core: {perf['operations_per_core']:.2f}")
        print(f"\nSystem Metrics:")
        print(f"  Average CPU: {sys_metrics['avg_cpu_percent']:.1f}%")
        print(f"  Peak CPU: {sys_metrics['peak_cpu']:.1f}%")
        print(f"  Average Memory: {sys_metrics['avg_memory_percent']:.1f}%")
        print(f"  Peak Memory: {sys_metrics['peak_memory']:.1f}%")
        if sys_metrics['avg_temperature_celsius']:
            print(f"  Average Temperature: {sys_metrics['avg_temperature_celsius']:.1f}°C")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='CPU-Intensive Workload Generator')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Benchmark duration in seconds (default: 300)')
    parser.add_argument('--intensity', type=float, default=1.0, 
                       help='Workload intensity 0.1-1.0 (default: 1.0)')
    parser.add_argument('--cores', type=int, default=None, 
                       help='Number of CPU cores to use (default: all)')
    parser.add_argument('--workload', choices=['prime', 'matrix', 'floating', 'fibonacci', 'mixed'], 
                       default='mixed', help='Type of workload (default: mixed)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output filename for results (default: auto-generated)')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Validate intensity
    if not 0.1 <= args.intensity <= 1.0:
        print("Error: Intensity must be between 0.1 and 1.0")
        return 1
    
    # Create and run benchmark
    benchmark = CPUBenchmark(
        duration=args.duration,
        intensity=args.intensity,
        cores=args.cores,
        workload_type=args.workload
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