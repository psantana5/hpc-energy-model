#!/usr/bin/env python3
"""
I/O-Intensive Workload Generator for HPC Energy Model

Generates I/O-bound workloads with configurable patterns and intensity
to simulate data-intensive HPC jobs and measure their impact on system resources.

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
import shutil
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import psutil

class IOBenchmark:
    """
    I/O-intensive benchmark that generates configurable disk access patterns
    """
    
    def __init__(self, duration: int = 300, intensity: float = 1.0, 
                 threads: int = 4, workload_type: str = "mixed", 
                 data_size_mb: int = 100, temp_dir: str = None):
        self.duration = duration
        self.intensity = intensity
        self.threads = threads
        self.workload_type = workload_type
        self.data_size_mb = data_size_mb
        self.temp_dir = Path(temp_dir) if temp_dir else Path("/tmp/hpc_io_benchmark")
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.stop_flag = threading.Event()
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_random_data(self, size_bytes: int) -> bytes:
        """Generate random data of specified size"""
        return ''.join(random.choices(string.ascii_letters + string.digits, 
                                    k=size_bytes)).encode('utf-8')
    
    def sequential_write_test(self, thread_id: int, file_size_mb: int) -> Dict[str, Any]:
        """Sequential write performance test"""
        filename = self.temp_dir / f"seq_write_{thread_id}.dat"
        chunk_size = 64 * 1024  # 64KB chunks
        total_bytes = file_size_mb * 1024 * 1024
        bytes_written = 0
        operations = 0
        
        start_time = time.time()
        
        try:
            with open(filename, 'wb') as f:
                while bytes_written < total_bytes and not self.stop_flag.is_set():
                    chunk_size_actual = min(chunk_size, total_bytes - bytes_written)
                    data = self.generate_random_data(chunk_size_actual)
                    f.write(data)
                    f.flush()  # Force write to disk
                    os.fsync(f.fileno())  # Ensure data is written
                    
                    bytes_written += chunk_size_actual
                    operations += 1
                    
                    # Control intensity
                    if self.intensity < 1.0:
                        time.sleep((1.0 - self.intensity) * 0.001)
        
        except Exception as e:
            return {'error': str(e), 'thread_id': thread_id}
        
        duration = time.time() - start_time
        
        # Clean up
        try:
            filename.unlink()
        except Exception:
            pass
        
        return {
            'thread_id': thread_id,
            'test_type': 'sequential_write',
            'bytes_written': bytes_written,
            'operations': operations,
            'duration': duration,
            'throughput_mbps': (bytes_written / (1024 * 1024)) / duration if duration > 0 else 0,
            'iops': operations / duration if duration > 0 else 0
        }
    
    def sequential_read_test(self, thread_id: int, file_size_mb: int) -> Dict[str, Any]:
        """Sequential read performance test"""
        filename = self.temp_dir / f"seq_read_{thread_id}.dat"
        chunk_size = 64 * 1024  # 64KB chunks
        total_bytes = file_size_mb * 1024 * 1024
        
        # First, create the file to read
        try:
            with open(filename, 'wb') as f:
                data = self.generate_random_data(total_bytes)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            return {'error': f'Failed to create test file: {e}', 'thread_id': thread_id}
        
        bytes_read = 0
        operations = 0
        start_time = time.time()
        
        try:
            with open(filename, 'rb') as f:
                while not self.stop_flag.is_set():
                    data = f.read(chunk_size)
                    if not data:  # EOF reached, start over
                        f.seek(0)
                        continue
                    
                    bytes_read += len(data)
                    operations += 1
                    
                    # Control intensity
                    if self.intensity < 1.0:
                        time.sleep((1.0 - self.intensity) * 0.001)
        
        except Exception as e:
            return {'error': str(e), 'thread_id': thread_id}
        
        duration = time.time() - start_time
        
        # Clean up
        try:
            filename.unlink()
        except Exception:
            pass
        
        return {
            'thread_id': thread_id,
            'test_type': 'sequential_read',
            'bytes_read': bytes_read,
            'operations': operations,
            'duration': duration,
            'throughput_mbps': (bytes_read / (1024 * 1024)) / duration if duration > 0 else 0,
            'iops': operations / duration if duration > 0 else 0
        }
    
    def random_access_test(self, thread_id: int, file_size_mb: int) -> Dict[str, Any]:
        """Random access I/O test"""
        filename = self.temp_dir / f"random_{thread_id}.dat"
        total_bytes = file_size_mb * 1024 * 1024
        block_size = 4 * 1024  # 4KB blocks
        
        # Create file with random data
        try:
            with open(filename, 'wb') as f:
                data = self.generate_random_data(total_bytes)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            return {'error': f'Failed to create test file: {e}', 'thread_id': thread_id}
        
        operations = 0
        bytes_accessed = 0
        start_time = time.time()
        
        try:
            with open(filename, 'r+b') as f:
                while not self.stop_flag.is_set():
                    # Random seek position
                    max_pos = max(0, total_bytes - block_size)
                    pos = random.randint(0, max_pos)
                    f.seek(pos)
                    
                    # Randomly choose read or write
                    if random.choice([True, False]):
                        # Read operation
                        data = f.read(block_size)
                        bytes_accessed += len(data)
                    else:
                        # Write operation
                        data = self.generate_random_data(block_size)
                        f.write(data)
                        f.flush()
                        bytes_accessed += len(data)
                    
                    operations += 1
                    
                    # Control intensity
                    if self.intensity < 1.0:
                        time.sleep((1.0 - self.intensity) * 0.002)
        
        except Exception as e:
            return {'error': str(e), 'thread_id': thread_id}
        
        duration = time.time() - start_time
        
        # Clean up
        try:
            filename.unlink()
        except Exception:
            pass
        
        return {
            'thread_id': thread_id,
            'test_type': 'random_access',
            'bytes_accessed': bytes_accessed,
            'operations': operations,
            'duration': duration,
            'throughput_mbps': (bytes_accessed / (1024 * 1024)) / duration if duration > 0 else 0,
            'iops': operations / duration if duration > 0 else 0
        }
    
    def file_operations_test(self, thread_id: int) -> Dict[str, Any]:
        """File creation, deletion, and metadata operations test"""
        test_dir = self.temp_dir / f"fileops_{thread_id}"
        test_dir.mkdir(exist_ok=True)
        
        operations = 0
        files_created = 0
        files_deleted = 0
        start_time = time.time()
        
        try:
            while not self.stop_flag.is_set():
                # Create files
                for i in range(10):
                    filename = test_dir / f"file_{operations}_{i}.tmp"
                    try:
                        with open(filename, 'w') as f:
                            f.write(f"Test data {operations}_{i}\n" * 100)
                        files_created += 1
                    except Exception:
                        pass
                
                # List directory
                try:
                    list(test_dir.iterdir())
                except Exception:
                    pass
                
                # Delete some files
                try:
                    existing_files = list(test_dir.glob("*.tmp"))
                    for filename in existing_files[:5]:  # Delete first 5 files
                        filename.unlink()
                        files_deleted += 1
                except Exception:
                    pass
                
                operations += 1
                
                # Control intensity
                if self.intensity < 1.0:
                    time.sleep((1.0 - self.intensity) * 0.01)
        
        except Exception as e:
            return {'error': str(e), 'thread_id': thread_id}
        
        duration = time.time() - start_time
        
        # Clean up
        try:
            shutil.rmtree(test_dir)
        except Exception:
            pass
        
        return {
            'thread_id': thread_id,
            'test_type': 'file_operations',
            'operations': operations,
            'files_created': files_created,
            'files_deleted': files_deleted,
            'duration': duration,
            'ops_per_second': operations / duration if duration > 0 else 0
        }
    
    def worker_thread(self, thread_id: int, test_duration: int) -> Dict[str, Any]:
        """Worker thread that runs I/O tests"""
        start_time = time.time()
        end_time = start_time + test_duration
        
        results = []
        
        while time.time() < end_time and not self.stop_flag.is_set():
            if self.workload_type == "sequential_write":
                result = self.sequential_write_test(thread_id, self.data_size_mb)
            elif self.workload_type == "sequential_read":
                result = self.sequential_read_test(thread_id, self.data_size_mb)
            elif self.workload_type == "random_access":
                result = self.random_access_test(thread_id, self.data_size_mb)
            elif self.workload_type == "file_operations":
                result = self.file_operations_test(thread_id)
            else:  # mixed workload
                test_type = random.choice([
                    "sequential_write", "sequential_read", 
                    "random_access", "file_operations"
                ])
                if test_type == "file_operations":
                    result = self.file_operations_test(thread_id)
                else:
                    if test_type == "sequential_write":
                        result = self.sequential_write_test(thread_id, self.data_size_mb // 4)
                    elif test_type == "sequential_read":
                        result = self.sequential_read_test(thread_id, self.data_size_mb // 4)
                    else:
                        result = self.random_access_test(thread_id, self.data_size_mb // 4)
            
            if 'error' not in result:
                results.append(result)
            
            # Short break between tests
            time.sleep(0.1)
        
        # Aggregate results
        if results:
            total_throughput = sum(r.get('throughput_mbps', 0) for r in results)
            total_iops = sum(r.get('iops', 0) for r in results)
            total_ops = sum(r.get('operations', 0) for r in results)
            
            return {
                'thread_id': thread_id,
                'test_count': len(results),
                'avg_throughput_mbps': total_throughput / len(results),
                'avg_iops': total_iops / len(results),
                'total_operations': total_ops,
                'detailed_results': results
            }
        else:
            return {'thread_id': thread_id, 'error': 'No successful tests completed'}
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system I/O and resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            # Disk I/O statistics
            disk_io = psutil.disk_io_counters()
            
            # Disk usage for temp directory
            disk_usage = psutil.disk_usage(str(self.temp_dir))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'disk_read_count': disk_io.read_count if disk_io else 0,
                'disk_write_count': disk_io.write_count if disk_io else 0,
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'disk_free_gb': disk_usage.free / (1024**3)
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the I/O benchmark with multiple threads"""
        print(f"Starting I/O benchmark:")
        print(f"  Duration: {self.duration} seconds")
        print(f"  Intensity: {self.intensity}")
        print(f"  Threads: {self.threads}")
        print(f"  Workload type: {self.workload_type}")
        print(f"  Data size per operation: {self.data_size_mb} MB")
        print(f"  Temp directory: {self.temp_dir}")
        
        self.start_time = time.time()
        
        # Collect initial system metrics
        initial_metrics = self.collect_system_metrics()
        
        # Start worker threads
        threads = []
        for i in range(self.threads):
            thread = threading.Thread(
                target=lambda tid=i: setattr(self, f'result_{tid}', 
                                            self.worker_thread(tid, self.duration))
            )
            threads.append(thread)
            thread.start()
        
        # Monitor system metrics during execution
        metrics_history = [initial_metrics]
        monitor_start = time.time()
        
        while time.time() - monitor_start < self.duration:
            time.sleep(5)  # Collect metrics every 5 seconds
            metrics_history.append(self.collect_system_metrics())
        
        # Signal threads to stop and wait for completion
        self.stop_flag.set()
        for thread in threads:
            thread.join(timeout=10)  # Wait up to 10 seconds for each thread
        
        self.end_time = time.time()
        actual_duration = self.end_time - self.start_time
        
        # Collect final metrics
        final_metrics = self.collect_system_metrics()
        metrics_history.append(final_metrics)
        
        # Gather thread results
        thread_results = []
        for i in range(self.threads):
            result = getattr(self, f'result_{i}', {'error': 'Thread did not complete'})
            thread_results.append(result)
        
        # Calculate aggregate results
        successful_threads = [r for r in thread_results if 'error' not in r]
        if successful_threads:
            total_throughput = sum(r.get('avg_throughput_mbps', 0) for r in successful_threads)
            total_iops = sum(r.get('avg_iops', 0) for r in successful_threads)
            total_operations = sum(r.get('total_operations', 0) for r in successful_threads)
        else:
            total_throughput = total_iops = total_operations = 0
        
        # Calculate system metrics averages
        valid_metrics = [m for m in metrics_history if 'error' not in m]
        if valid_metrics and len(valid_metrics) > 1:
            # Calculate I/O deltas
            start_io = valid_metrics[0]
            end_io = valid_metrics[-1]
            
            io_read_delta = end_io.get('disk_read_bytes', 0) - start_io.get('disk_read_bytes', 0)
            io_write_delta = end_io.get('disk_write_bytes', 0) - start_io.get('disk_write_bytes', 0)
            
            avg_cpu = sum(m.get('cpu_percent', 0) for m in valid_metrics) / len(valid_metrics)
            avg_memory = sum(m.get('memory_percent', 0) for m in valid_metrics) / len(valid_metrics)
        else:
            io_read_delta = io_write_delta = avg_cpu = avg_memory = 0
        
        self.metrics = {
            'benchmark_info': {
                'duration_requested': self.duration,
                'duration_actual': actual_duration,
                'intensity': self.intensity,
                'threads_used': self.threads,
                'workload_type': self.workload_type,
                'data_size_mb': self.data_size_mb,
                'temp_directory': str(self.temp_dir),
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat()
            },
            'performance_metrics': {
                'total_throughput_mbps': total_throughput,
                'total_iops': total_iops,
                'total_operations': total_operations,
                'avg_throughput_per_thread': total_throughput / len(successful_threads) if successful_threads else 0,
                'avg_iops_per_thread': total_iops / len(successful_threads) if successful_threads else 0
            },
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'total_disk_read_mb': io_read_delta / (1024 * 1024),
                'total_disk_write_mb': io_write_delta / (1024 * 1024),
                'disk_read_rate_mbps': (io_read_delta / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0,
                'disk_write_rate_mbps': (io_write_delta / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0
            },
            'thread_results': thread_results,
            'metrics_history': metrics_history
        }
        
        # Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
        
        return self.metrics
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"io_benchmark_{self.workload_type}_{timestamp}.json"
        
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
        print("I/O BENCHMARK SUMMARY")
        print("="*60)
        
        info = self.metrics['benchmark_info']
        perf = self.metrics['performance_metrics']
        sys_metrics = self.metrics['system_metrics']
        
        print(f"Duration: {info['duration_actual']:.2f}s (requested: {info['duration_requested']}s)")
        print(f"Workload Type: {info['workload_type']}")
        print(f"Intensity: {info['intensity']}")
        print(f"Threads Used: {info['threads_used']}")
        print(f"Data Size per Op: {info['data_size_mb']} MB")
        print(f"\nPerformance:")
        print(f"  Total Throughput: {perf['total_throughput_mbps']:.2f} MB/s")
        print(f"  Total IOPS: {perf['total_iops']:.2f}")
        print(f"  Total Operations: {perf['total_operations']:,}")
        print(f"  Avg Throughput/Thread: {perf['avg_throughput_per_thread']:.2f} MB/s")
        print(f"  Avg IOPS/Thread: {perf['avg_iops_per_thread']:.2f}")
        print(f"\nSystem Metrics:")
        print(f"  Average CPU: {sys_metrics['avg_cpu_percent']:.1f}%")
        print(f"  Average Memory: {sys_metrics['avg_memory_percent']:.1f}%")
        print(f"  Total Disk Read: {sys_metrics['total_disk_read_mb']:.2f} MB")
        print(f"  Total Disk Write: {sys_metrics['total_disk_write_mb']:.2f} MB")
        print(f"  Disk Read Rate: {sys_metrics['disk_read_rate_mbps']:.2f} MB/s")
        print(f"  Disk Write Rate: {sys_metrics['disk_write_rate_mbps']:.2f} MB/s")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='I/O-Intensive Workload Generator')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Benchmark duration in seconds (default: 300)')
    parser.add_argument('--intensity', type=float, default=1.0, 
                       help='Workload intensity 0.1-1.0 (default: 1.0)')
    parser.add_argument('--threads', type=int, default=4, 
                       help='Number of I/O threads (default: 4)')
    parser.add_argument('--workload', 
                       choices=['sequential_write', 'sequential_read', 'random_access', 
                               'file_operations', 'mixed'], 
                       default='mixed', help='Type of I/O workload (default: mixed)')
    parser.add_argument('--data-size', type=int, default=100, 
                       help='Data size per operation in MB (default: 100)')
    parser.add_argument('--temp-dir', type=str, default=None, 
                       help='Temporary directory for test files (default: /tmp/hpc_io_benchmark)')
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
    benchmark = IOBenchmark(
        duration=args.duration,
        intensity=args.intensity,
        threads=args.threads,
        workload_type=args.workload,
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