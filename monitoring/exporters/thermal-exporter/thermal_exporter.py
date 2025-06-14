#!/usr/bin/env python3
"""
Thermal Exporter for HPC Energy Model

Custom Prometheus exporter that collects thermal metrics from CPU sensors,
GPU temperatures, and system thermal zones.

Author: HPC Energy Model Project
License: MIT
"""

import time
import os
import glob
import logging
from typing import Dict, List, Optional
from prometheus_client import start_http_server, Gauge, Info
from prometheus_client.core import CollectorRegistry
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThermalCollector:
    """
    Collects thermal metrics from various system sources:
    - CPU core temperatures
    - GPU temperatures (if available)
    - System thermal zones
    - Fan speeds
    - Power consumption estimates
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.cpu_temp_gauge = Gauge(
            'hpc_cpu_temperature_celsius',
            'CPU core temperature in Celsius',
            ['core', 'socket'],
            registry=self.registry
        )
        
        self.gpu_temp_gauge = Gauge(
            'hpc_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.thermal_zone_gauge = Gauge(
            'hpc_thermal_zone_temperature_celsius',
            'System thermal zone temperature',
            ['zone_type', 'zone_id'],
            registry=self.registry
        )
        
        self.fan_speed_gauge = Gauge(
            'hpc_fan_speed_rpm',
            'Fan speed in RPM',
            ['fan_id', 'fan_label'],
            registry=self.registry
        )
        
        self.power_consumption_gauge = Gauge(
            'hpc_power_consumption_watts',
            'Estimated power consumption in Watts',
            ['component'],
            registry=self.registry
        )
        
        self.thermal_throttling_gauge = Gauge(
            'hpc_thermal_throttling_active',
            'Thermal throttling status (1=active, 0=inactive)',
            ['core'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'hpc_thermal_system_info',
            'System thermal monitoring information',
            registry=self.registry
        )
        
        self._update_system_info()
    
    def _update_system_info(self):
        """Update system information metrics"""
        try:
            info_dict = {
                'hostname': os.uname().nodename,
                'cpu_count': str(psutil.cpu_count()),
                'cpu_count_physical': str(psutil.cpu_count(logical=False)),
                'thermal_zones_available': str(len(self._get_thermal_zones())),
            }
            self.system_info.info(info_dict)
        except Exception as e:
            logger.error(f"Error updating system info: {e}")
    
    def _get_thermal_zones(self) -> List[str]:
        """Get available thermal zones from /sys/class/thermal"""
        try:
            thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*')
            return sorted(thermal_zones)
        except Exception as e:
            logger.error(f"Error reading thermal zones: {e}")
            return []
    
    def _read_thermal_zone_temp(self, zone_path: str) -> Optional[float]:
        """Read temperature from a thermal zone"""
        try:
            temp_path = os.path.join(zone_path, 'temp')
            if os.path.exists(temp_path):
                with open(temp_path, 'r') as f:
                    # Temperature is in millidegrees Celsius
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
        except Exception as e:
            logger.debug(f"Error reading temperature from {zone_path}: {e}")
        return None
    
    def _get_thermal_zone_type(self, zone_path: str) -> str:
        """Get the type of a thermal zone"""
        try:
            type_path = os.path.join(zone_path, 'type')
            if os.path.exists(type_path):
                with open(type_path, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.debug(f"Error reading thermal zone type from {zone_path}: {e}")
        return 'unknown'
    
    def _collect_cpu_temperatures(self):
        """Collect CPU core temperatures"""
        try:
            # Try to get CPU temperatures from psutil
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                
                for sensor_name, sensor_list in temps.items():
                    if 'cpu' in sensor_name.lower() or 'core' in sensor_name.lower():
                        for i, sensor in enumerate(sensor_list):
                            if sensor.current is not None:
                                self.cpu_temp_gauge.labels(
                                    core=f"core_{i}",
                                    socket="0"
                                ).set(sensor.current)
            
            # Fallback: try to read from thermal zones
            thermal_zones = self._get_thermal_zones()
            for zone_path in thermal_zones:
                zone_type = self._get_thermal_zone_type(zone_path)
                if 'cpu' in zone_type.lower() or 'x86_pkg_temp' in zone_type.lower():
                    temp = self._read_thermal_zone_temp(zone_path)
                    if temp is not None:
                        zone_id = os.path.basename(zone_path)
                        self.thermal_zone_gauge.labels(
                            zone_type=zone_type,
                            zone_id=zone_id
                        ).set(temp)
                        
        except Exception as e:
            logger.error(f"Error collecting CPU temperatures: {e}")
    
    def _collect_gpu_temperatures(self):
        """Collect GPU temperatures (if available)"""
        try:
            # Try to use nvidia-ml-py for NVIDIA GPUs
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    self.gpu_temp_gauge.labels(
                        gpu_id=str(i),
                        gpu_name=name
                    ).set(temp)
                    
            except ImportError:
                logger.debug("pynvml not available, skipping NVIDIA GPU temperatures")
            except Exception as e:
                logger.debug(f"Error collecting NVIDIA GPU temperatures: {e}")
                
        except Exception as e:
            logger.error(f"Error in GPU temperature collection: {e}")
    
    def _collect_fan_speeds(self):
        """Collect fan speeds"""
        try:
            if hasattr(psutil, 'sensors_fans'):
                fans = psutil.sensors_fans()
                
                for fan_name, fan_list in fans.items():
                    for i, fan in enumerate(fan_list):
                        if fan.current is not None:
                            self.fan_speed_gauge.labels(
                                fan_id=f"{fan_name}_{i}",
                                fan_label=fan.label or f"fan_{i}"
                            ).set(fan.current)
                            
        except Exception as e:
            logger.error(f"Error collecting fan speeds: {e}")
    
    def _estimate_power_consumption(self):
        """Estimate power consumption based on CPU usage and temperature"""
        try:
            # Simple power estimation model
            # In a real implementation, this would use actual power sensors
            
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Rough estimation: base power + dynamic power based on usage
            base_power_per_core = 15  # Watts
            max_dynamic_power_per_core = 35  # Watts
            
            estimated_cpu_power = (
                base_power_per_core * cpu_count +
                (max_dynamic_power_per_core * cpu_count * cpu_percent / 100)
            )
            
            self.power_consumption_gauge.labels(component='cpu').set(estimated_cpu_power)
            
            # Memory power estimation
            memory_info = psutil.virtual_memory()
            memory_power = (memory_info.total / (1024**3)) * 3  # ~3W per GB
            self.power_consumption_gauge.labels(component='memory').set(memory_power)
            
        except Exception as e:
            logger.error(f"Error estimating power consumption: {e}")
    
    def _check_thermal_throttling(self):
        """Check for thermal throttling indicators"""
        try:
            # Check CPU frequency scaling as an indicator of thermal throttling
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                # If current frequency is significantly below max, might be throttling
                throttling_ratio = cpu_freq.current / cpu_freq.max if cpu_freq.max > 0 else 1.0
                
                # Consider throttling if frequency is below 80% of max
                is_throttling = 1 if throttling_ratio < 0.8 else 0
                
                self.thermal_throttling_gauge.labels(core='all').set(is_throttling)
                
        except Exception as e:
            logger.error(f"Error checking thermal throttling: {e}")
    
    def collect_metrics(self):
        """Collect all thermal metrics"""
        logger.info("Collecting thermal metrics...")
        
        self._collect_cpu_temperatures()
        self._collect_gpu_temperatures()
        self._collect_fan_speeds()
        self._estimate_power_consumption()
        self._check_thermal_throttling()
        
        logger.info("Thermal metrics collection completed")

def main():
    """Main function to start the thermal exporter"""
    port = int(os.getenv('EXPORTER_PORT', 9200))
    polling_interval = int(os.getenv('POLLING_INTERVAL', 5))
    
    logger.info(f"Starting HPC Thermal Exporter on port {port}")
    logger.info(f"Polling interval: {polling_interval} seconds")
    
    # Create collector
    collector = ThermalCollector()
    
    # Start HTTP server
    start_http_server(port, registry=collector.registry)
    
    logger.info(f"Thermal exporter started successfully on port {port}")
    
    # Main collection loop
    try:
        while True:
            collector.collect_metrics()
            time.sleep(polling_interval)
    except KeyboardInterrupt:
        logger.info("Thermal exporter stopped by user")
    except Exception as e:
        logger.error(f"Thermal exporter error: {e}")
        raise

if __name__ == '__main__':
    main()