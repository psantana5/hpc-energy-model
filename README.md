# HPC Energy Model Project

A comprehensive High-Performance Computing (HPC) energy monitoring and prediction system designed for virtualized environments. This project provides real-time energy consumption tracking, predictive modeling, and energy-aware job scheduling for HPC clusters.

## 🎯 Project Overview

This project implements an end-to-end energy monitoring solution for HPC systems, featuring:

- **Real-time Energy Monitoring**: Custom exporters for thermal and job metrics
- **Predictive Energy Modeling**: Machine learning-based energy consumption prediction
- **Energy-Aware Scheduling**: Slurm integration with QoS-based energy optimization
- **Comprehensive Visualization**: Grafana dashboards for energy analysis
- **Data Storage**: TimescaleDB for time-series energy data
- **Alerting System**: Prometheus-based alerts for thermal and energy events

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HPC Nodes     │    │   Monitoring    │    │   Analysis      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Slurm Jobs  │ │───▶│ │ Prometheus  │ │───▶│ │ Grafana     │ │
│ │ Thermal     │ │    │ │ Exporters   │ │    │ │ Dashboards  │ │
│ │ Sensors     │ │    │ │ Alertmanager│ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   TimescaleDB   │              │
         └──────────────│   Time-series   │──────────────┘
                        │   Database      │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │ Energy Prediction│
                        │ API & ML Models │
                        └─────────────────┘
```

## 📋 Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **Linux Environment**: Ubuntu 20.04+ or CentOS 8+ recommended
- **Hardware**: Minimum 8GB RAM, 4 CPU cores, 50GB storage
- **Network**: Access to HPC nodes for monitoring
- **Optional**: GPU support for accelerated workloads

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hpc-energy-model
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env
```

### 3. Deploy the Stack

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Energy Prediction API**: http://localhost:5000
- **TimescaleDB**: localhost:5432 (postgres/password)

## 📁 Project Structure

```
hpc-energy-model/
├── analysis/                    # Data analysis and ML scripts
│   └── data_analysis.py        # Energy data analysis tools
├── api/                        # Energy prediction API
│   ├── energy_prediction_api.py # Flask API for predictions
│   ├── Dockerfile              # API container
│   ├── requirements.txt        # Python dependencies
│   └── entrypoint.sh          # API startup script
├── database/                   # Database initialization
│   └── init.sql               # TimescaleDB schema
├── infrastructure/             # Infrastructure configuration
│   ├── grafana/               # Grafana provisioning
│   │   └── provisioning/      # Dashboards and datasources
│   └── prometheus/            # Prometheus configuration
│       ├── prometheus.yml     # Main config
│       └── rules/             # Alert rules
├── monitoring/                 # Monitoring components
│   ├── exporters/             # Custom Prometheus exporters
│   │   ├── job-exporter/      # Slurm job metrics
│   │   └── thermal-exporter/  # System thermal metrics
│   └── grafana/               # Grafana dashboards
│       └── dashboards/        # Dashboard JSON files
├── slurm/                     # Slurm configuration
│   ├── scripts/               # Job lifecycle scripts
│   │   ├── job_prolog.sh      # Pre-job energy setup
│   │   ├── job_epilog.sh      # Post-job energy analysis
│   │   └── energy_monitor.sh  # Continuous monitoring
│   ├── slurm.conf            # Main Slurm configuration
│   └── qos.conf              # Quality of Service definitions
├── workloads/                 # Benchmark workloads
│   ├── cpu-intensive/         # CPU benchmark scripts
│   ├── io-intensive/          # I/O benchmark scripts
│   └── mixed/                 # Mixed workload scripts
├── docker-compose.yml         # Main deployment file
├── requirements.txt           # Global Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database Configuration
TIMESCALE_HOST=timescaledb
TIMESCALE_PORT=5432
TIMESCALE_DB=hpc_energy
TIMESCALE_USER=postgres
TIMESCALE_PASS=password

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERTMANAGER_PORT=9093

# API Configuration
API_PORT=5000
API_DEBUG=false
REDIS_HOST=redis
REDIS_PORT=6379

# Energy Monitoring
MONITORING_INTERVAL=30
THERMAL_WARNING_TEMP=75
THERMAL_CRITICAL_TEMP=85
```

### Slurm Integration

1. **Copy Slurm Configuration**:
   ```bash
   sudo cp slurm/slurm.conf /etc/slurm/
   sudo cp slurm/qos.conf /etc/slurm/
   ```

2. **Install Job Scripts**:
   ```bash
   sudo cp slurm/scripts/* /etc/slurm/scripts/
   sudo chmod +x /etc/slurm/scripts/*
   ```

3. **Restart Slurm Services**:
   ```bash
   sudo systemctl restart slurmctld
   sudo systemctl restart slurmd
   ```

## 📊 Usage Examples

### Running Benchmark Workloads

```bash
# CPU-intensive benchmark
sbatch --qos=cpu_intensive --job-name=cpu_bench \
       --output=cpu_bench_%j.out \
       --wrap="python3 workloads/cpu-intensive/cpu_benchmark.py --duration 300 --intensity high"

# I/O-intensive benchmark
sbatch --qos=io_intensive --job-name=io_bench \
       --output=io_bench_%j.out \
       --wrap="python3 workloads/io-intensive/io_benchmark.py --duration 300 --workload_type mixed"

# Mixed workload with energy monitoring
sbatch --qos=energy_efficient --job-name=mixed_bench \
       --output=mixed_bench_%j.out \
       --wrap="python3 workloads/mixed/mixed_benchmark.py --duration 600 --pattern concurrent"
```

### Energy Prediction API

```bash
# Predict energy consumption for a job
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "duration_seconds": 3600,
       "cpu_usage_percent": 80,
       "memory_usage_percent": 60,
       "io_read_mbps": 10,
       "io_write_mbps": 5,
       "cpu_cores": 8,
       "job_type": "cpu_intensive"
     }'

# Get scheduling recommendations
curl http://localhost:5000/recommend/schedule?job_type=cpu_intensive&duration=3600

# Check API health
curl http://localhost:5000/health
```

### Data Analysis

```bash
# Run energy analysis
python3 analysis/data_analysis.py --start_date 2024-01-01 --end_date 2024-01-31

# Generate energy efficiency report
python3 analysis/data_analysis.py --report_type efficiency --output report.json
```

## 📈 Monitoring and Dashboards

### Grafana Dashboards

1. **HPC System Overview**: Real-time system metrics and job status
2. **Energy Analysis**: Energy consumption patterns and efficiency metrics
3. **Thermal Monitoring**: Temperature trends and thermal events
4. **Job Performance**: Job execution metrics and resource utilization

### Prometheus Metrics

Key metrics collected:

- `hpc_job_energy_wh`: Energy consumption per job
- `hpc_node_cpu_temp_celsius`: CPU temperature
- `hpc_node_power_watts`: Estimated power consumption
- `hpc_job_duration_seconds`: Job execution time
- `hpc_thermal_events_total`: Thermal event counters

### Alerts

Configured alerts include:

- High CPU/GPU temperature warnings
- Excessive energy consumption
- Job failure rate thresholds
- System resource exhaustion
- Exporter downtime

## 🔬 Research and Analysis

### Energy Efficiency Metrics

- **Energy per FLOP**: Floating-point operations per joule
- **Performance per Watt**: Computational throughput per watt
- **Thermal Efficiency**: Performance vs. temperature correlation
- **Job Type Analysis**: Energy patterns by workload characteristics

### Machine Learning Models

The system includes:

- **Linear Regression**: Baseline energy prediction
- **Random Forest**: Complex pattern recognition
- **Feature Engineering**: CPU, memory, I/O, and thermal features
- **Model Validation**: Cross-validation and accuracy metrics

### Data Collection

Metrics collected every 30 seconds:

- CPU usage and frequency
- Memory utilization
- Disk I/O rates
- Network traffic
- Temperature sensors
- Power estimates

## 🛠️ Development

### Adding New Exporters

1. Create exporter directory in `monitoring/exporters/`
2. Implement Prometheus metrics endpoint
3. Add Dockerfile and configuration
4. Update docker-compose.yml
5. Configure Prometheus scraping

### Extending the API

1. Add new endpoints in `api/energy_prediction_api.py`
2. Update requirements if needed
3. Add tests and documentation
4. Rebuild API container

### Custom Workloads

1. Create workload script in appropriate `workloads/` subdirectory
2. Follow existing patterns for metrics collection
3. Add Slurm job submission examples
4. Document resource requirements

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check TimescaleDB status
   docker-compose logs timescaledb
   
   # Verify connection
   docker-compose exec timescaledb psql -U postgres -d hpc_energy -c "\dt"
   ```

2. **Missing Thermal Sensors**:
   ```bash
   # Check available sensors
   ls /sys/class/thermal/thermal_zone*/
   
   # Install lm-sensors
   sudo apt-get install lm-sensors
   sudo sensors-detect
   ```

3. **Slurm Integration Issues**:
   ```bash
   # Check Slurm logs
   sudo journalctl -u slurmctld -f
   sudo journalctl -u slurmd -f
   
   # Verify configuration
   sudo scontrol show config
   ```

4. **Prometheus Scraping Failures**:
   ```bash
   # Check target status
   curl http://localhost:9090/targets
   
   # Verify exporter endpoints
   curl http://localhost:9100/metrics
   ```

### Performance Tuning

1. **Database Optimization**:
   - Adjust TimescaleDB chunk intervals
   - Configure retention policies
   - Optimize indexes for query patterns

2. **Monitoring Overhead**:
   - Increase monitoring intervals for production
   - Reduce metric cardinality
   - Use sampling for high-frequency data

3. **Resource Allocation**:
   - Scale containers based on load
   - Adjust memory limits
   - Configure CPU affinity

## 📚 References

- [Slurm Workload Manager](https://slurm.schedmd.com/)
- [Prometheus Monitoring](https://prometheus.io/)
- [Grafana Visualization](https://grafana.com/)
- [TimescaleDB Time-series](https://www.timescale.com/)
- [HPC Energy Efficiency Research](https://www.top500.org/green500/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions and support:

- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation
- Contact the development team

---

**Note**: This project is designed for research and educational purposes. For production HPC environments, additional security hardening and performance optimization may be required.