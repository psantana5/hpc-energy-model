# HPC Energy Model Deployment Guide

This guide provides comprehensive instructions for deploying the HPC Energy Model system on various platforms including bare-metal servers, virtual machines, and different hypervisors.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Platform-Specific Deployment](#platform-specific-deployment)
- [Configuration](#configuration)
- [Advanced Logging Setup](#advanced-logging-setup)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum, SSD recommended
- **Network**: Gigabit Ethernet recommended

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- Python 3.8+ (for development)
- Slurm Workload Manager (for HPC integration)

### Supported Platforms

- **Bare Metal**: Linux servers (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Hypervisors**: VMware vSphere, KVM/QEMU, Xen, Hyper-V
- **Cloud**: AWS EC2, Google Cloud, Azure VMs
- **Container Platforms**: Docker, Podman, Kubernetes

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/hpc-energy-model.git
cd hpc-energy-model
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit the configuration
nano .env
```

### 3. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the System

- **Grafana Dashboard**: http://localhost:3000 (admin/admin_change_me)
- **Prometheus**: http://localhost:9090
- **Energy API**: http://localhost:5000
- **Nginx Proxy**: http://localhost:80

## Platform-Specific Deployment

### Bare Metal Deployment

#### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Configure environment
echo 'HYPERVISOR_TYPE=baremetal' >> .env

# Deploy
docker-compose up -d
```

#### CentOS/RHEL

```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Configure SELinux (if enabled)
sudo setsebool -P container_manage_cgroup on

# Deploy
docker-compose up -d
```

### VMware vSphere Deployment

```bash
# Configure VMware-specific settings
echo 'HYPERVISOR_TYPE=vmware' >> .env
echo 'VMWARE_HOST=your-vcenter-host' >> .env
echo 'VMWARE_USER=your-username' >> .env
echo 'VMWARE_PASSWORD=your-password' >> .env
echo 'VMWARE_DATACENTER=your-datacenter' >> .env

# Deploy with VMware monitoring
docker-compose -f docker-compose.yml -f docker-compose.vmware.yml up -d
```

### KVM/QEMU Deployment

```bash
# Configure KVM-specific settings
echo 'HYPERVISOR_TYPE=kvm' >> .env

# Install libvirt tools for monitoring
sudo apt install -y libvirt-clients qemu-utils

# Deploy
docker-compose up -d
```

### Hyper-V Deployment

```powershell
# PowerShell commands for Windows Server
# Enable Hyper-V monitoring
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Management-PowerShell

# Configure environment
echo 'HYPERVISOR_TYPE=hyperv' | Out-File -Append .env

# Deploy using Docker Desktop
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace hpc-energy

# Deploy using Helm (if available)
helm install hpc-energy ./k8s/helm-chart -n hpc-energy

# Or deploy using kubectl
kubectl apply -f k8s/manifests/ -n hpc-energy
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Hypervisor Configuration
HYPERVISOR_TYPE=baremetal  # Options: baremetal, vmware, kvm, xen, hyperv

# Database Configuration
TIMESCALE_HOST=timescaledb
TIMESCALE_DB=hpc_energy
TIMESCALE_USER=postgres
TIMESCALE_PASS=secure_password_change_me

# Monitoring Configuration
MONITORING_INTERVAL=30
THERMAL_WARNING_TEMP=75
THERMAL_CRITICAL_TEMP=85

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_RETENTION_DAYS=30
```

### Slurm Integration

1. **Copy Slurm Configuration**:
   ```bash
   sudo cp slurm/slurm.conf /etc/slurm/
   sudo cp slurm/qos.conf /etc/slurm/
   ```

2. **Install Slurm Scripts**:
   ```bash
   sudo cp slurm/scripts/* /etc/slurm/scripts/
   sudo chmod +x /etc/slurm/scripts/*
   ```

3. **Configure Slurm Database**:
   ```bash
   # Update slurmdbd.conf
   sudo nano /etc/slurm/slurmdbd.conf
   ```

4. **Restart Slurm Services**:
   ```bash
   sudo systemctl restart slurmdbd
   sudo systemctl restart slurmctld
   sudo systemctl restart slurmd
   ```

## Advanced Logging Setup

### Centralized Logging with Fluentd

The system includes Fluentd for centralized log collection:

```yaml
# Fluentd configuration is included in docker-compose.yml
fluentd:
  image: fluent/fluentd:v1.16-debian-1
  volumes:
    - ./infrastructure/fluentd/fluent.conf:/fluentd/etc/fluent.conf
    - ./logs:/var/log/hpc
  ports:
    - "24224:24224"
```

### Log Aggregation Configuration

1. **Configure Log Sources**:
   ```bash
   # Edit Fluentd configuration
   nano infrastructure/fluentd/fluent.conf
   ```

2. **Set Log Levels**:
   ```bash
   # In .env file
   LOG_LEVEL=INFO
   API_LOG_LEVEL=DEBUG
   THERMAL_LOG_LEVEL=INFO
   JOB_LOG_LEVEL=INFO
   ```

3. **Configure Log Rotation**:
   ```bash
   LOG_RETENTION_DAYS=30
   LOG_MAX_SIZE=100M
   LOG_MAX_FILES=5
   ```

### ELK Stack Integration (Optional)

```bash
# Deploy with ELK stack
docker-compose -f docker-compose.yml -f docker-compose.elk.yml up -d
```

## Monitoring and Alerting

### Prometheus Configuration

Prometheus is configured to scrape metrics from:
- Node Exporter (system metrics)
- Thermal Exporter (temperature data)
- Job Exporter (Slurm job metrics)
- Energy API (prediction metrics)

### Grafana Dashboards

Pre-configured dashboards:
- **HPC System Overview**: `dashboards/hpc-system-overview.json`
- **Energy Analysis**: `dashboards/energy-analysis.json`

### Alerting Rules

Configure alerts in `infrastructure/prometheus/alert.rules.yml`:

```yaml
groups:
  - name: hpc_energy_alerts
    rules:
      - alert: HighTemperature
        expr: thermal_temperature > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High temperature detected on {{ $labels.instance }}"
```

## Troubleshooting

### Common Issues

1. **Services Not Starting**:
   ```bash
   # Check service logs
   docker-compose logs service_name
   
   # Check system resources
   docker system df
   docker system prune
   ```

2. **Database Connection Issues**:
   ```bash
   # Test database connectivity
   docker-compose exec timescaledb psql -U postgres -d hpc_energy
   
   # Check database logs
   docker-compose logs timescaledb
   ```

3. **Slurm Integration Issues**:
   ```bash
   # Check Slurm status
   sinfo
   squeue
   
   # Check Slurm logs
   sudo tail -f /var/log/slurm/slurmctld.log
   ```

4. **Monitoring Data Missing**:
   ```bash
   # Check exporter status
   curl http://localhost:9100/metrics  # Node Exporter
   curl http://localhost:9101/metrics  # Thermal Exporter
   curl http://localhost:9102/metrics  # Job Exporter
   ```

### Performance Tuning

1. **Database Optimization**:
   ```sql
   -- Connect to TimescaleDB
   \c hpc_energy
   
   -- Check table sizes
   SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables WHERE schemaname = 'hpc_energy';
   
   -- Optimize queries
   ANALYZE;
   ```

2. **Container Resource Limits**:
   ```bash
   # Monitor container resources
   docker stats
   
   # Adjust limits in .env
   TIMESCALEDB_MEMORY_LIMIT=4g
   PROMETHEUS_MEMORY_LIMIT=2g
   ```

### Log Analysis

```bash
# View aggregated logs
docker-compose logs -f --tail=100

# Filter specific service logs
docker-compose logs -f api

# Search logs for errors
docker-compose logs | grep ERROR

# Export logs for analysis
docker-compose logs > system_logs.txt
```

## Maintenance

### Regular Maintenance Tasks

1. **Database Maintenance**:
   ```bash
   # Run weekly
   docker-compose exec timescaledb psql -U postgres -d hpc_energy -c "VACUUM ANALYZE;"
   ```

2. **Log Rotation**:
   ```bash
   # Automated via logrotate
   sudo logrotate -f /etc/logrotate.d/hpc-energy
   ```

3. **Backup**:
   ```bash
   # Database backup
   docker-compose exec timescaledb pg_dump -U postgres hpc_energy > backup_$(date +%Y%m%d).sql
   
   # Configuration backup
   tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml infrastructure/
   ```

4. **Updates**:
   ```bash
   # Update images
   docker-compose pull
   docker-compose up -d
   
   # Clean up old images
   docker image prune -f
   ```

### Monitoring Health

```bash
# Check service health
docker-compose ps

# Check system health endpoint
curl http://localhost:5000/health

# Monitor resource usage
docker stats --no-stream
```

### Scaling

```bash
# Scale specific services
docker-compose up -d --scale api=3

# Load balancing with nginx
# Configuration in infrastructure/nginx/nginx.conf
```

## Security Considerations

1. **Change Default Passwords**:
   ```bash
   # Update all passwords in .env
   TIMESCALE_PASS=your_secure_password
   GRAFANA_ADMIN_PASSWORD=your_secure_password
   REDIS_PASSWORD=your_secure_password
   ```

2. **Enable SSL/TLS**:
   ```bash
   # Generate certificates
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout infrastructure/nginx/ssl/key.pem \
     -out infrastructure/nginx/ssl/cert.pem
   
   # Enable SSL in .env
   SSL_ENABLED=true
   ```

3. **Network Security**:
   ```bash
   # Configure firewall
   sudo ufw allow 22/tcp
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

4. **Container Security**:
   ```bash
   # Run security scan
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
     aquasec/trivy image timescale/timescaledb:latest-pg14
   ```

## Support

For additional support:
- Check the [README.md](README.md) for general information
- Review logs using the troubleshooting section
- Submit issues on the project repository
- Contact the development team

---

**Note**: This deployment guide is designed to be hypervisor-agnostic and supports deployment on bare-metal servers, virtual machines, and various cloud platforms. The system automatically detects the environment and adjusts monitoring accordingly.