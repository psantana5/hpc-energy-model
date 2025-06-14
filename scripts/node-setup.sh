#!/bin/bash

# Node setup script for HPC Energy Model
set -euo pipefail

# Install Docker
apt-get update
apt-get install -y docker.io docker-compose
systemctl enable docker
systemctl start docker

# Install Python dependencies
apt-get install -y python3-pip python3-venv
pip3 install boto3 psutil numpy pandas scikit-learn

# Install monitoring tools
apt-get install -y htop iotop sysstat

# Configure S3 sync for shared data
echo "*/5 * * * * root aws s3 sync /shared/logs/ s3://hpc-energy-model-104398007985-eu-west-1/logs/" >> /etc/crontab
echo "0 2 * * * root aws s3 sync /shared/models/ s3://hpc-energy-model-104398007985-eu-west-1/models/" >> /etc/crontab

# Set up energy monitoring
mkdir -p /opt/hpc-energy
cp /shared/scripts/energy_monitor.sh /opt/hpc-energy/
chmod +x /opt/hpc-energy/energy_monitor.sh

# Configure Slurm with energy monitoring
cp /shared/slurm/* /opt/slurm/etc/

# Start services
systemctl restart slurmd

# Set up log rotation
cat > /etc/logrotate.d/hpc-energy << 'LOGROTATE_EOF'
/shared/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        aws s3 sync /shared/logs/ s3://hpc-energy-model-104398007985-eu-west-1/logs/archived/
    endscript
}
LOGROTATE_EOF

echo "Node setup completed"
