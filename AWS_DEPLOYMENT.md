# AWS ParallelCluster Deployment Guide

This guide provides step-by-step instructions for deploying the HPC Energy Model on AWS using ParallelCluster with S3 storage integration.

## Overview

The AWS deployment creates:
- **ParallelCluster**: Managed HPC cluster with Slurm scheduler
- **S3 Storage**: Scalable storage for models, data, and backups
- **FSx Lustre**: High-performance shared filesystem
- **VPC**: Isolated network environment
- **IAM Roles**: Secure access management
- **Monitoring**: CloudWatch integration

## Prerequisites

### 1. AWS Account Setup
- Active AWS account with appropriate permissions
- AWS CLI installed and configured
- Sufficient service limits (EC2 instances, VPC resources)

### 2. Local Requirements
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install ParallelCluster CLI
pip3 install aws-parallelcluster

# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose
```

### 3. AWS Configuration
```bash
# Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, Region, and output format

# Verify configuration
aws sts get-caller-identity
```

## Quick Start

### 1. Deploy the Cluster
```bash
# Make the script executable
chmod +x scripts/deploy-aws-parallelcluster.sh

# Run deployment
./scripts/deploy-aws-parallelcluster.sh
```

### 2. Monitor Deployment
The script will:
1. Create S3 buckets and upload project files
2. Set up VPC and networking
3. Create IAM roles and policies
4. Deploy ParallelCluster (15-20 minutes)
5. Install and configure the HPC Energy Model

### 3. Access the Cluster
```bash
# SSH to head node
./scripts/deploy-aws-parallelcluster.sh connect

# Check cluster status
./scripts/deploy-aws-parallelcluster.sh status
```

## Configuration Options

Edit the script variables to customize your deployment:

```bash
# Cluster configuration
CLUSTER_NAME="hpc-energy-cluster"          # Cluster name
REGION="us-east-1"                         # AWS region
INSTANCE_TYPE="c5.xlarge"                  # Instance type
MAX_QUEUE_SIZE="10"                        # Maximum compute nodes
MIN_QUEUE_SIZE="0"                         # Minimum compute nodes

# Storage configuration
S3_BUCKET_PREFIX="hpc-energy-model"        # S3 bucket prefix

# Network configuration
VPC_CIDR="10.0.0.0/16"                     # VPC CIDR block
SUBNET_CIDR="10.0.1.0/24"                  # Subnet CIDR block
```

## S3 Storage Structure

The deployment creates the following S3 bucket structure:

```
hpc-energy-model-{account-id}-{region}/
├── models/
│   ├── production/          # Production ML models
│   ├── experiments/         # Experimental models
│   └── archived/           # Archived models
├── data/
│   ├── training/           # Training datasets
│   ├── validation/         # Validation datasets
│   └── raw/               # Raw data files
├── backups/
│   ├── database/          # Database backups
│   └── configurations/    # Configuration backups
├── logs/
│   └── archived/          # Archived log files
└── shared/
    ├── slurm/             # Slurm configuration
    ├── workloads/         # Benchmark workloads
    ├── scripts/           # Deployment scripts
    ├── analysis/          # Analysis modules
    └── infrastructure/    # Docker configurations
```

## Cluster Architecture

### Head Node
- **Instance Type**: c5.xlarge (configurable)
- **Storage**: 100GB EBS volume
- **Services**: Slurm controller, Docker services, monitoring
- **Access**: SSH, web interfaces

### Compute Nodes
- **Instance Type**: c5.xlarge (configurable)
- **Auto Scaling**: 0-10 nodes (configurable)
- **Storage**: Shared FSx Lustre filesystem
- **Services**: Slurm compute, energy monitoring

### Shared Storage
- **FSx Lustre**: High-performance filesystem (1.2TB)
- **S3 Integration**: Automatic sync with S3 bucket
- **Mount Point**: `/shared`

## Services and Endpoints

After deployment, the following services are available:

### Web Interfaces
- **Grafana**: `http://{head-node-ip}:3000` (admin/admin)
- **Prometheus**: `http://{head-node-ip}:9090`
- **API**: `http://{head-node-ip}:8000`

### Databases
- **TimescaleDB**: Port 5432
- **Redis**: Port 6379

### Monitoring
- **Node Exporter**: Port 9100
- **Job Exporter**: Port 9101
- **Thermal Exporter**: Port 9102

## Usage Examples

### 1. Submit a Job
```bash
# SSH to head node
ssh -i hpc-energy-keypair.pem ubuntu@{head-node-ip}

# Submit CPU benchmark
sbatch /shared/workloads/cpu-intensive/cpu_benchmark.py

# Check job status
squeue

# View job output
cat slurm-{job-id}.out
```

### 2. Run ML Training
```bash
# Access the cluster
ssh -i hpc-energy-keypair.pem ubuntu@{head-node-ip}

# Run energy prediction training
cd /shared
python3 analysis/advanced_ml_models.py --train --model xgboost

# View results
ls -la models/
```

### 3. Monitor Performance
```bash
# View real-time metrics
htop

# Check energy consumption
python3 analysis/performance_benchmarking.py --monitor

# Access Grafana dashboards
# Open http://{head-node-ip}:3000 in browser
```

## Cost Optimization

### 1. Instance Selection
```bash
# For development/testing
INSTANCE_TYPE="t3.medium"
MAX_QUEUE_SIZE="2"

# For production workloads
INSTANCE_TYPE="c5.2xlarge"
MAX_QUEUE_SIZE="20"

# For ML-intensive tasks
INSTANCE_TYPE="p3.2xlarge"  # GPU instances
```

### 2. Auto Scaling
- Compute nodes automatically scale based on job queue
- Minimum nodes set to 0 to minimize costs
- Nodes terminate when idle for 10 minutes

### 3. Storage Optimization
- Use S3 Intelligent Tiering for automatic cost optimization
- Archive old logs and models to S3 Glacier
- Regular cleanup of temporary files

### 4. Spot Instances
Modify the cluster configuration to use Spot instances:

```yaml
# Add to cluster config
Scheduling:
  SlurmQueues:
    - Name: spot-queue
      ComputeResources:
        - Name: spot-resource
          InstanceType: c5.xlarge
          MinCount: 0
          MaxCount: 10
          SpotPrice: 0.05  # Maximum spot price
```

## Monitoring and Logging

### CloudWatch Integration
- Cluster metrics automatically sent to CloudWatch
- Custom dashboards for energy monitoring
- Alerts for resource utilization

### Log Management
- Application logs stored in `/shared/logs/`
- Automatic rotation and archival to S3
- Centralized logging with Fluentd

### Performance Monitoring
- Real-time system metrics collection
- Energy consumption tracking
- Job performance analysis

## Security

### Network Security
- VPC with private subnets for compute nodes
- Security groups restrict access to necessary ports
- SSH access only from your IP (configurable)

### IAM Security
- Least privilege access policies
- Separate roles for head node and compute nodes
- S3 bucket policies for secure data access

### Data Security
- S3 bucket encryption at rest
- EBS volume encryption
- Secure data transfer with SSL/TLS

## Troubleshooting

### Common Issues

1. **Cluster Creation Fails**
   ```bash
   # Check CloudFormation events
   aws cloudformation describe-stack-events --stack-name parallelcluster-{cluster-name}
   
   # Check ParallelCluster logs
   pcluster describe-cluster --cluster-name {cluster-name} --region {region}
   ```

2. **SSH Connection Issues**
   ```bash
   # Verify key pair permissions
   chmod 400 hpc-energy-keypair.pem
   
   # Check security group rules
   aws ec2 describe-security-groups --group-names parallelcluster-{cluster-name}-*
   ```

3. **S3 Access Issues**
   ```bash
   # Verify IAM policies
   aws iam list-attached-role-policies --role-name ParallelClusterInstanceRole-{cluster-name}
   
   # Test S3 access from head node
   aws s3 ls s3://{bucket-name}/
   ```

4. **Service Startup Issues**
   ```bash
   # Check Docker services
   sudo docker ps -a
   sudo docker-compose logs
   
   # Check system logs
   sudo journalctl -u docker
   sudo tail -f /var/log/syslog
   ```

### Performance Issues

1. **Slow Job Execution**
   - Check compute node scaling
   - Verify shared filesystem performance
   - Monitor network bandwidth

2. **High Costs**
   - Review instance types and sizes
   - Check auto-scaling configuration
   - Monitor idle resources

3. **Storage Issues**
   - Monitor FSx Lustre performance
   - Check S3 sync frequency
   - Verify storage quotas

## Maintenance

### Regular Tasks

1. **Update Software**
   ```bash
   # Update ParallelCluster CLI
   pip3 install --upgrade aws-parallelcluster
   
   # Update cluster (requires recreation)
   ./scripts/deploy-aws-parallelcluster.sh destroy
   ./scripts/deploy-aws-parallelcluster.sh
   ```

2. **Backup Data**
   ```bash
   # Database backup (automated)
   aws s3 ls s3://{bucket-name}/backups/database/
   
   # Manual backup
   aws s3 sync /shared/data/ s3://{bucket-name}/backups/manual/
   ```

3. **Clean Up Logs**
   ```bash
   # Archive old logs
   find /shared/logs/ -name "*.log" -mtime +7 -exec aws s3 mv {} s3://{bucket-name}/logs/archived/ \;
   ```

### Scaling Operations

1. **Increase Cluster Size**
   ```bash
   # Edit cluster configuration
   vim {cluster-name}-config.yaml
   
   # Update cluster
   pcluster update-cluster --cluster-name {cluster-name} --cluster-configuration {cluster-name}-config.yaml
   ```

2. **Add New Instance Types**
   ```bash
   # Add new queue with different instance type
   # Edit configuration and update cluster
   ```

## Cleanup

### Destroy Cluster
```bash
# Destroy cluster and all resources
./scripts/deploy-aws-parallelcluster.sh destroy

# This will:
# 1. Delete the ParallelCluster
# 2. Remove VPC and networking resources
# 3. Optionally delete S3 bucket and data
```

### Partial Cleanup
```bash
# Keep S3 data, destroy cluster only
pcluster delete-cluster --cluster-name {cluster-name} --region {region}

# Manual VPC cleanup if needed
aws ec2 delete-vpc --vpc-id {vpc-id}
```

## Support

For issues and questions:

1. **AWS ParallelCluster**: [Official Documentation](https://docs.aws.amazon.com/parallelcluster/)
2. **Project Issues**: GitHub Issues
3. **AWS Support**: AWS Support Center

## Cost Estimation

### Typical Monthly Costs (us-east-1)

**Development Environment:**
- Head Node (c5.xlarge): ~$120/month
- Compute Nodes (2x c5.xlarge, 8h/day): ~$60/month
- FSx Lustre (1.2TB): ~$180/month
- S3 Storage (100GB): ~$3/month
- **Total**: ~$363/month

**Production Environment:**
- Head Node (c5.2xlarge): ~$240/month
- Compute Nodes (10x c5.2xlarge, 12h/day): ~$900/month
- FSx Lustre (2.4TB): ~$360/month
- S3 Storage (1TB): ~$25/month
- **Total**: ~$1,525/month

*Note: Costs vary by region and usage patterns. Use AWS Cost Calculator for accurate estimates.*

---

*This deployment guide is part of the HPC Energy Model project for AWS cloud deployment.*