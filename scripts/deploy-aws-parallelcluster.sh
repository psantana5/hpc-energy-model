#!/bin/bash

# AWS ParallelCluster Deployment Script for HPC Energy Model
# This script deploys the HPC Energy Model to AWS using ParallelCluster
# with S3 storage integration

set -euo pipefail

# Configuration
CLUSTER_NAME="hpc-energy-cluster"
REGION="eu-west-1"
KEY_PAIR_NAME="hpc-energy-keypair"
S3_BUCKET_PREFIX="hpc-energy-model"
VPC_CIDR="10.0.0.0/16"
SUBNET_CIDR="10.0.1.0/24"
HEAD_NODE_INSTANCE_TYPE="c5.xlarge"
COMPUTE_NODE_INSTANCE_TYPE="c5.metal"
MAX_QUEUE_SIZE="10"
MIN_QUEUE_SIZE="0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check pcluster CLI
    if ! command -v pcluster &> /dev/null; then
        log_error "AWS ParallelCluster CLI is not installed. Installing..."
        pip3 install aws-parallelcluster
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    log_success "Prerequisites check completed."
}

# Create S3 buckets
create_s3_buckets() {
    log_info "Creating S3 buckets..."
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local bucket_name="${S3_BUCKET_PREFIX}-${account_id}-${REGION}"
    
    # Create main bucket
    if ! aws s3 ls "s3://${bucket_name}" >/dev/null 2>&1; then
        aws s3 mb "s3://${bucket_name}" --region "${REGION}"
        log_success "Created S3 bucket: ${bucket_name}"
    else
        log_warning "S3 bucket ${bucket_name} already exists."
    fi
    
    # Create bucket structure
    log_info "Creating S3 bucket structure..."
    aws s3api put-object --bucket "${bucket_name}" --key "models/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "models/production/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "models/experiments/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "models/archived/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "data/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "data/training/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "data/validation/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "data/raw/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "backups/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "backups/database/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "backups/configurations/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "logs/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "logs/archived/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "shared/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "shared/slurm/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "shared/workloads/" --region "${REGION}"
    aws s3api put-object --bucket "${bucket_name}" --key "shared/scripts/" --region "${REGION}"
    
    # Set bucket policy for cluster access
    cat > /tmp/bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowParallelClusterAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::${account_id}:root"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${bucket_name}",
                "arn:aws:s3:::${bucket_name}/*"
            ]
        }
    ]
}
EOF
    
    aws s3api put-bucket-policy --bucket "${bucket_name}" --policy file:///tmp/bucket-policy.json
    rm /tmp/bucket-policy.json
    
    echo "${bucket_name}" > /tmp/s3_bucket_name
    log_success "S3 buckets created and configured."
}

# Create key pair
create_key_pair() {
    log_info "Creating EC2 key pair..."
    
    if aws ec2 describe-key-pairs --key-names "${KEY_PAIR_NAME}" --region "${REGION}" &> /dev/null; then
        log_warning "Key pair ${KEY_PAIR_NAME} already exists."
    else
        aws ec2 create-key-pair --key-name "${KEY_PAIR_NAME}" --region "${REGION}" \
            --query 'KeyMaterial' --output text > "${KEY_PAIR_NAME}.pem"
        chmod 400 "${KEY_PAIR_NAME}.pem"
        log_success "Created key pair: ${KEY_PAIR_NAME}.pem"
    fi
}

# Create IAM roles and policies
create_iam_resources() {
    log_info "Creating IAM resources..."
    
    local role_name="ParallelClusterInstanceRole-${CLUSTER_NAME}"
    local policy_name="ParallelClusterS3Policy-${CLUSTER_NAME}"
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    
    # Create IAM policy for S3 access
    local bucket_name=$(cat /tmp/s3_bucket_name)
    cat > /tmp/s3-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::${bucket_name}",
                "arn:aws:s3:::${bucket_name}/*"
            ]
        }
    ]
}
EOF
    
    # Create policy if it doesn't exist
    if ! aws iam get-policy --policy-arn "arn:aws:iam::${account_id}:policy/${policy_name}" &> /dev/null; then
        aws iam create-policy --policy-name "${policy_name}" --policy-document file:///tmp/s3-policy.json
        log_success "Created IAM policy: ${policy_name}"
    else
        log_warning "IAM policy ${policy_name} already exists."
        # Update the policy version if needed
        aws iam create-policy-version --policy-arn "arn:aws:iam::${account_id}:policy/${policy_name}" \
            --policy-document file:///tmp/s3-policy.json --set-as-default 2>/dev/null || \
            log_warning "Policy version update skipped (may have reached version limit)"
    fi
    
    rm /tmp/s3-policy.json
    log_success "IAM resources configured."
}

# Upload project files to S3
upload_project_files() {
    log_info "Uploading project files to S3..."
    
    local bucket_name=$(cat /tmp/s3_bucket_name)
    local project_root="$(dirname "$(dirname "$(realpath "$0")")")" # Go up two levels from scripts/
    
    # Upload Slurm configuration
    aws s3 cp "${project_root}/slurm/" "s3://${bucket_name}/shared/slurm/" --recursive
    
    # Upload workloads
    aws s3 cp "${project_root}/workloads/" "s3://${bucket_name}/shared/workloads/" --recursive
    
    # Upload scripts
    aws s3 cp "${project_root}/scripts/" "s3://${bucket_name}/shared/scripts/" --recursive
    
    # Upload analysis modules
    aws s3 cp "${project_root}/analysis/" "s3://${bucket_name}/shared/analysis/" --recursive
    
    # Upload Docker configurations
    aws s3 cp "${project_root}/docker-compose.yml" "s3://${bucket_name}/shared/"
    aws s3 cp "${project_root}/infrastructure/" "s3://${bucket_name}/shared/infrastructure/" --recursive
    
    log_success "Project files uploaded to S3."
}

# Create ParallelCluster configuration
create_cluster_config() {
    log_info "Creating ParallelCluster configuration..."
    
    local bucket_name=$(cat /tmp/s3_bucket_name)
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    
    cat > "${CLUSTER_NAME}-config.yaml" << EOF
Region: ${REGION}
Image:
  Os: ubuntu2004
HeadNode:
  InstanceType: ${HEAD_NODE_INSTANCE_TYPE}
  Networking:
    SubnetId: subnet-placeholder
  Ssh:
    KeyName: ${KEY_PAIR_NAME}
  LocalStorage:
    RootVolume:
      Size: 100
      VolumeType: gp3
  Iam:
    AdditionalIamPolicies:
      - Policy: arn:aws:iam::${account_id}:policy/ParallelClusterS3Policy-${CLUSTER_NAME}
  CustomActions:
    OnNodeStart:
      Script: s3://${bucket_name}/shared/scripts/node-setup.sh
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: compute
      ComputeResources:
        - Name: compute-resource
          InstanceType: ${COMPUTE_NODE_INSTANCE_TYPE}
          MinCount: ${MIN_QUEUE_SIZE}
          MaxCount: ${MAX_QUEUE_SIZE}
      Networking:
        SubnetIds:
          - subnet-placeholder
      Iam:
        AdditionalIamPolicies:
          - Policy: arn:aws:iam::${account_id}:policy/ParallelClusterS3Policy-${CLUSTER_NAME}
SharedStorage:
  - MountDir: /shared
    Name: shared-storage
    StorageType: FsxLustre
    FsxLustreSettings:
      StorageCapacity: 1200
      DeploymentType: SCRATCH_2
      ImportPath: s3://${bucket_name}/shared/
      ExportPath: s3://${bucket_name}/shared/
Monitoring:
  DetailedMonitoring: true
  Logs:
    CloudWatch:
      Enabled: true
      RetentionInDays: 14
# AdditionalPackages:
#   IntelSoftware:
#     IntelHpcPlatform: true
Tags:
  - Key: Project
    Value: HPC-Energy-Model
  - Key: Environment
    Value: Development
EOF
    
    log_success "ParallelCluster configuration created."
}

# Create VPC and networking
create_networking() {
    log_info "Creating VPC and networking..."
    
    # Check if VPC already exists
    local existing_vpc_id=$(aws ec2 describe-vpcs --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-vpc" \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "None")
    
    local vpc_id
    if [ "${existing_vpc_id}" != "None" ] && [ -n "${existing_vpc_id}" ]; then
        log_warning "VPC ${CLUSTER_NAME}-vpc already exists: ${existing_vpc_id}"
        vpc_id="${existing_vpc_id}"
    else
        # Create VPC
        vpc_id=$(aws ec2 create-vpc --cidr-block "${VPC_CIDR}" --region "${REGION}" \
            --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${CLUSTER_NAME}-vpc}]" \
            --query 'Vpc.VpcId' --output text)
        log_success "Created VPC: ${vpc_id}"
    fi
    
    # Enable DNS hostnames
    aws ec2 modify-vpc-attribute --vpc-id "${vpc_id}" --enable-dns-hostnames
    
    # Check if internet gateway already exists
    local existing_igw_id=$(aws ec2 describe-internet-gateways --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-igw" \
        --query 'InternetGateways[0].InternetGatewayId' --output text 2>/dev/null || echo "None")
    
    local igw_id
    if [ "${existing_igw_id}" != "None" ] && [ -n "${existing_igw_id}" ]; then
        log_warning "Internet Gateway ${CLUSTER_NAME}-igw already exists: ${existing_igw_id}"
        igw_id="${existing_igw_id}"
    else
        # Create internet gateway
        igw_id=$(aws ec2 create-internet-gateway --region "${REGION}" \
            --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${CLUSTER_NAME}-igw}]" \
            --query 'InternetGateway.InternetGatewayId' --output text)
        log_success "Created Internet Gateway: ${igw_id}"
    fi
    
    # Attach internet gateway to VPC (idempotent)
    aws ec2 attach-internet-gateway --vpc-id "${vpc_id}" --internet-gateway-id "${igw_id}" 2>/dev/null || \
        log_warning "Internet Gateway ${igw_id} already attached to VPC ${vpc_id}"
    
    # Check if subnet already exists
    local existing_subnet_id=$(aws ec2 describe-subnets --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-subnet" \
        --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "None")
    
    local subnet_id
    if [ "${existing_subnet_id}" != "None" ] && [ -n "${existing_subnet_id}" ]; then
        log_warning "Subnet ${CLUSTER_NAME}-subnet already exists: ${existing_subnet_id}"
        subnet_id="${existing_subnet_id}"
    else
        # Create subnet
        subnet_id=$(aws ec2 create-subnet --vpc-id "${vpc_id}" --cidr-block "${SUBNET_CIDR}" \
            --availability-zone "${REGION}a" --region "${REGION}" \
            --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${CLUSTER_NAME}-subnet}]" \
            --query 'Subnet.SubnetId' --output text)
        log_success "Created Subnet: ${subnet_id}"
    fi
    
    # Check if route table already exists
    local existing_rt_id=$(aws ec2 describe-route-tables --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-rt" \
        --query 'RouteTables[0].RouteTableId' --output text 2>/dev/null || echo "None")
    
    local rt_id
    if [ "${existing_rt_id}" != "None" ] && [ -n "${existing_rt_id}" ]; then
        log_warning "Route Table ${CLUSTER_NAME}-rt already exists: ${existing_rt_id}"
        rt_id="${existing_rt_id}"
    else
        # Create route table
        rt_id=$(aws ec2 create-route-table --vpc-id "${vpc_id}" --region "${REGION}" \
            --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=${CLUSTER_NAME}-rt}]" \
            --query 'RouteTable.RouteTableId' --output text)
        log_success "Created Route Table: ${rt_id}"
        
        # Add route to internet gateway
        aws ec2 create-route --route-table-id "${rt_id}" --destination-cidr-block "0.0.0.0/0" \
            --gateway-id "${igw_id}"
        
        # Associate route table with subnet
        aws ec2 associate-route-table --subnet-id "${subnet_id}" --route-table-id "${rt_id}"
    fi
    
    # Update cluster config with actual subnet ID
    sed -i "" "s|subnet-placeholder|${subnet_id}|g" "${CLUSTER_NAME}-config.yaml"
    
    echo "${vpc_id}" > /tmp/vpc_id
    echo "${subnet_id}" > /tmp/subnet_id
    
    log_success "VPC and networking configured."
}

# Create node setup script
create_node_setup_script() {
    log_info "Creating node setup script..."
    
    local bucket_name=$(cat /tmp/s3_bucket_name)
    
    cat > node-setup.sh << 'EOF'
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
echo "*/5 * * * * root aws s3 sync /shared/logs/ s3://BUCKET_NAME/logs/" >> /etc/crontab
echo "0 2 * * * root aws s3 sync /shared/models/ s3://BUCKET_NAME/models/" >> /etc/crontab

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
        aws s3 sync /shared/logs/ s3://BUCKET_NAME/logs/archived/
    endscript
}
LOGROTATE_EOF

echo "Node setup completed"
EOF
    
    # Replace bucket name placeholder
    sed -i "" "s/BUCKET_NAME/${bucket_name}/g" node-setup.sh
    
    # Upload to S3
    aws s3 cp node-setup.sh "s3://${bucket_name}/shared/scripts/"
    
    log_success "Node setup script created and uploaded."
}

# Create cluster
create_cluster() {
    log_info "Creating ParallelCluster..."
    
    # Check if cluster already exists
    local existing_status=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" \
        --region "${REGION}" 2>/dev/null | grep -o '"clusterStatus": "[^"]*"' | cut -d'"' -f4 || echo "NOT_FOUND")
    
    if [ "${existing_status}" = "CREATE_COMPLETE" ]; then
        log_warning "Cluster ${CLUSTER_NAME} already exists and is ready."
        return 0
    elif [ "${existing_status}" = "CREATE_IN_PROGRESS" ]; then
        log_warning "Cluster ${CLUSTER_NAME} is already being created. Waiting for completion..."
    elif [ "${existing_status}" = "CREATE_FAILED" ]; then
        log_error "Cluster ${CLUSTER_NAME} exists but creation failed. Please delete it first."
        exit 1
    elif [ "${existing_status}" = "DELETE_IN_PROGRESS" ]; then
        log_error "Cluster ${CLUSTER_NAME} is being deleted. Please wait for deletion to complete."
        exit 1
    elif [ "${existing_status}" = "NOT_FOUND" ]; then
        # Validate configuration (skip if config already validated)
        if [ ! -f "${CLUSTER_NAME}-config.yaml.bak" ]; then
            cp "${CLUSTER_NAME}-config.yaml" "${CLUSTER_NAME}-config.yaml.bak"
            pcluster configure --config "${CLUSTER_NAME}-config.yaml" || {
                log_warning "Configuration validation skipped (file may already be validated)"
                cp "${CLUSTER_NAME}-config.yaml.bak" "${CLUSTER_NAME}-config.yaml"
            }
        else
            log_warning "Configuration already validated, skipping pcluster configure"
        fi
        
        # Create cluster
        pcluster create-cluster --cluster-name "${CLUSTER_NAME}" \
            --cluster-configuration "${CLUSTER_NAME}-config.yaml" \
            --region "${REGION}"
        
        log_info "Cluster creation initiated. This may take 15-20 minutes..."
    fi
    
    # Wait for cluster to be ready
    while true; do
        local status=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" \
            --region "${REGION}" 2>/dev/null | grep -o '"clusterStatus": "[^"]*"' | cut -d'"' -f4 || echo "UNKNOWN")
        
        if [ "$status" = "CREATE_COMPLETE" ]; then
            log_success "Cluster ${CLUSTER_NAME} is ready!"
            break
        elif [ "$status" = "CREATE_FAILED" ]; then
            log_error "Cluster creation failed!"
            exit 1
        elif [ "$status" = "UNKNOWN" ]; then
            log_error "Unable to get cluster status. Please check manually."
            exit 1
        else
            log_info "Cluster status: $status. Waiting... Note that this will take a long time..."
            sleep 60
        fi
    done
}

# Deploy application
deploy_application() {
    log_info "Deploying HPC Energy Model application..."
    
    local bucket_name=$(cat /tmp/s3_bucket_name)
    local head_node_ip=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" \
        --region "${REGION}" --query 'headNode.publicIpAddress' --output text)
    
    # Create deployment script
    cat > deploy-app.sh << EOF
#!/bin/bash
set -euo pipefail

# Download application files
aws s3 sync s3://${bucket_name}/shared/ /shared/

# Set up environment
export AWS_S3_BUCKET=${bucket_name}
export AWS_REGION=${REGION}

# Create environment file
cat > /shared/.env << ENV_EOF
TIMESCALE_DB=hpc_energy
TIMESCALE_USER=postgres
TIMESCALE_PASS=hpc_energy_pass
TIMESCALE_PORT=5432
REDIS_PASSWORD=hpc_redis_pass
REDIS_PORT=6379
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
AWS_S3_BUCKET=${bucket_name}
AWS_REGION=${REGION}
S3_MODEL_PREFIX=models/
S3_DATA_PREFIX=data/
S3_BACKUP_PREFIX=backups/
ENV_EOF

# Start services using Docker Compose
cd /shared
docker-compose up -d

# Wait for services to be ready
sleep 30

# Run initial setup
python3 /shared/analysis/data_analysis.py --setup

echo "Application deployment completed"
EOF
    
    # Copy and execute deployment script
    scp -i "${KEY_PAIR_NAME}.pem" -o StrictHostKeyChecking=no \
        deploy-app.sh "ubuntu@${head_node_ip}:/tmp/"
    
    ssh -i "${KEY_PAIR_NAME}.pem" -o StrictHostKeyChecking=no \
        "ubuntu@${head_node_ip}" "chmod +x /tmp/deploy-app.sh && sudo /tmp/deploy-app.sh"
    
    log_success "Application deployed successfully!"
    
    # Display connection information
    echo ""
    echo "=== Deployment Complete ==="
    echo "Cluster Name: ${CLUSTER_NAME}"
    echo "Head Node IP: ${head_node_ip}"
    echo "S3 Bucket: ${bucket_name}"
    echo "SSH Command: ssh -i ${KEY_PAIR_NAME}.pem ubuntu@${head_node_ip}"
    echo "Grafana URL: http://${head_node_ip}:3000 (admin/admin)"
    echo "Prometheus URL: http://${head_node_ip}:9090"
    echo "API URL: http://${head_node_ip}:8000"
    echo ""
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/s3_bucket_name /tmp/vpc_id /tmp/subnet_id
    rm -f node-setup.sh deploy-app.sh
}

# Main deployment function
main() {
    log_info "Starting AWS ParallelCluster deployment for HPC Energy Model..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    create_s3_buckets
    create_key_pair
    create_iam_resources
    upload_project_files
    create_cluster_config
    create_networking
    create_node_setup_script
    create_cluster
    deploy_application
    
    log_success "Deployment completed successfully!"
}

# Script options
case "${1:-}" in
    "destroy")
        log_info "Destroying cluster and resources..."
        pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}"
        
        # Wait for cluster deletion
        while pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" &> /dev/null; do
            log_info "Waiting for cluster deletion..."
            sleep 30
        done
        
        # Clean up VPC resources
        if [ -f /tmp/vpc_id ]; then
            vpc_id=$(cat /tmp/vpc_id)
            aws ec2 delete-vpc --vpc-id "${vpc_id}" --region "${REGION}" || true
        fi
        
        # Clean up S3 bucket (optional)
        read -p "Delete S3 bucket and all data? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bucket_name=$(cat /tmp/s3_bucket_name 2>/dev/null || echo "")
            if [ -n "${bucket_name}" ]; then
                aws s3 rb "s3://${bucket_name}" --force
                log_success "S3 bucket deleted."
            fi
        fi
        
        log_success "Cleanup completed."
        ;;
    "status")
        pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}"
        ;;
    "connect")
        head_node_ip=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" \
            --region "${REGION}" --query 'headNode.publicIpAddress' --output text)
        ssh -i "${KEY_PAIR_NAME}.pem" "ubuntu@${head_node_ip}"
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [destroy|status|connect]"
        echo "  (no args)  - Deploy cluster"
        echo "  destroy    - Destroy cluster and resources"
        echo "  status     - Show cluster status"
        echo "  connect    - SSH to head node"
        exit 1
        ;;
esac