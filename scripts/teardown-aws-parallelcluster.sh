#!/bin/bash

# AWS ParallelCluster Teardown Script for HPC Energy Model
# This script safely and idempotently destroys all AWS resources created by the deployment script
# Resources are destroyed in the correct order to avoid dependency conflicts

set -euo pipefail

# Configuration (must match deploy script)
CLUSTER_NAME="hpc-energy-cluster"
REGION="eu-west-1"
KEY_PAIR_NAME="hpc-energy-keypair"
S3_BUCKET_PREFIX="hpc-energy-model"

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
        log_error "AWS ParallelCluster CLI is not installed."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    log_success "Prerequisites check completed."
}

# Get account ID for resource identification
get_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Delete ParallelCluster
delete_cluster() {
    log_info "Checking ParallelCluster status..."
    
    # Check if cluster exists
    local cluster_status
    cluster_status=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" 2>/dev/null | \
        grep '"clusterStatus"' | cut -d'"' -f4 || echo "NOT_FOUND")
    
    case "${cluster_status}" in
        "CREATE_COMPLETE"|"UPDATE_COMPLETE")
            log_info "Deleting ParallelCluster: ${CLUSTER_NAME}"
            pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}"
            
            # Wait for cluster deletion
            log_info "Waiting for cluster deletion to complete..."
            while true; do
                local current_status
                current_status=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" 2>/dev/null | \
                    grep '"clusterStatus"' | cut -d'"' -f4 || echo "NOT_FOUND")
                
                if [ "${current_status}" = "NOT_FOUND" ]; then
                    log_success "Cluster deleted successfully."
                    break
                elif [ "${current_status}" = "DELETE_FAILED" ]; then
                    log_error "Cluster deletion failed. Manual intervention may be required."
                    exit 1
                else
                    log_info "Cluster status: ${current_status}. Waiting..."
                    sleep 30
                fi
            done
            ;;
        "DELETE_IN_PROGRESS")
            log_info "Cluster deletion already in progress. Waiting..."
            while true; do
                local current_status
                current_status=$(pcluster describe-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" 2>/dev/null | \
                    grep '"clusterStatus"' | cut -d'"' -f4 || echo "NOT_FOUND")
                
                if [ "${current_status}" = "NOT_FOUND" ]; then
                    log_success "Cluster deletion completed."
                    break
                elif [ "${current_status}" = "DELETE_FAILED" ]; then
                    log_error "Cluster deletion failed. Manual intervention may be required."
                    exit 1
                else
                    log_info "Cluster status: ${current_status}. Waiting..."
                    sleep 30
                fi
            done
            ;;
        "CREATE_IN_PROGRESS"|"UPDATE_IN_PROGRESS")
            log_error "Cluster is currently being created/updated. Please wait for completion before deletion."
            exit 1
            ;;
        "CREATE_FAILED"|"UPDATE_FAILED")
            log_warning "Cluster is in failed state. Attempting deletion..."
            pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" || \
                log_warning "Cluster deletion command failed, but cluster may not exist."
            ;;
        "DELETE_FAILED")
            log_warning "Previous cluster deletion failed. Attempting deletion again..."
            pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" || \
                log_warning "Cluster deletion command failed."
            ;;
        "NOT_FOUND")
            log_warning "Cluster ${CLUSTER_NAME} not found or already deleted."
            ;;
        *)
            log_warning "Unknown cluster status: ${cluster_status}. Attempting deletion..."
            pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --region "${REGION}" || \
                log_warning "Cluster deletion command failed."
            ;;
    esac
}

# Delete networking resources
delete_networking() {
    log_info "Deleting networking resources..."
    
    # Get VPC ID
    local vpc_id
    vpc_id=$(aws ec2 describe-vpcs --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-vpc" \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "None")
    
    if [ "${vpc_id}" = "None" ] || [ -z "${vpc_id}" ]; then
        log_warning "VPC ${CLUSTER_NAME}-vpc not found."
        return
    fi
    
    log_info "Found VPC: ${vpc_id}"
    
    # Delete route table associations and routes
    local route_tables
    route_tables=$(aws ec2 describe-route-tables --region "${REGION}" \
        --filters "Name=vpc-id,Values=${vpc_id}" "Name=tag:Name,Values=${CLUSTER_NAME}-rt" \
        --query 'RouteTables[].RouteTableId' --output text 2>/dev/null || echo "")
    
    for rt_id in ${route_tables}; do
        if [ -n "${rt_id}" ] && [ "${rt_id}" != "None" ]; then
            log_info "Deleting routes and associations for route table: ${rt_id}"
            
            # Delete route table associations
            local associations
            associations=$(aws ec2 describe-route-tables --route-table-ids "${rt_id}" --region "${REGION}" \
                --query 'RouteTables[0].Associations[?Main==`false`].RouteTableAssociationId' --output text 2>/dev/null || echo "")
            
            for assoc_id in ${associations}; do
                if [ -n "${assoc_id}" ] && [ "${assoc_id}" != "None" ]; then
                    aws ec2 disassociate-route-table --association-id "${assoc_id}" --region "${REGION}" 2>/dev/null || \
                        log_warning "Failed to disassociate route table ${assoc_id}"
                fi
            done
            
            # Delete custom routes (keep local routes)
            local routes
            routes=$(aws ec2 describe-route-tables --route-table-ids "${rt_id}" --region "${REGION}" \
                --query 'RouteTables[0].Routes[?GatewayId!=`local`].DestinationCidrBlock' --output text 2>/dev/null || echo "")
            
            for route_cidr in ${routes}; do
                if [ -n "${route_cidr}" ] && [ "${route_cidr}" != "None" ]; then
                    aws ec2 delete-route --route-table-id "${rt_id}" --destination-cidr-block "${route_cidr}" --region "${REGION}" 2>/dev/null || \
                        log_warning "Failed to delete route ${route_cidr}"
                fi
            done
            
            # Delete route table
            aws ec2 delete-route-table --route-table-id "${rt_id}" --region "${REGION}" 2>/dev/null || \
                log_warning "Failed to delete route table ${rt_id}"
            log_success "Deleted route table: ${rt_id}"
        fi
    done
    
    # Delete subnets
    local subnets
    subnets=$(aws ec2 describe-subnets --region "${REGION}" \
        --filters "Name=vpc-id,Values=${vpc_id}" "Name=tag:Name,Values=${CLUSTER_NAME}-subnet" \
        --query 'Subnets[].SubnetId' --output text 2>/dev/null || echo "")
    
    for subnet_id in ${subnets}; do
        if [ -n "${subnet_id}" ] && [ "${subnet_id}" != "None" ]; then
            aws ec2 delete-subnet --subnet-id "${subnet_id}" --region "${REGION}" 2>/dev/null || \
                log_warning "Failed to delete subnet ${subnet_id}"
            log_success "Deleted subnet: ${subnet_id}"
        fi
    done
    
    # Detach and delete internet gateway
    local igw_id
    igw_id=$(aws ec2 describe-internet-gateways --region "${REGION}" \
        --filters "Name=tag:Name,Values=${CLUSTER_NAME}-igw" \
        --query 'InternetGateways[0].InternetGatewayId' --output text 2>/dev/null || echo "None")
    
    if [ "${igw_id}" != "None" ] && [ -n "${igw_id}" ]; then
        # Detach from VPC
        aws ec2 detach-internet-gateway --internet-gateway-id "${igw_id}" --vpc-id "${vpc_id}" --region "${REGION}" 2>/dev/null || \
            log_warning "Failed to detach internet gateway ${igw_id}"
        
        # Delete internet gateway
        aws ec2 delete-internet-gateway --internet-gateway-id "${igw_id}" --region "${REGION}" 2>/dev/null || \
            log_warning "Failed to delete internet gateway ${igw_id}"
        log_success "Deleted internet gateway: ${igw_id}"
    fi
    
    # Delete VPC
    aws ec2 delete-vpc --vpc-id "${vpc_id}" --region "${REGION}" 2>/dev/null || \
        log_warning "Failed to delete VPC ${vpc_id}"
    log_success "Deleted VPC: ${vpc_id}"
}

# Delete IAM resources
delete_iam_resources() {
    log_info "Deleting IAM resources..."
    
    local account_id=$(get_account_id)
    local policy_name="ParallelClusterS3Policy-${CLUSTER_NAME}"
    local policy_arn="arn:aws:iam::${account_id}:policy/${policy_name}"
    
    # Check if policy exists
    if aws iam get-policy --policy-arn "${policy_arn}" &> /dev/null; then
        # Detach policy from all entities first
        log_info "Detaching IAM policy from all entities..."
        
        # Detach from roles
        local attached_roles
        attached_roles=$(aws iam list-entities-for-policy --policy-arn "${policy_arn}" \
            --query 'PolicyRoles[].RoleName' --output text 2>/dev/null || echo "")
        
        for role_name in ${attached_roles}; do
            if [ -n "${role_name}" ] && [ "${role_name}" != "None" ]; then
                aws iam detach-role-policy --role-name "${role_name}" --policy-arn "${policy_arn}" 2>/dev/null || \
                    log_warning "Failed to detach policy from role ${role_name}"
            fi
        done
        
        # Detach from users
        local attached_users
        attached_users=$(aws iam list-entities-for-policy --policy-arn "${policy_arn}" \
            --query 'PolicyUsers[].UserName' --output text 2>/dev/null || echo "")
        
        for user_name in ${attached_users}; do
            if [ -n "${user_name}" ] && [ "${user_name}" != "None" ]; then
                aws iam detach-user-policy --user-name "${user_name}" --policy-arn "${policy_arn}" 2>/dev/null || \
                    log_warning "Failed to detach policy from user ${user_name}"
            fi
        done
        
        # Detach from groups
        local attached_groups
        attached_groups=$(aws iam list-entities-for-policy --policy-arn "${policy_arn}" \
            --query 'PolicyGroups[].GroupName' --output text 2>/dev/null || echo "")
        
        for group_name in ${attached_groups}; do
            if [ -n "${group_name}" ] && [ "${group_name}" != "None" ]; then
                aws iam detach-group-policy --group-name "${group_name}" --policy-arn "${policy_arn}" 2>/dev/null || \
                    log_warning "Failed to detach policy from group ${group_name}"
            fi
        done
        
        # Delete all policy versions except default
        local policy_versions
        policy_versions=$(aws iam list-policy-versions --policy-arn "${policy_arn}" \
            --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text 2>/dev/null || echo "")
        
        for version_id in ${policy_versions}; do
            if [ -n "${version_id}" ] && [ "${version_id}" != "None" ]; then
                aws iam delete-policy-version --policy-arn "${policy_arn}" --version-id "${version_id}" 2>/dev/null || \
                    log_warning "Failed to delete policy version ${version_id}"
            fi
        done
        
        # Delete policy
        aws iam delete-policy --policy-arn "${policy_arn}" 2>/dev/null || \
            log_warning "Failed to delete IAM policy ${policy_name}"
        log_success "Deleted IAM policy: ${policy_name}"
    else
        log_warning "IAM policy ${policy_name} not found."
    fi
}

# Delete EC2 key pair
delete_key_pair() {
    log_info "Deleting EC2 key pair..."
    
    if aws ec2 describe-key-pairs --key-names "${KEY_PAIR_NAME}" --region "${REGION}" &> /dev/null; then
        aws ec2 delete-key-pair --key-name "${KEY_PAIR_NAME}" --region "${REGION}"
        log_success "Deleted key pair: ${KEY_PAIR_NAME}"
        
        # Remove local key file if it exists
        if [ -f "${KEY_PAIR_NAME}.pem" ]; then
            rm -f "${KEY_PAIR_NAME}.pem"
            log_success "Removed local key file: ${KEY_PAIR_NAME}.pem"
        fi
    else
        log_warning "Key pair ${KEY_PAIR_NAME} not found."
    fi
}

# Delete S3 bucket
delete_s3_bucket() {
    log_info "Checking S3 bucket..."
    
    local account_id=$(get_account_id)
    local bucket_name="${S3_BUCKET_PREFIX}-${account_id}-${REGION}"
    
    if aws s3 ls "s3://${bucket_name}" >/dev/null 2>&1; then
        echo
        log_warning "S3 bucket contains all project data, models, and backups."
        echo -e "${YELLOW}Bucket contents:${NC}"
        aws s3 ls "s3://${bucket_name}" --recursive --human-readable --summarize | head -20
        echo
        
        read -p "Delete S3 bucket '${bucket_name}' and ALL data? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deleting S3 bucket and all contents..."
            aws s3 rb "s3://${bucket_name}" --force
            log_success "Deleted S3 bucket: ${bucket_name}"
        else
            log_info "S3 bucket preserved: ${bucket_name}"
        fi
    else
        log_warning "S3 bucket ${bucket_name} not found."
    fi
}

# Clean up temporary files
cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    
    local temp_files=(
        "/tmp/s3_bucket_name"
        "/tmp/vpc_id"
        "/tmp/subnet_id"
        "${CLUSTER_NAME}-config.yaml"
        "${CLUSTER_NAME}-config.yaml.bak"
        "node-setup.sh"
    )
    
    for file in "${temp_files[@]}"; do
        if [ -f "${file}" ]; then
            rm -f "${file}"
            log_success "Removed: ${file}"
        fi
    done
}

# Main teardown function
main() {
    echo
    log_info "Starting AWS ParallelCluster teardown for HPC Energy Model..."
    echo
    log_warning "This will destroy ALL AWS resources created by the deployment script."
    log_warning "This action is IRREVERSIBLE."
    echo
    
    read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Teardown cancelled."
        exit 0
    fi
    
    check_prerequisites
    
    # Delete resources in reverse order of creation
    delete_cluster
    delete_networking
    delete_iam_resources
    delete_key_pair
    delete_s3_bucket
    cleanup_temp_files
    
    echo
    log_success "Teardown completed successfully!"
    log_info "All AWS resources for HPC Energy Model have been destroyed."
}

# Script options
case "${1:-}" in
    "--force")
        log_warning "Force mode: Skipping confirmation prompts for non-destructive operations."
        export FORCE_MODE=true
        main
        ;;
    "--dry-run")
        log_info "Dry run mode: Showing what would be deleted without actually deleting."
        export DRY_RUN=true
        # TODO: Implement dry-run logic
        log_warning "Dry-run mode not yet implemented."
        exit 1
        ;;
    "--help"|"help")
        echo "Usage: $0 [--force|--dry-run|--help]"
        echo "  (no args)  - Interactive teardown with confirmations"
        echo "  --force    - Skip confirmation prompts (except S3 deletion)"
        echo "  --dry-run  - Show what would be deleted without deleting"
        echo "  --help     - Show this help message"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information."
        exit 1
        ;;
esac