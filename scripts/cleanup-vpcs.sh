#!/bin/bash

# Script to safely delete VPCs and their dependencies
# This script will delete all non-default VPCs in the account

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to delete VPC dependencies
delete_vpc_dependencies() {
    local vpc_id=$1
    log_info "Cleaning up dependencies for VPC: $vpc_id"
    
    # Delete NAT Gateways
    local nat_gateways=$(aws ec2 describe-nat-gateways --profile personal-account --filter "Name=vpc-id,Values=$vpc_id" --query 'NatGateways[*].NatGatewayId' --output text)
    for nat_gw in $nat_gateways; do
        if [ "$nat_gw" != "None" ] && [ -n "$nat_gw" ]; then
            log_info "Deleting NAT Gateway: $nat_gw"
            aws ec2 delete-nat-gateway --profile personal-account --nat-gateway-id "$nat_gw" || true
        fi
    done
    
    # Delete Internet Gateways
    local igws=$(aws ec2 describe-internet-gateways --profile personal-account --filters "Name=attachment.vpc-id,Values=$vpc_id" --query 'InternetGateways[*].InternetGatewayId' --output text)
    for igw in $igws; do
        if [ "$igw" != "None" ] && [ -n "$igw" ]; then
            log_info "Detaching and deleting Internet Gateway: $igw"
            aws ec2 detach-internet-gateway --profile personal-account --internet-gateway-id "$igw" --vpc-id "$vpc_id" || true
            aws ec2 delete-internet-gateway --profile personal-account --internet-gateway-id "$igw" || true
        fi
    done
    
    # Delete Subnets
    local subnets=$(aws ec2 describe-subnets --profile personal-account --filters "Name=vpc-id,Values=$vpc_id" --query 'Subnets[*].SubnetId' --output text)
    for subnet in $subnets; do
        if [ "$subnet" != "None" ] && [ -n "$subnet" ]; then
            log_info "Deleting subnet: $subnet"
            aws ec2 delete-subnet --profile personal-account --subnet-id "$subnet" || true
        fi
    done
    
    # Delete Route Tables (except main)
    local route_tables=$(aws ec2 describe-route-tables --profile personal-account --filters "Name=vpc-id,Values=$vpc_id" --query 'RouteTables[?Associations[0].Main!=`true`].RouteTableId' --output text)
    for rt in $route_tables; do
        if [ "$rt" != "None" ] && [ -n "$rt" ]; then
            log_info "Deleting route table: $rt"
            aws ec2 delete-route-table --profile personal-account --route-table-id "$rt" || true
        fi
    done
    
    # Delete Security Groups (except default)
    local security_groups=$(aws ec2 describe-security-groups --profile personal-account --filters "Name=vpc-id,Values=$vpc_id" --query 'SecurityGroups[?GroupName!=`default`].GroupId' --output text)
    for sg in $security_groups; do
        if [ "$sg" != "None" ] && [ -n "$sg" ]; then
            log_info "Deleting security group: $sg"
            aws ec2 delete-security-group --profile personal-account --group-id "$sg" || true
        fi
    done
    
    # Delete Network ACLs (except default)
    local network_acls=$(aws ec2 describe-network-acls --profile personal-account --filters "Name=vpc-id,Values=$vpc_id" --query 'NetworkAcls[?IsDefault!=`true`].NetworkAclId' --output text)
    for acl in $network_acls; do
        if [ "$acl" != "None" ] && [ -n "$acl" ]; then
            log_info "Deleting network ACL: $acl"
            aws ec2 delete-network-acl --profile personal-account --network-acl-id "$acl" || true
        fi
    done
}

# Function to delete a VPC
delete_vpc() {
    local vpc_id=$1
    log_info "Deleting VPC: $vpc_id"
    
    # Delete dependencies first
    delete_vpc_dependencies "$vpc_id"
    
    # Wait a bit for resources to be cleaned up
    sleep 10
    
    # Delete the VPC
    aws ec2 delete-vpc --profile personal-account --vpc-id "$vpc_id" || {
        log_error "Failed to delete VPC $vpc_id. It may still have dependencies."
        return 1
    }
    
    log_success "VPC $vpc_id deleted successfully"
}

# Main execution
log_info "Starting VPC cleanup..."

# Get all non-default VPCs
vpc_ids=$(aws ec2 describe-vpcs --profile personal-account --query 'Vpcs[?IsDefault==`false`].VpcId' --output text)

if [ -z "$vpc_ids" ] || [ "$vpc_ids" = "None" ]; then
    log_info "No non-default VPCs found to delete."
    exit 0
fi

log_info "Found VPCs to delete: $vpc_ids"

# Delete each VPC
for vpc_id in $vpc_ids; do
    if [ -n "$vpc_id" ] && [ "$vpc_id" != "None" ]; then
        delete_vpc "$vpc_id"
    fi
done

log_success "VPC cleanup completed!"