#!/bin/bash

# HPC Energy Model Deployment Script
# Automated deployment for different platforms and configurations

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/deployment.log"
CONFIG_FILE="$PROJECT_ROOT/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add cleanup tasks here if needed
}

trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
HPC Energy Model Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    deploy          Deploy the complete system
    start           Start all services
    stop            Stop all services
    restart         Restart all services
    status          Show service status
    logs            Show service logs
    update          Update and restart services
    backup          Create system backup
    restore         Restore from backup
    clean           Clean up unused resources
    health          Check system health

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -e, --env FILE  Use specific environment file
    -p, --platform  Target platform (baremetal|vmware|kvm|hyperv)
    -s, --ssl       Enable SSL/TLS
    -d, --dev       Development mode
    --no-logs       Disable advanced logging
    --quick         Quick deployment (skip optional components)

Examples:
    $0 deploy --platform baremetal --ssl
    $0 start --verbose
    $0 logs --follow
    $0 backup --compress

EOF
}

# Parse command line arguments
VERBOSE=false
PLATFORM="baremetal"
SSL_ENABLED=false
DEV_MODE=false
ENABLE_LOGS=true
QUICK_DEPLOY=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -e|--env)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -s|--ssl)
            SSL_ENABLED=true
            shift
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        --no-logs)
            ENABLE_LOGS=false
            shift
            ;;
        --quick)
            QUICK_DEPLOY=true
            shift
            ;;
        deploy|start|stop|restart|status|logs|update|backup|restore|clean|health)
            COMMAND="$1"
            shift
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/backups"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/infrastructure/nginx/ssl"
    mkdir -p "$PROJECT_ROOT/infrastructure/prometheus/data"
    mkdir -p "$PROJECT_ROOT/infrastructure/grafana/data"
    mkdir -p "$PROJECT_ROOT/infrastructure/timescaledb/data"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running. Please start Docker first."
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then
        log_warn "Low disk space detected. At least 10GB is recommended."
    fi
    
    # Check available memory (minimum 4GB)
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 4096 ]]; then
        log_warn "Low memory detected. At least 4GB is recommended."
    fi
    
    log_success "Prerequisites check completed"
}

# Setup environment configuration
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "Creating environment configuration from template..."
        cp "$PROJECT_ROOT/.env.example" "$CONFIG_FILE"
    fi
    
    # Update platform-specific settings
    sed -i "s/HYPERVISOR_TYPE=.*/HYPERVISOR_TYPE=$PLATFORM/g" "$CONFIG_FILE"
    
    # Update SSL settings
    if [[ "$SSL_ENABLED" == "true" ]]; then
        sed -i "s/SSL_ENABLED=.*/SSL_ENABLED=true/g" "$CONFIG_FILE"
        generate_ssl_certificates
    fi
    
    # Update development mode settings
    if [[ "$DEV_MODE" == "true" ]]; then
        sed -i "s/API_ENV=.*/API_ENV=development/g" "$CONFIG_FILE"
        sed -i "s/API_DEBUG=.*/API_DEBUG=true/g" "$CONFIG_FILE"
        sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/g" "$CONFIG_FILE"
    fi
    
    # Generate secure passwords if not set
    generate_passwords
    
    log_success "Environment configuration completed"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    local ssl_dir="$PROJECT_ROOT/infrastructure/nginx/ssl"
    
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$ssl_dir/key.pem" \
            -out "$ssl_dir/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            &> /dev/null
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Generate secure passwords
generate_passwords() {
    log_info "Generating secure passwords..."
    
    # Generate random passwords if not set
    if ! grep -q "TIMESCALE_PASS=.*[a-zA-Z0-9]" "$CONFIG_FILE"; then
        local db_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        sed -i "s/TIMESCALE_PASS=.*/TIMESCALE_PASS=$db_password/g" "$CONFIG_FILE"
    fi
    
    if ! grep -q "REDIS_PASSWORD=.*[a-zA-Z0-9]" "$CONFIG_FILE"; then
        local redis_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$redis_password/g" "$CONFIG_FILE"
    fi
    
    if ! grep -q "GRAFANA_ADMIN_PASSWORD=.*[a-zA-Z0-9]" "$CONFIG_FILE"; then
        local grafana_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        sed -i "s/GRAFANA_ADMIN_PASSWORD=.*/GRAFANA_ADMIN_PASSWORD=$grafana_password/g" "$CONFIG_FILE"
    fi
    
    if ! grep -q "JWT_SECRET_KEY=.*[a-zA-Z0-9]" "$CONFIG_FILE"; then
        local jwt_secret=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
        sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$jwt_secret/g" "$CONFIG_FILE"
    fi
}

# Platform-specific setup
setup_platform() {
    log_info "Setting up platform-specific configuration for: $PLATFORM"
    
    case $PLATFORM in
        baremetal)
            setup_baremetal
            ;;
        vmware)
            setup_vmware
            ;;
        kvm)
            setup_kvm
            ;;
        hyperv)
            setup_hyperv
            ;;
        *)
            log_warn "Unknown platform: $PLATFORM. Using default configuration."
            ;;
    esac
}

setup_baremetal() {
    log_info "Configuring for bare-metal deployment..."
    # Add bare-metal specific configuration
    echo "# Bare-metal specific configuration" >> "$CONFIG_FILE"
    echo "MONITORING_INTERFACE=eth0" >> "$CONFIG_FILE"
}

setup_vmware() {
    log_info "Configuring for VMware deployment..."
    # Add VMware specific configuration
    echo "# VMware specific configuration" >> "$CONFIG_FILE"
    echo "VMWARE_TOOLS_ENABLED=true" >> "$CONFIG_FILE"
}

setup_kvm() {
    log_info "Configuring for KVM deployment..."
    # Add KVM specific configuration
    echo "# KVM specific configuration" >> "$CONFIG_FILE"
    echo "VIRTIO_ENABLED=true" >> "$CONFIG_FILE"
}

setup_hyperv() {
    log_info "Configuring for Hyper-V deployment..."
    # Add Hyper-V specific configuration
    echo "# Hyper-V specific configuration" >> "$CONFIG_FILE"
    echo "HYPERV_INTEGRATION_ENABLED=true" >> "$CONFIG_FILE"
}

# Deploy services
deploy_services() {
    log_info "Deploying HPC Energy Model services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker-compose pull
    
    # Build custom images
    log_info "Building custom images..."
    docker-compose build
    
    # Start services
    log_info "Starting services..."
    if [[ "$ENABLE_LOGS" == "true" ]]; then
        docker-compose up -d
    else
        docker-compose -f docker-compose.yml -f docker-compose.no-logs.yml up -d
    fi
    
    # Wait for services to be ready
    wait_for_services
    
    # Initialize database
    initialize_database
    
    # Import Grafana dashboards
    import_dashboards
    
    log_success "Deployment completed successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local max_attempts=60
    local attempt=0
    
    # Wait for TimescaleDB
    while ! docker-compose exec -T timescaledb pg_isready -U postgres &> /dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "TimescaleDB failed to start within expected time"
        fi
        sleep 5
    done
    log_success "TimescaleDB is ready"
    
    # Wait for Redis
    while ! docker-compose exec -T redis redis-cli ping &> /dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "Redis failed to start within expected time"
        fi
        sleep 5
    done
    log_success "Redis is ready"
    
    # Wait for API
    while ! curl -f http://localhost:5000/health &> /dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "Energy API failed to start within expected time"
        fi
        sleep 5
    done
    log_success "Energy API is ready"
    
    # Wait for Grafana
    while ! curl -f http://localhost:3000/api/health &> /dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "Grafana failed to start within expected time"
        fi
        sleep 5
    done
    log_success "Grafana is ready"
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    # Check if database is already initialized
    if docker-compose exec -T timescaledb psql -U postgres -d hpc_energy -c "SELECT 1 FROM hpc_energy.job_metrics LIMIT 1;" &> /dev/null; then
        log_info "Database already initialized"
        return
    fi
    
    # Run initialization script
    docker-compose exec -T timescaledb psql -U postgres -d hpc_energy -f /docker-entrypoint-initdb.d/init.sql
    
    log_success "Database initialized"
}

# Import Grafana dashboards
import_dashboards() {
    log_info "Importing Grafana dashboards..."
    
    # Wait a bit more for Grafana to be fully ready
    sleep 10
    
    # Import dashboards using Grafana API
    local grafana_url="http://localhost:3000"
    local admin_password=$(grep GRAFANA_ADMIN_PASSWORD "$CONFIG_FILE" | cut -d'=' -f2)
    
    for dashboard in "$PROJECT_ROOT"/dashboards/*.json; do
        if [[ -f "$dashboard" ]]; then
            log_info "Importing dashboard: $(basename "$dashboard")"
            curl -X POST \
                -H "Content-Type: application/json" \
                -u "admin:$admin_password" \
                -d @"$dashboard" \
                "$grafana_url/api/dashboards/db" &> /dev/null || log_warn "Failed to import $(basename "$dashboard")"
        fi
    done
    
    log_success "Dashboards imported"
}

# Show service status
show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    log_info "\nService Health:"
    
    # Check API health
    if curl -f http://localhost:5000/health &> /dev/null; then
        log_success "Energy API: Healthy"
    else
        log_error "Energy API: Unhealthy"
    fi
    
    # Check Grafana health
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana: Healthy"
    else
        log_error "Grafana: Unhealthy"
    fi
    
    # Check Prometheus health
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus: Healthy"
    else
        log_error "Prometheus: Unhealthy"
    fi
}

# Show logs
show_logs() {
    if [[ "$1" == "--follow" ]] || [[ "$1" == "-f" ]]; then
        docker-compose logs -f
    else
        docker-compose logs --tail=100
    fi
}

# Create backup
create_backup() {
    log_info "Creating system backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    log_info "Backing up database..."
    docker-compose exec -T timescaledb pg_dump -U postgres hpc_energy > "$backup_dir/database.sql"
    
    # Backup configuration
    log_info "Backing up configuration..."
    cp "$CONFIG_FILE" "$backup_dir/"
    cp -r "$PROJECT_ROOT/infrastructure" "$backup_dir/"
    
    # Backup Grafana data
    log_info "Backing up Grafana data..."
    docker-compose exec -T grafana tar -czf - /var/lib/grafana > "$backup_dir/grafana_data.tar.gz"
    
    # Create archive if requested
    if [[ "$1" == "--compress" ]]; then
        log_info "Compressing backup..."
        tar -czf "$backup_dir.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
        rm -rf "$backup_dir"
        log_success "Backup created: $backup_dir.tar.gz"
    else
        log_success "Backup created: $backup_dir"
    fi
}

# Clean up resources
clean_resources() {
    log_info "Cleaning up unused resources..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove old log files
    find "$PROJECT_ROOT/logs" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Remove old backups
    find "$PROJECT_ROOT/backups" -name "*.tar.gz" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting HPC Energy Model deployment script"
    log_info "Platform: $PLATFORM, SSL: $SSL_ENABLED, Dev Mode: $DEV_MODE"
    
    # Create directories
    create_directories
    
    case $COMMAND in
        deploy)
            check_prerequisites
            setup_environment
            setup_platform
            deploy_services
            show_status
            log_success "Deployment completed! Access Grafana at http://localhost:3000"
            ;;
        start)
            cd "$PROJECT_ROOT"
            docker-compose start
            log_success "Services started"
            ;;
        stop)
            cd "$PROJECT_ROOT"
            docker-compose stop
            log_success "Services stopped"
            ;;
        restart)
            cd "$PROJECT_ROOT"
            docker-compose restart
            log_success "Services restarted"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$@"
            ;;
        update)
            cd "$PROJECT_ROOT"
            docker-compose pull
            docker-compose up -d
            log_success "Services updated"
            ;;
        backup)
            create_backup "$@"
            ;;
        clean)
            clean_resources
            ;;
        health)
            show_status
            ;;
        "")
            log_error "No command specified. Use --help for usage information."
            exit 1
            ;;
        *)
            error_exit "Unknown command: $COMMAND"
            ;;
    esac
}

# Run main function
main "$@"