# HPC Energy Model Deployment Script for Windows
# PowerShell script for automated deployment on Windows systems

param(
    [Parameter(Position=0)]
    [ValidateSet('deploy', 'start', 'stop', 'restart', 'status', 'logs', 'update', 'backup', 'restore', 'clean', 'health')]
    [string]$Command = '',
    
    [Parameter()]
    [ValidateSet('baremetal', 'vmware', 'kvm', 'hyperv')]
    [string]$Platform = 'baremetal',
    
    [Parameter()]
    [string]$EnvFile = '.env',
    
    [Parameter()]
    [switch]$SSL,
    
    [Parameter()]
    [switch]$Dev,
    
    [Parameter()]
    [switch]$NoLogs,
    
    [Parameter()]
    [switch]$Quick,
    
    [Parameter()]
    [switch]$Verbose,
    
    [Parameter()]
    [switch]$Follow,
    
    [Parameter()]
    [switch]$Compress,
    
    [Parameter()]
    [switch]$Help
)

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogFile = Join-Path $ProjectRoot "logs\deployment.log"
$ConfigFile = Join-Path $ProjectRoot $EnvFile

# Ensure logs directory exists
$LogsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Logging functions
function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $LogMessage = "[$Timestamp] [$Level] $Message"
    
    # Color coding
    switch ($Level) {
        'INFO' { Write-Host $LogMessage -ForegroundColor Blue }
        'WARN' { Write-Host $LogMessage -ForegroundColor Yellow }
        'ERROR' { Write-Host $LogMessage -ForegroundColor Red }
        'SUCCESS' { Write-Host $LogMessage -ForegroundColor Green }
        default { Write-Host $LogMessage }
    }
    
    # Write to log file
    Add-Content -Path $LogFile -Value $LogMessage
}

function Write-Info { param([string]$Message) Write-Log $Message 'INFO' }
function Write-Warn { param([string]$Message) Write-Log $Message 'WARN' }
function Write-Error { param([string]$Message) Write-Log $Message 'ERROR' }
function Write-Success { param([string]$Message) Write-Log $Message 'SUCCESS' }

# Error handling
function Exit-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

# Help function
function Show-Help {
    @"
HPC Energy Model Deployment Script for Windows

Usage: .\deploy.ps1 [COMMAND] [OPTIONS]

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
    -Platform       Target platform (baremetal|vmware|kvm|hyperv)
    -EnvFile        Use specific environment file
    -SSL            Enable SSL/TLS
    -Dev            Development mode
    -NoLogs         Disable advanced logging
    -Quick          Quick deployment (skip optional components)
    -Verbose        Enable verbose output
    -Follow         Follow logs (for logs command)
    -Compress       Compress backup (for backup command)
    -Help           Show this help message

Examples:
    .\deploy.ps1 deploy -Platform baremetal -SSL
    .\deploy.ps1 start -Verbose
    .\deploy.ps1 logs -Follow
    .\deploy.ps1 backup -Compress

"@
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Info "Docker found: $dockerVersion"
    }
    catch {
        Exit-WithError "Docker is not installed or not in PATH. Please install Docker Desktop for Windows."
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version
        Write-Info "Docker Compose found: $composeVersion"
    }
    catch {
        Exit-WithError "Docker Compose is not installed or not in PATH."
    }
    
    # Check Docker daemon
    try {
        docker info | Out-Null
        Write-Info "Docker daemon is running"
    }
    catch {
        Exit-WithError "Docker daemon is not running. Please start Docker Desktop."
    }
    
    # Check available disk space (minimum 10GB)
    $drive = (Get-Location).Drive
    $freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($drive.Name)'").FreeSpace / 1GB
    if ($freeSpace -lt 10) {
        Write-Warn "Low disk space detected: $([math]::Round($freeSpace, 2))GB available. At least 10GB is recommended."
    }
    
    # Check available memory (minimum 4GB)
    $totalMemory = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    if ($totalMemory -lt 4) {
        Write-Warn "Low memory detected: $([math]::Round($totalMemory, 2))GB available. At least 4GB is recommended."
    }
    
    Write-Success "Prerequisites check completed"
}

# Create necessary directories
function New-ProjectDirectories {
    Write-Info "Creating necessary directories..."
    
    $directories = @(
        "logs",
        "data",
        "backups",
        "models",
        "infrastructure\nginx\ssl",
        "infrastructure\prometheus\data",
        "infrastructure\grafana\data",
        "infrastructure\timescaledb\data"
    )
    
    foreach ($dir in $directories) {
        $fullPath = Join-Path $ProjectRoot $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Setup environment configuration
function Set-Environment {
    Write-Info "Setting up environment configuration..."
    
    $exampleFile = Join-Path $ProjectRoot ".env.example"
    
    if (-not (Test-Path $ConfigFile)) {
        Write-Info "Creating environment configuration from template..."
        Copy-Item $exampleFile $ConfigFile
    }
    
    # Update platform-specific settings
    (Get-Content $ConfigFile) -replace 'HYPERVISOR_TYPE=.*', "HYPERVISOR_TYPE=$Platform" | Set-Content $ConfigFile
    
    # Update SSL settings
    if ($SSL) {
        (Get-Content $ConfigFile) -replace 'SSL_ENABLED=.*', 'SSL_ENABLED=true' | Set-Content $ConfigFile
        New-SSLCertificates
    }
    
    # Update development mode settings
    if ($Dev) {
        (Get-Content $ConfigFile) -replace 'API_ENV=.*', 'API_ENV=development' | Set-Content $ConfigFile
        (Get-Content $ConfigFile) -replace 'API_DEBUG=.*', 'API_DEBUG=true' | Set-Content $ConfigFile
        (Get-Content $ConfigFile) -replace 'LOG_LEVEL=.*', 'LOG_LEVEL=DEBUG' | Set-Content $ConfigFile
    }
    
    # Generate secure passwords
    New-SecurePasswords
    
    Write-Success "Environment configuration completed"
}

# Generate SSL certificates
function New-SSLCertificates {
    Write-Info "Generating SSL certificates..."
    
    $sslDir = Join-Path $ProjectRoot "infrastructure\nginx\ssl"
    $certFile = Join-Path $sslDir "cert.pem"
    $keyFile = Join-Path $sslDir "key.pem"
    
    if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
        # Use OpenSSL if available, otherwise use PowerShell certificate creation
        try {
            & openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout $keyFile -out $certFile -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" 2>$null
            Write-Success "SSL certificates generated using OpenSSL"
        }
        catch {
            # Fallback to PowerShell certificate creation
            $cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "cert:\LocalMachine\My" -KeyAlgorithm RSA -KeyLength 2048 -Provider "Microsoft RSA SChannel Cryptographic Provider" -KeyExportPolicy Exportable -KeyUsage DigitalSignature,KeyEncipherment -Type SSLServerAuthentication
            
            # Export certificate
            $certPassword = ConvertTo-SecureString -String "password" -Force -AsPlainText
            Export-PfxCertificate -Cert $cert -FilePath (Join-Path $sslDir "cert.pfx") -Password $certPassword | Out-Null
            
            # Convert to PEM format (requires OpenSSL or manual conversion)
            Write-Warn "SSL certificate created in Windows certificate store. Manual conversion to PEM format may be required."
        }
    }
    else {
        Write-Info "SSL certificates already exist"
    }
}

# Generate secure passwords
function New-SecurePasswords {
    Write-Info "Generating secure passwords..."
    
    $content = Get-Content $ConfigFile
    
    # Generate random passwords if not set
    if ($content -notmatch 'TIMESCALE_PASS=.*[a-zA-Z0-9]') {
        $dbPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 25 | ForEach-Object {[char]$_})
        $content = $content -replace 'TIMESCALE_PASS=.*', "TIMESCALE_PASS=$dbPassword"
    }
    
    if ($content -notmatch 'REDIS_PASSWORD=.*[a-zA-Z0-9]') {
        $redisPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 25 | ForEach-Object {[char]$_})
        $content = $content -replace 'REDIS_PASSWORD=.*', "REDIS_PASSWORD=$redisPassword"
    }
    
    if ($content -notmatch 'GRAFANA_ADMIN_PASSWORD=.*[a-zA-Z0-9]') {
        $grafanaPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 25 | ForEach-Object {[char]$_})
        $content = $content -replace 'GRAFANA_ADMIN_PASSWORD=.*', "GRAFANA_ADMIN_PASSWORD=$grafanaPassword"
    }
    
    if ($content -notmatch 'JWT_SECRET_KEY=.*[a-zA-Z0-9]') {
        $jwtSecret = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 50 | ForEach-Object {[char]$_})
        $content = $content -replace 'JWT_SECRET_KEY=.*', "JWT_SECRET_KEY=$jwtSecret"
    }
    
    Set-Content $ConfigFile $content
}

# Platform-specific setup
function Set-Platform {
    Write-Info "Setting up platform-specific configuration for: $Platform"
    
    switch ($Platform) {
        'baremetal' { Set-BareMetal }
        'vmware' { Set-VMware }
        'kvm' { Set-KVM }
        'hyperv' { Set-HyperV }
        default { Write-Warn "Unknown platform: $Platform. Using default configuration." }
    }
}

function Set-BareMetal {
    Write-Info "Configuring for bare-metal deployment..."
    Add-Content $ConfigFile "`n# Bare-metal specific configuration"
    Add-Content $ConfigFile "MONITORING_INTERFACE=Ethernet"
}

function Set-VMware {
    Write-Info "Configuring for VMware deployment..."
    Add-Content $ConfigFile "`n# VMware specific configuration"
    Add-Content $ConfigFile "VMWARE_TOOLS_ENABLED=true"
}

function Set-KVM {
    Write-Info "Configuring for KVM deployment..."
    Add-Content $ConfigFile "`n# KVM specific configuration"
    Add-Content $ConfigFile "VIRTIO_ENABLED=true"
}

function Set-HyperV {
    Write-Info "Configuring for Hyper-V deployment..."
    Add-Content $ConfigFile "`n# Hyper-V specific configuration"
    Add-Content $ConfigFile "HYPERV_INTEGRATION_ENABLED=true"
}

# Deploy services
function Start-Deployment {
    Write-Info "Deploying HPC Energy Model services..."
    
    Set-Location $ProjectRoot
    
    # Pull latest images
    Write-Info "Pulling Docker images..."
    & docker-compose pull
    
    # Build custom images
    Write-Info "Building custom images..."
    & docker-compose build
    
    # Start services
    Write-Info "Starting services..."
    if ($NoLogs) {
        & docker-compose -f docker-compose.yml -f docker-compose.no-logs.yml up -d
    }
    else {
        & docker-compose up -d
    }
    
    # Wait for services to be ready
    Wait-ForServices
    
    # Initialize database
    Initialize-Database
    
    # Import Grafana dashboards
    Import-Dashboards
    
    Write-Success "Deployment completed successfully"
}

# Wait for services to be ready
function Wait-ForServices {
    Write-Info "Waiting for services to be ready..."
    
    $maxAttempts = 60
    $attempt = 0
    
    # Wait for TimescaleDB
    do {
        $attempt++
        Start-Sleep -Seconds 5
        try {
            & docker-compose exec -T timescaledb pg_isready -U postgres 2>$null
            $dbReady = $LASTEXITCODE -eq 0
        }
        catch {
            $dbReady = $false
        }
    } while (-not $dbReady -and $attempt -lt $maxAttempts)
    
    if (-not $dbReady) {
        Exit-WithError "TimescaleDB failed to start within expected time"
    }
    Write-Success "TimescaleDB is ready"
    
    # Wait for Redis
    $attempt = 0
    do {
        $attempt++
        Start-Sleep -Seconds 5
        try {
            & docker-compose exec -T redis redis-cli ping 2>$null
            $redisReady = $LASTEXITCODE -eq 0
        }
        catch {
            $redisReady = $false
        }
    } while (-not $redisReady -and $attempt -lt $maxAttempts)
    
    if (-not $redisReady) {
        Exit-WithError "Redis failed to start within expected time"
    }
    Write-Success "Redis is ready"
    
    # Wait for API
    $attempt = 0
    do {
        $attempt++
        Start-Sleep -Seconds 5
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5
            $apiReady = $response.StatusCode -eq 200
        }
        catch {
            $apiReady = $false
        }
    } while (-not $apiReady -and $attempt -lt $maxAttempts)
    
    if (-not $apiReady) {
        Exit-WithError "Energy API failed to start within expected time"
    }
    Write-Success "Energy API is ready"
    
    # Wait for Grafana
    $attempt = 0
    do {
        $attempt++
        Start-Sleep -Seconds 5
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5
            $grafanaReady = $response.StatusCode -eq 200
        }
        catch {
            $grafanaReady = $false
        }
    } while (-not $grafanaReady -and $attempt -lt $maxAttempts)
    
    if (-not $grafanaReady) {
        Exit-WithError "Grafana failed to start within expected time"
    }
    Write-Success "Grafana is ready"
}

# Initialize database
function Initialize-Database {
    Write-Info "Initializing database..."
    
    # Check if database is already initialized
    try {
        & docker-compose exec -T timescaledb psql -U postgres -d hpc_energy -c "SELECT 1 FROM hpc_energy.job_metrics LIMIT 1;" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Database already initialized"
            return
        }
    }
    catch {
        # Database not initialized, continue
    }
    
    # Run initialization script
    & docker-compose exec -T timescaledb psql -U postgres -d hpc_energy -f /docker-entrypoint-initdb.d/init.sql
    
    Write-Success "Database initialized"
}

# Import Grafana dashboards
function Import-Dashboards {
    Write-Info "Importing Grafana dashboards..."
    
    # Wait a bit more for Grafana to be fully ready
    Start-Sleep -Seconds 10
    
    # Get admin password
    $adminPassword = (Get-Content $ConfigFile | Where-Object { $_ -match 'GRAFANA_ADMIN_PASSWORD=' }) -replace 'GRAFANA_ADMIN_PASSWORD=', ''
    
    # Import dashboards
    $dashboardsDir = Join-Path $ProjectRoot "dashboards"
    if (Test-Path $dashboardsDir) {
        Get-ChildItem -Path $dashboardsDir -Filter "*.json" | ForEach-Object {
            Write-Info "Importing dashboard: $($_.Name)"
            try {
                $dashboardContent = Get-Content $_.FullName -Raw
                $credentials = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:$adminPassword"))
                $headers = @{
                    'Authorization' = "Basic $credentials"
                    'Content-Type' = 'application/json'
                }
                
                Invoke-RestMethod -Uri "http://localhost:3000/api/dashboards/db" -Method Post -Body $dashboardContent -Headers $headers
            }
            catch {
                Write-Warn "Failed to import $($_.Name): $($_.Exception.Message)"
            }
        }
    }
    
    Write-Success "Dashboards imported"
}

# Show service status
function Show-Status {
    Write-Info "Service Status:"
    & docker-compose ps
    
    Write-Info "`nService Health:"
    
    # Check API health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Energy API: Healthy"
        }
        else {
            Write-Error "Energy API: Unhealthy"
        }
    }
    catch {
        Write-Error "Energy API: Unhealthy"
    }
    
    # Check Grafana health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Grafana: Healthy"
        }
        else {
            Write-Error "Grafana: Unhealthy"
        }
    }
    catch {
        Write-Error "Grafana: Unhealthy"
    }
    
    # Check Prometheus health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Prometheus: Healthy"
        }
        else {
            Write-Error "Prometheus: Unhealthy"
        }
    }
    catch {
        Write-Error "Prometheus: Unhealthy"
    }
}

# Show logs
function Show-Logs {
    if ($Follow) {
        & docker-compose logs -f
    }
    else {
        & docker-compose logs --tail=100
    }
}

# Create backup
function New-Backup {
    Write-Info "Creating system backup..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = Join-Path $ProjectRoot "backups\$timestamp"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # Backup database
    Write-Info "Backing up database..."
    & docker-compose exec -T timescaledb pg_dump -U postgres hpc_energy > "$backupDir\database.sql"
    
    # Backup configuration
    Write-Info "Backing up configuration..."
    Copy-Item $ConfigFile $backupDir
    Copy-Item -Path (Join-Path $ProjectRoot "infrastructure") -Destination $backupDir -Recurse
    
    # Backup Grafana data
    Write-Info "Backing up Grafana data..."
    & docker-compose exec -T grafana tar -czf - /var/lib/grafana > "$backupDir\grafana_data.tar.gz"
    
    # Create archive if requested
    if ($Compress) {
        Write-Info "Compressing backup..."
        $archivePath = "$backupDir.zip"
        Compress-Archive -Path $backupDir -DestinationPath $archivePath
        Remove-Item -Path $backupDir -Recurse -Force
        Write-Success "Backup created: $archivePath"
    }
    else {
        Write-Success "Backup created: $backupDir"
    }
}

# Clean up resources
function Remove-UnusedResources {
    Write-Info "Cleaning up unused resources..."
    
    # Remove unused Docker images
    & docker image prune -f
    
    # Remove unused volumes
    & docker volume prune -f
    
    # Remove old log files
    $logsDir = Join-Path $ProjectRoot "logs"
    Get-ChildItem -Path $logsDir -Filter "*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | Remove-Item -Force
    
    # Remove old backups
    $backupsDir = Join-Path $ProjectRoot "backups"
    if (Test-Path $backupsDir) {
        Get-ChildItem -Path $backupsDir -Filter "*.zip" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force
    }
    
    Write-Success "Cleanup completed"
}

# Main execution
function Main {
    if ($Help -or $Command -eq '') {
        Show-Help
        return
    }
    
    Write-Info "Starting HPC Energy Model deployment script"
    Write-Info "Platform: $Platform, SSL: $SSL, Dev Mode: $Dev"
    
    # Set verbose mode
    if ($Verbose) {
        $VerbosePreference = 'Continue'
    }
    
    # Create directories
    New-ProjectDirectories
    
    switch ($Command) {
        'deploy' {
            Test-Prerequisites
            Set-Environment
            Set-Platform
            Start-Deployment
            Show-Status
            Write-Success "Deployment completed! Access Grafana at http://localhost:3000"
        }
        'start' {
            Set-Location $ProjectRoot
            & docker-compose start
            Write-Success "Services started"
        }
        'stop' {
            Set-Location $ProjectRoot
            & docker-compose stop
            Write-Success "Services stopped"
        }
        'restart' {
            Set-Location $ProjectRoot
            & docker-compose restart
            Write-Success "Services restarted"
        }
        'status' {
            Show-Status
        }
        'logs' {
            Show-Logs
        }
        'update' {
            Set-Location $ProjectRoot
            & docker-compose pull
            & docker-compose up -d
            Write-Success "Services updated"
        }
        'backup' {
            New-Backup
        }
        'clean' {
            Remove-UnusedResources
        }
        'health' {
            Show-Status
        }
        default {
            Exit-WithError "Unknown command: $Command"
        }
    }
}

# Run main function
Main